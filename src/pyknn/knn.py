from nltk import word_tokenize, PorterStemmer, edit_distance
from nltk.corpus import wordnet as wn
from pyprofile import timed
import numpy as np
import pickle
import logging as logger

#%% utility functions.

## utility function
def cosine_distance(t1, t2):
    """
    Perform the cosine distance between params t1 and t2.

    We expect that t1 will have the shape (n, 1) a row vector, and that t2 will be one or more stacked row vectors.
    """
    if isinstance(t1, list):
        t1 = np.array(t1)

    if isinstance(t2, list):
        t2 = np.array(t2)
        #fix the shape

    #this is batch 1    
    if t1.shape == t2.shape:
        t2 = t2.reshape((1, t2.shape[0]))
        
    dp = np.dot(t1, t2.T)
    norm_t1 = np.linalg.norm(t1)
    norm_t2 = np.linalg.norm(t2, axis=1)
    epsilon = 1e-9

    #I use to avoid division by 0 1e-9
    cd =  dp/(norm_t1 * norm_t2 + epsilon)

    return cd.T


#%% Framework classes.


class Embedder():
    """
    This class acts as an `interface` and it defines the methods that need to be implemented to use the given embedder for KNN search.
    """

    _supports_all_words_embeds = True 
    
    def embed_query(self, query, **kwargs):
        """
        Use the embedding model to embed a query, if _supports_all_words_embeds returns True, then this method must be prepared to return 
        a dictionary of embeddings using every word of the query when the flag is passed as a keyword argument.
        """
        raise NotImplementedError("This method must be overwritten to be useful.")

    @property
    def embed_length(self):
        """
        Returns the length of the embedding tensors. The index uses this value to calculate hash values.
        """
        raise NotImplementedError("This method must be overwritten to be useful.")

    @property
    def zeros(self):
        """
        Create a tensor with zeros.
        """
        raise NotImplementedError("This method must be overwritten to be useful.")

    @property
    def supports_all_words_embeds(self) -> bool:
        """
        Define wether the embedder will support embedding every single word in the search query. This functionality typically
         helps with accuracy at the cost of performance because terms will map to multiple hash values increasing the likelihood 
         of a perfect match.
        """
        return self._supports_all_words_embeds
    
    @property
    def vocab(self) -> list[str]:
        """
        Return the vocabulary of the embedder. This functionality is optional, and it helps when going through the exercise of 
        removing from the embedding model words that are not (and potentially will never be) used, thus helping to save storage space.
        """
        return []

class GloVeEmbeddings(Embedder):
    """
    Sample implementation of an Embedder, using GloVe embeddings.
    """

    glove_weights = '/embeds.npy'
    glove_vocab = '/embeds.vocab.pickle'

    @timed
    def __init__(self, keys: dict[str, int], weights: list) -> None:
        
        self.__keymap = keys
        self.__weights = weights
        self.__length = len(weights[0])
        self.__stemmer = PorterStemmer()

    @timed
    def from_output_dir(embeds_dir: str, mem_mapped: bool = True):
        logger.info(f"Loading numpy tensors in mem-mapped mode: {mem_mapped}")
        tensors_file = embeds_dir + GloVeEmbeddings.glove_weights
        vocab_file = embeds_dir + GloVeEmbeddings.glove_vocab
        tensors = np.lib.format.open_memmap(tensors_file) if mem_mapped else np.load(tensors_file)
        with open(vocab_file, 'rb') as vf:
            key_map = pickle.load(vf)

        return GloVeEmbeddings(key_map, tensors)
    

    def to_output_dir(self, embeds_dir: str):
        logger.info("Saving embeddings to %s", embeds_dir)
        tensors_file = embeds_dir + GloVeEmbeddings.glove_weights
        vocab_file = embeds_dir + GloVeEmbeddings.glove_vocab

        logger.info("Saving weights to %s", tensors_file)
        np.save(tensors_file, self.__weights)
        
        logger.info("Saving vocab to %s", vocab_file)
        with open(vocab_file, 'wb') as vf:
            pickle.dump(self.__keymap, vf)

    def __getitem__(self, key):
        idx = self.__keymap.get(key, -1)
        if idx == -1:
            return self.zeros
        return self.__weights[idx]
    
    @property
    def zeros(self):
        return np.zeros((self.embed_length))

    def __call__(self, key: str):
        return self.__getitem__(key)

    @property
    def embed_length(self):
        return self.__length

    @property
    def vocab(self):
        return list(self.__keymap.keys())
    
    def cleanup(self, words: list[str], with_length: int = 3) -> list[str]:
        #collect the indices of every word on the list that's over the threshold and remove from the index.
        removed = []
        #pop all the words from the vocab and rebuild the index
        for word in words:
            if word in self.__keymap:
                if len(word) > with_length:
                    self.__keymap.pop(word)
                    removed.append(word)
        #rebuild
        weights = []
        
        for i, word in enumerate(self.__keymap.keys()):
            embed = self.__keymap[word]
            weights.append(self.__weights[embed])
            self.__keymap[word] = i
        
        #finally replace the weights
        self.__weights = np.array(weights)
        return removed

    #use the retrieved embeddings to embed a query
    def embed_query(self, query, do_stem=True, do_mean=True, all_word_embeds=False):
        """ Generate one or multiple embeddings for the query. 
        OPTIONS: 
            * `do_stem`: Since the embeddings index might be stemmed to for performance reasons, should I use the stemmer to look into the index?
            * `do_mean`: Mean the embeddings of multiple words to produce a sentence embedding.
            * `all_word_embeds`: Instead of returning one embedding, return a dictionary with all the words embedded separately in addition to the full query.
        """

        ret = self.zeros

        tokens = word_tokenize(query)
        #initialize an embeds dict to collect 
        embeds_dict = {}
        
        if len(tokens) == 0: return {query: ret} if all_word_embeds else ret

        for token in tokens:
            #because the keys in the embeddings dict are now stemmed.
            if do_stem:
                token = self.__stemmer.stem(token.lower())
            else:
                token = token.lower()
            
            word_embed = self[token]

            if all_word_embeds:
                embeds_dict[token] = word_embed

            ret += word_embed
        
        if do_mean:
            ret = np.divide(ret, len(tokens))

        if all_word_embeds:
            embeds_dict[query] = ret
            return embeds_dict

        return ret


class EmbeddingIndex():
    """
    This class models the index for KNN search.

    Initial Indexing:
    * Initially, the user will want to build an index using the `from_scratch` method and a number of hyperplanes.
    * Next, the user will want to build one or more indexes using the `build_index` method. This class supports indexing multiple spaces.
    * Finally, the user will want to save the indexes to a pickle file using the `to_pickle` method.

    Production Uage:
    * Load the pre-built index using the `from_pickle` method.
    * Perform searches using the `knn_search` method.

    """
    def __init__(self, planes, embeds: Embedder, index: dict = {}, synonyms: dict = {}) -> None:
        self.__planes = planes
        self.__embeds = embeds
        self.__hash_of_zeros = self.hash(embeds.zeros)
        self.__index = index
        self.__synonyms = synonyms
        self.__default_space = 'default'

    def from_scratch(num_planes: int, embeds: Embedder):
        """
        Start a brand new index using an embedder and a number of random planes.

        PARAMETERS:
            * `num_planes`: Defines the number of hyperplanes of the model, this is an hyperparameter and needs to be picked manually according to the amount of data. A fair starting point is 8 to 10.
            * `embeds`: The implementation of an embedder for this index to use.


        """
        planes = np.random.normal(size=(num_planes, embeds.embed_length))
        return EmbeddingIndex(planes, embeds)

    @timed
    def from_pickle(filename: str, embeds: Embedder):
        """
        Load an embedding index from a file.

        PARAMETERS:
            * `filename`: The pickle filename where the index is stored.
            * `embeds`: The embedder used to build the index.
        """
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                planes = data['planes']
                index = data['index']
                synonyms = data.get('synonyms', {})
                return EmbeddingIndex(planes, embeds, index, synonyms)

        except:
            logger.error("Error while loading the embedding index from a pickle file...")
            raise
    @timed
    def to_pickle(self, filename):
        """
        Save this index to a pickle file.
        """
        try: 
            with open(filename, 'wb') as f:
                data = {
                    'planes': self.__planes,
                    'index' : self.__index,
                    'synonyms': self.__synonyms
                }
                pickle.dump(data, f)
        except:
            logger.error(f"Error while writing pickle file {f}")
            raise

    def hash(self, embedding):
        """
        INTERNAL: Apply the hashing function to an embedding tensor.
        """
        dp = np.dot(embedding, self.__planes.T)
        dps = np.sign(dp)
        bucket = 0
        for i in range(self.__planes.shape[0]):
            bucket += 2**i * (1 if dps[i] == 1 else 0)

        return bucket
    
    def __get_space(self, space_name: str = None) -> dict:
        
        if not space_name: space_name = self.__default_space
        
        ret = self.__index.setdefault(space_name, {})
        return ret

    #check the words out of dict
    @timed
    def out_of_dict_find(self, word, space_name = None, distance=3):
        """
        INTERNAL: Retrieve all the words for the bucket "0".
        """
        ood = self.__get_space(space_name).get(self.__hash_of_zeros, [])
        return [ w for w in ood if edit_distance(word, w) <= distance]
    
    
    def build_index(self, keys: list[str], space_name: str = None, do_stem=False, collect_synonyms=False, embed_all_words=True, clean_space = True):
        """
        Build the index for the given keys.

        PARAMETERS:
            * `keys`: The words to index.
            * `space_name`: The name of the index to use.
            * `do_stem`: Wether to use stemming before embedding the words or not. This option applies only if the embedder supports embedding each word.
            * `collect_synonyms`: Call the synonyms method as to cache the indexed synonyms while indexing.
        """

        space = self.__get_space(space_name)

        if clean_space:
            space.clear()
                

        for key in keys:
            #to avoid double stemming
            if collect_synonyms:
                self.synonyms(key) #simple as tea!

            embedding_map = self.__embeds.embed_query(key, do_stem=do_stem, all_word_embeds=embed_all_words)
            
            if not self.__embeds.supports_all_words_embeds:
                embedding_map = {"key": embedding_map} ## if the embedder doesn't support that feature, we just move on
            
            for embedding_key, embed in embedding_map.items():    
                bucket = self.hash(embed)
                bucket_list = space.get(bucket, [])
                if key not in bucket_list: bucket_list.append(key)
                space[bucket] = bucket_list
        
        return self
    
    @timed
    def knn_search(self, term: str, k=10, space_name = None, search_words=False, use_synonyms=False, include_search_terms=False, use_stemmer=False):
        """
        Perform a search over the index.

        PARAMETERS:
            * `term`: The search term.
            * `k`: The number of results to retrieve.
            * `space_name`: The index to use.
            * `search_words`: Wether to use each word on the terms or not.
            * `use_synonyms`: Wether to include synonyms of each word. This feature uses wordnet.
            * `include_search_terms` Wether to include the search terms + synonyms (if using) on the results or not.
            * `use_stemmer` Indicates if the embedder should try to stem the words before embedding.
        """
        search_words = False if not self.__embeds.supports_all_words_embeds else search_words

        index = self.__get_space(space_name)
        ##first, locate the bucket of the term.
        embed_q = self.__embeds.embed_query(term, all_word_embeds=search_words, do_stem=use_stemmer)
        term_embeddings = None

        synonym_embeds = []
        synonyms = []

        if use_synonyms:
            synonyms = self.synonyms(term)
            if term in synonyms:
                synonyms.pop(synonyms.index(term))
            #embed the synonyms
            for s in synonyms:
                s_embed = self.__embeds.embed_query(s, all_word_embeds=search_words)
                synonym_embeds.append(s_embed)

        if not search_words:
            buckets = [self.hash(embed_q)] + [self.hash(s_embed) for s_embed in synonym_embeds]
            term_embeddings = [embed_q] + synonym_embeds
        else:
            buckets = [self.hash(embed) for embed in embed_q.values()] + [self.hash(syn_embed[term]) for syn_embed in synonym_embeds for term in syn_embed]
            term_embeddings = [embed_q[term]] + [syn_embed[term] for syn_embed in synonym_embeds for term in syn_embed]

        candidate_words = []

        for bucket in buckets:
            if bucket == 0: continue # skip the out of dictionary bucket.
            candidate_words = candidate_words + [i for i in index.get(bucket, []) if i not in candidate_words]

        #if we don't have enough words, first we go out of dict, and then neighboring buckets.

        if len(candidate_words) < k:
            candidate_words += self.out_of_dict_find(term, space_name)

        i = 1
        while len(candidate_words) < k:
            bucket_search = index.get(bucket + i, []) + index.get(bucket - i, [])
            candidate_words = candidate_words + [ i for i in bucket_search if i not in candidate_words]
            i += 1
            #when we run out of options.
            if i >= 2 ** len(self.__planes): break

        ret = []
        
        #now we embed each word we found, calculate the cosine 
        #similaity with the search term, and return the results.
        
        #embed the candidate words only once
        candidate_embeds = [self.__embeds.embed_query(w, do_stem=use_stemmer) for w in candidate_words]
        
        #for each term, we calculate the cosine distance of that term with all the candidate words.
        for term_embedding in term_embeddings:
            
            cosine = cosine_distance(term_embedding, candidate_embeds)

            #we make an array of tuples 
            tuples = [(w, c) for w, c in zip(candidate_words, cosine)]

            ret += tuples
        ##finally, sort by cosine similairty and return the K first
        ret = sorted(ret, key=lambda x: x[1], reverse=True)
        return ret[0: k] if not include_search_terms else (ret[0: k], [term] + synonyms)
    
    def synonyms(self, term:str):
        """
        Collect synonyms of a given search term using wordnet.
        """
        
        key = term
        ret = self.__synonyms.get(key, [])

        if len(ret) > 0:
            return ret

        term = term.replace(' ', '_')
        m = wn.morphy(term, wn.NOUN)
        if m:
            synonyms = wn.synsets(m)
            for s in synonyms:
                lemmas = s.lemmas()
            for l in lemmas:
                parts = l.name().split('.')
                ret.append(parts[-1].replace('_', ' '))    
        
        ##cache
        self.__synonyms[key] = ret

        return ret

    def dump_index(self, space = None) -> list[str]:
        """
        Find all the words that have been indexed and dump them.
        """

        index = self.__get_space(space)

        ##use a set comprehension
        words = {w for k in index for w in index[k]}

        return sorted(words)

    @property
    def planes(self):
        return np.copy(self.__planes)

    def collect_unrelated_keys(self) -> list[str]:
        """
        Cleanup method that hashes all the terms on the embedded's vocabulary, finds their buckets and if not found, flags them as unrelated terms.
        This should be a good heuristic for removing embeddings that will never get used.
        """
        index = self.__index
        embedder = self.__embeds
        
        ret = []

        for key in embedder.vocab:
            bucket = self.hash(embedder[key])
            #check if there is a bucket on any space of the index, if there is we just skip the word

            #there isnt a bucket for the key in any space
            if not any([index[space].get(bucket, None) != None for space in index]):
                ret.append(key)

        return ret

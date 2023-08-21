from nltk import edit_distance
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
from .embeds import Embedder
from .index import IndexBackend, DictionaryIndexBackend

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
    def __init__(self, planes, embeds: Embedder, index: dict = {}, synonyms: dict = {}, index_backend: IndexBackend = None) -> None:
        self.__planes = planes
        self.__embeds = embeds
        self.__hash_of_zeros = self.hash(embeds.zeros)
        self.__index = index_backend if index_backend != None else DictionaryIndexBackend(data=index)
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

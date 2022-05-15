from mimetypes import init
from nltk import word_tokenize, PorterStemmer, edit_distance
from nltk.corpus import wordnet as wn
from pyprofile import timed
import numpy as np
import pickle
import logging as logger
from functools import lru_cache

glove_weights = '/embeds.npy'
glove_vocab = '/embeds.vocab.pickle'

## utility function
def cosine_distance(t1, t2):
    #I use to avoid division by 0 1e-9
    return np.dot(t1, t2)/(np.linalg.norm(t1)*np.linalg.norm(t2) + 1e-9)


class GloVeEmbeddings():
    
    @timed
    def __init__(self, keys: dict[str, int], weights: list) -> None:
        
        self.__keymap = keys
        self.__weights = weights
        self.__length = len(weights[0])
        self.__stemmer = PorterStemmer()

    @timed
    def from_output_dir(embeds_dir: str, mem_mapped: bool = True):
        logger.info(f"Loading numpy tensors in mem-mapped mode: {mem_mapped}")
        tensors_file = embeds_dir + glove_weights
        vocab_file = embeds_dir + glove_vocab
        tensors = np.lib.format.open_memmap(tensors_file) if mem_mapped else np.load(tensors_file)
        with open(vocab_file, 'rb') as vf:
            key_map = pickle.load(vf)

        return GloVeEmbeddings(key_map, tensors)
    
    def __getitem__(self, key):
        idx = self.__keymap.get(key, -1)
        if idx == -1:
            return self.zeros()
        return self.__weights[idx]
    
    def zeros(self):
        return np.zeros((self.embed_length()))

    def __call__(self, key: str):
        return self.__getitem__(key)

    def embed_length(self):
        return self.__length
    
    #use the retrieved embeddings to embed a query
    def embed_query(self, query, do_stem=True, do_mean=True, all_word_embeds=False):
        """ Generate one or multiple embeddings for the query. 
        OPTIONS: 
            * `do_stem`: Since the embeddings index is stemmed to reduce size, should this function take care of stemming or is the query pre-stemmed?
            * `do_mean`: Mean the embeddings of multiple words to produce a sentence embedding.
            * `all_word_embeds`: Instead of returning one embedding, return a dictionary with all the words embedded separately in addition to the full query.
        """

        ret = self.zeros()

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

    def __init__(self, planes, embeds: GloVeEmbeddings, index: dict = {}) -> None:
        self.__planes = planes
        self.__embeds = embeds
        self.__hash_of_zeros = self.hash(embeds.zeros())
        self.__index = index
        self.__default_space = 'default'

    def from_scratch(num_planes, embeds: GloVeEmbeddings):
        planes = np.random.normal(size=(num_planes, embeds.embed_length()))
        return EmbeddingIndex(planes, embeds)

    @timed
    def from_pickle(filename: str, embeds: GloVeEmbeddings):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                planes = data['planes']
                index = data['index']
                return EmbeddingIndex(planes, embeds, index)

        except:
            logger.error("Error while loading the embedding index from a pickle file...")
            raise
    @timed
    def to_pickle(self, filename):
        try: 
            with open(filename, 'wb') as f:
                data = {
                    'planes': self.__planes,
                    'index' : self.__index
                }
                pickle.dump(data, f)
        except:
            logger.error(f"Error while writing pickle file {f}")
            raise

    def hash(self, embedding):
        dp = np.dot(embedding, self.__planes.T)
        dps = np.sign(dp)
        bucket = 0
        for i in range(self.__planes.shape[0]):
            bucket += 2**i * (1 if dps[i] == 1 else 0)

        return bucket
    
    def __get_space(self, space_name: str = None) -> dict:
        
        if not space_name: space_name = self.__default_space
        
        ret = self.__index.get(space_name, {})
        self.__index[space_name] = ret
        return ret

    #check the words out of dict
    def out_of_dict_find(self, word, space_name = None, distance=3):
        ood = self.__get_space(space_name).get(self.__hash_of_zeros, [])
        return [ w for w in ood if edit_distance(word, w) <= distance]
    
    @timed
    def build_index(self, keys: list[str], space_name: str = None, do_stem=False):
        for key in keys:
            #to avoid double stemming
            embedding_map = self.__embeds.embed_query(key, do_stem=do_stem, all_word_embeds=True)
            for embedding_key, embed in embedding_map.items():    
                bucket = self.hash(embed)
                space = self.__get_space(space_name)
                bucket_list = space.get(bucket, [])
                if key not in bucket_list: bucket_list.append(key)
                space[bucket] = bucket_list
        
        return self
    
    @timed
    def knn_search(self, term: str, k=10, space_name = None, search_words=False, use_synonyms=False, include_search_terms=False):
        
        index = self.__get_space(space_name)
        ##first, locate the bucket of the term.
        embed_q = self.__embeds.embed_query(term, all_word_embeds=search_words)
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
  
        ##then gather elements looking back and forth until we have enough items to compare.
        candidate_words = self.out_of_dict_find(term, space_name)

        for bucket in buckets:
            candidate_words = candidate_words + [i for i in index.get(bucket, []) if i not in candidate_words]

    
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
        for term_embedding in term_embeddings:
            for word in candidate_words:
                word_embed = self.__embeds.embed_query(word)
                cosine = cosine_distance(word_embed, term_embedding)
                ret.append((word, cosine))

        ##finally, sort by cosine similairty and return the K first
        ret = sorted(ret, key=lambda x: x[1], reverse=True)
        return ret[0: k] if not include_search_terms else (ret[0: k], [term] + synonyms)
    
    @timed
    def synonyms(self, term:str):
        term = term.replace(' ', '_')
        m = wn.morphy(term, wn.NOUN)
        ret = []
        if m:
            synonyms = wn.synsets(m)
            for s in synonyms:
                lemmas = s.lemmas()
            for l in lemmas:
                parts = l.name().split('.')
                ret.append(parts[-1].replace('_', ' '))    
        return ret

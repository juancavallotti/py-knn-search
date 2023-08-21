from pyprofile import timed
from nltk import PorterStemmer, word_tokenize

import pickle
import numpy as np
import logging as logger

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

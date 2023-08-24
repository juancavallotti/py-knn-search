from functools import lru_cache
from pyknn import Embedder
import numpy as np
from pytest import fixture
from nltk import word_tokenize
from pyknn.index import DictionaryIndexBackend

from pyknn.knn import EmbeddingIndex

class MyEmb(Embedder):
    
    def __init__(self) -> None:
        super().__init__()
        self._supports_all_words_embeds = False
        self.calls = []
        

    @property
    def zeros(self):
        return np.zeros(self.embed_length)

    @property
    def embed_length(self):
        return 300

    def embed_query(self, query, **kwargs):
        self.called = True
        self.calls.append((query, kwargs))
        return self.zeros


class MyAllWordEmb(Embedder):
    called = False

    def __init__(self) -> None:
        super().__init__()
        self._supports_all_words_embeds = True
        

    @property
    def zeros(self):
        return np.zeros(self.embed_length)

    @property
    def embed_length(self):
        return 300

    def embed_query(self, query, all_word_embeds: bool = False, **kwargs):
        self.called = True

        if not all_word_embeds:
            return self.generate_embed(query)

        tokens = word_tokenize(query)
        #initialize an embeds dict to collect 
        embeds_dict = {}
        
        if len(tokens) == 0: 
            return {query: self.generate_embed(query)}

        tokens.append(query)

        ##for each token we return 
        return {k.lower(): self.generate_embed(k.lower()) for k in tokens}
    
    @lru_cache
    def generate_embed(self, term: str):
        return np.random.randn(self.embed_length)

@fixture
def embedder():
    return MyEmb()

@fixture
def awembedder():
    return MyAllWordEmb()

@fixture
def simpleIndex(awembedder):
    return EmbeddingIndex.from_scratch(2, awembedder, DictionaryIndexBackend(data={}))
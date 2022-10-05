from functools import lru_cache
from pyknn import Embedder
import numpy as np
from pytest import fixture
from nltk import word_tokenize

class MyEmb(Embedder):
    
    called = False

    def __init__(self) -> None:
        super().__init__()
        self._supports_all_words_embeds = False
        

    @property
    def zeros(self):
        return np.zeros(self.embed_length)

    @property
    def embed_length(self):
        return 300

    def embed_query(self, query, **kwargs):
        self.called = True
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

    def embed_query(self, query, all_word_embeds: bool, **kwargs):
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

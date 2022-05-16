from pyknn import Embedder
import numpy as np
from pytest import fixture

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

@fixture
def embedder():
    return MyEmb()

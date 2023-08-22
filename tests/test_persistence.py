from numpy import ndarray
from pyknn import PersistenceProvider, EmbeddingIndex, IndexBackend, DictionaryIndexBackend
import numpy as np

class TestPersistenceProvider(PersistenceProvider):

    def __init__(self) -> None:
        super().__init__()
        self.__planes = np.random.normal(size=(3, 300))
        self.__index = DictionaryIndexBackend(data={0: 'hello'})
        self.__synonyms = {'hello': 'world'}

    def persist(self, planes: ndarray, index: IndexBackend, synonyms: dict):
        assert self.__planes is planes, "Could not verify planes."
        assert self.__index is index, "Could not verify index."
        assert self.__synonyms is synonyms, "Could not verify synonyms."

    def read(self) -> tuple[ndarray, IndexBackend, dict]:
        return self.__planes, self.__index, self.__synonyms

def test_persistence_provider(awembedder):
    provider = TestPersistenceProvider()
    index = EmbeddingIndex.from_provider(provider=provider, embeds=awembedder)
    index.to_provider(provider)
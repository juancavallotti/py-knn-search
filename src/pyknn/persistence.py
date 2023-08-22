from pyknn.index import IndexBackend, DictionaryIndexBackend

import logging as logger
from numpy import ndarray
import pickle

class PersistenceProvider():

    def persist(self, planes: ndarray, index: IndexBackend, synonyms: dict):
        """
        Store the relevant information of a search index.
        """
        raise NotImplementedError("Needs implementation")
    
    def read(self) -> tuple[ndarray, IndexBackend, dict]:
        """
        Reads the planes, index and synonyms and return it as a tuple.
        """
        raise NotImplementedError("Needs implementation")
    

class PicklePersistenceProvider(PersistenceProvider):
    
    def __init__(self, filename: str) -> None:
        self.__filename = filename
    
    def persist(self, planes, index, synonyms):
        try: 
            with open(self.__filename, 'wb') as f:
                data = {
                    'planes': planes,
                    'index' : index.dump(),
                    'synonyms': synonyms
                }
                pickle.dump(data, f)
        except:
            logger.error(f"Error while writing pickle file {f}")
            raise
    
    def read(self) -> tuple[ndarray, IndexBackend, dict]:
        try:
            with open(self.__filename, 'rb') as f:
                data = pickle.load(f)
                planes = data['planes']
                index = data['index']
                synonyms = data.get('synonyms', {})
                return planes, DictionaryIndexBackend(index), synonyms

        except:
            logger.error("Error while loading the embedding index from a pickle file...")
            raise

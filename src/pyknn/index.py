class IndexBackend():
    """
    This class acts as an interface for different indexing backends. This is helpful if we want to switch from the current in-memory setting
    to something else, like a database.
    """

    def __getitem__(self, key):
        """
        Read an item by key in the index.
        """
        raise NotImplementedError("Not yet implemented")
    
    def __setitem__(self, key, value):
        """
        Write an item by key in the inex.
        """
        raise NotImplementedError("Not yet implemented")
    
    def setdefault(self, key, defValue):
        """
        Simulate the dictionary setdefault operation.
        """
        raise NotImplementedError("Not yet implemented")

    def dump(self) -> dict:
        """
        Dump the entire index into a dictionary so it could be stored in pickle format. This operation is optional.
        """
        raise NotImplementedError("Not yet implemented")

class DictionaryIndexBackend(IndexBackend):
    """
    Basic implementation to keep backward-compatibility with the old framework.
    """
    __dict: dict

    def __init__(self, data: dict = {}) -> None:
        self.__dict = data
    
    def __getitem__(self, key):
        return self.__dict[key]

    def __setitem__(self, key, value):
        self.__dict[key] = value
    
    def setdefault(self, key, defValue):
        return self.__dict.setdefault(key, defValue)
    
    def dump(self) -> dict:
        return self.__dict
from abc import ABC, abstractmethod


class I_Cache(ABC):
    @abstractmethod
    def get(self, key: str):
        """
        Retrieve a value by key. Returns None if key not found.
        """
        pass

    
    @abstractmethod
    def set(self, key: str, value):
        """
        Set a value for a key.
        """
        pass


    @abstractmethod
    def clear(self):
        """
        Clear all cache entries.
        """
        pass


    @abstractmethod
    def keys(self) -> list[str]:
        """
        Return a list of all keys in the cache.
        """
        pass


    @abstractmethod
    def delete(self, key: str):
        """
        Delete a key from the cache.
        """
        pass
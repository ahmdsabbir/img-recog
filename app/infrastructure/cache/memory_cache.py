from app.interfaces.cache import I_Cache


class MemoryCache(I_Cache):
    def __init__(self):
        self._store = {}


    def get(self, key: str):
        return self._store.get(key)
    

    def set(self, key: str, value):
        self._store[key] = value


    def clear(self):
        self._store.clear()

    
    def keys(self) -> list[str]:
        return list(self._store.keys())
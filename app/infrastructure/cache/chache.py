from app.interfaces.cache import I_Cache
from app.infrastructure.cache.providers.memory_cache import MemoryCache


class Cache(I_Cache):
    def __init__(self):
        self._provider: I_Cache = MemoryCache()

    def get(self, key: str):
        return self._provider.get(key=key)
    
    
    def set(self, key: str, value):
        return self._provider.set(key=key, value=value)
    

    def clear(self):
        self._provider.clear()


    def keys(self) -> list[str]:
        return self._provider.keys()
    

    def delete(self, key: str):
        self._provider.delete(key)

    def info(self):
        return self._provider.info()
from abc import ABC, abstractmethod


class I_Cache(ABC):
    @abstractmethod
    def get(self, key: str):
        pass

    @abstractmethod
    def set(self, key: str, value):
        pass


    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def keys(self) -> list[str]:
        pass
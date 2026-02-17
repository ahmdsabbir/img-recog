import hashlib

class CacheKeys:
    @staticmethod
    def attribute_model(category: str, attribute: str) -> str:
        if not category or not attribute:
            raise ValueError("Invalid cache key arguments")
        
        return f"attribute_model:{category}:{attribute}"
    

    @staticmethod
    def category_models(category: str) -> str:
        if not category:
            raise ValueError("Invalid cache key argument")
        
        return f"category_models:{category}"
    
    
    @staticmethod
    def embedding(img_path: str) -> str:
        if not img_path:
            raise ValueError("Invalid cache key argument")
        
        img_hash = CacheKeys._hash_img_path(img_path)
        
        return f"embedding:{img_hash}"
    

    @staticmethod
    def faiss_index(category: str) -> str:
        if not category:
            raise ValueError("Invalid cache key argument")
        
        return f"faiss_index:{category}"
    

    def _hash_img_path(path: str) -> str:
        return hashlib.md5(path.encode()).hexdigest()
from typing import Literal

from cli.message import Message
from app.interfaces.cache import I_Cache

Msg = Message()


def run_cache(
    cache: I_Cache,
    *,
    sub_command: Literal["list", "clear", "delete", "info"] | None = None,
    key: str | None = None,
) -> None:
    """
    Inspect or manage the in-memory cache.

    Subcommands:
    - list: List all cache keys
    - clear: Clear all cache entries
    - delete: Delete a specific cache key (requires --key)
    - info (default): Show cache status
    """
    if sub_command == "clear":
        count = len(cache.keys())
        cache.clear()
        print(Msg.info(f"Cleared {count} cache entries."))

    elif sub_command == "list":
        keys = cache.keys()
        print(Msg.highlight(f"Listing {len(keys)} cache keys:"))
        for k in keys:
            print(f"  - {k}")

    elif sub_command == "delete":
        if not key:
            print(Msg.alert("Error: --key is required for delete command"))
            print(Msg.info("Usage: cache delete --key <cache_key>"))
            return

        if key in cache.keys():
            cache.delete(key)
            print(Msg.info(f"Deleted cache key: {key}"))
        else:
            print(Msg.alert(f"Cache key not found: {key}"))

    else:  # info or None
        info = cache.info()
        print(Msg.highlight("Cache status:"))
        print (Msg.neutral(f'\tNumber of entries: {info["num_entries"]}'))
        print(Msg.neutral(f'\tSpace taken: {info["size"]} (it\'s not the most accurate, but this\'ll do)'))


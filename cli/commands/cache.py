from cli.message import Message


Msg = Message()

def run_cache(cache, *, clear: bool = False, list_keys: bool = False) -> None:
    """
    Inspect or manage the in-memory cache.

    Flags are mutually exclusive: `clear` takes priority over `list_keys`.
    With neither flag, just confirms the cache is live.
    """
    if clear:
        cache.clear()
        print(Msg.info("All caches cleared."))
    elif list_keys:
        keys = cache.keys()
        print(Msg.highlight(f"Listing {len(keys)} keys:"))
        for k in keys:
            print(" -", k)
    else:
        print(Msg.highlight("Cache is live and will speed up repeated queries/classifications."))

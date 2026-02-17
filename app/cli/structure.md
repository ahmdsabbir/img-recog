# CLI Modularization Plan

## Proposed File Structure

```
app/
└── cli/
    ├── __init__.py
    ├── main.py                  # Entry point: argparse + serve loop
    ├── parser.py                # Interactive command parser (parse_command)
    ├── container.py             # Dependency wiring (services, models, stores)
    └── commands/
        ├── __init__.py
        ├── rebuild.py           # rebuild command handler
        ├── query.py             # query command handler
        ├── classify.py          # classify command handler
        ├── cache.py             # cache command handler
        └── train.py             # train command handler
```

## Why This Way

- `container.py` centralizes infrastructure construction — swap a model or store without touching handlers
- Each command file is independently testable and has a single reason to change
- `parser.py` is isolated so the interactive parsing logic can be unit tested or replaced
- `main.py` becomes a thin orchestrator: parse → dispatch → done

## Module Responsibilities

| File | Responsibility |
|---|---|
| `main.py` | `argparse`, serve loop, dispatch table |
| `parser.py` | `parse_command()` for interactive input |
| `container.py` | Builds and exposes shared services |
| `commands/rebuild.py` | Encodes images, builds FAISS index |
| `commands/query.py` | Loads index, runs similarity search |
| `commands/classify.py` | Category + attribute classification |
| `commands/cache.py` | Cache inspection and clearing |
| `commands/train.py` | Delegates to training pipeline |
# Adding New Commands to the CLI

This guide explains how to add new commands to the interactive CLI.

## Architecture Overview

```
app/cli/
├── main.py              # CLI entry point, orchestrates commands
├── parser.py            # Parses interactive command strings
├── container.py         # Dependency injection (shared services)
└── commands/            # Command implementations
    ├── query.py
    ├── classify.py
    ├── rebuild.py
    ├── train.py
    └── cache.py
```

## Quick Start

To add a new command (e.g., `export`), follow these steps:

### Step 1: Create the Command Handler

Create a new file `app/cli/commands/export.py`:

```python
def run_export(recommender, vector_store, output_path: str) -> None:
    """
    Export the FAISS index to a file.

    Args:
        recommender: The RecommenderService instance
        vector_store: The FaissVectorStore instance
        output_path: Destination path for the export
    """
    print(f"Exporting index to {output_path}...")
    # Your implementation here
    print("Export complete!")
```

**Naming Convention:** Command functions should be named `run_<command_name>`.

### Step 2: Add Argument Parsing

Edit `app/cli/parser.py` to add your new arguments to the `cmd_args` dict:

```python
cmd_args = {
    # ... existing fields ...
    "output_path": None,   # Add your new fields here
}
```

Then add parsing logic for your flags:

```python
elif p == "--output" and i + 1 < len(parts):
    cmd_args["output_path"] = parts[i + 1]
    i += 2
```

### Step 3: Register the Command in main.py

**Import your command:**

```python
from app.cli.commands import rebuild, train, query, classify, cache, export
```

**Add it to the interactive serve loop:**

```python
# In main.py, inside the serve mode loop:
elif cmd["command"] == "export":
    output_path = cmd["output_path"]
    if not output_path:
        print("Please provide --output for export")
        continue

    export.run_export(
        container.recommender,
        container.vectore_store,
        output_path
    )
```

**Optional: Add as a non-interactive command**

If your command should also work outside serve mode, add it to the argparse choices and handler:

```python
# In main():
parser.add_argument("command", choices=["serve", "rebuild", "train", "export"])
# ... then add:
if args.command == "export":
    export.run_export(
        container.recommender,
        container.vectore_store,
        args.output_path
    )
    return
```

### Step 4: Update Container (if needed)

If your command needs new dependencies, add them to `app/cli/container.py`:

```python
class Container:
    def __init__(self):
        # ... existing dependencies ...
        from app.services.my_new_service import MyNewService
        self.my_service = MyNewService()
```

Then pass `container.my_service` to your command function.

## Example: Complete Walkthrough

Let's add a `stats` command that shows index statistics.

### 1. Create `app/cli/commands/stats.py`:

```python
import os
from app.config import settings


def run_stats(vector_store) -> None:
    """Display statistics about the FAISS index."""
    if not os.path.exists(settings.FAISS_INDEX_PATH):
        print("No index found. Run rebuild first.")
        return

    vector_store.load()
    # Assume vector_store has a get_stats() method
    stats = vector_store.get_stats()
    print(f"Total vectors: {stats['count']}")
    print(f"Dimension: {stats['dimension']}")
```

### 2. Update `app/cli/parser.py`:

No new arguments needed for this simple command.

### 3. Update `app/cli/main.py`:

**Import:**
```python
from app.cli.commands import rebuild, train, query, classify, cache, stats
```

**Add to serve loop:**
```python
elif cmd["command"] == "stats":
    stats.run_stats(container.vectore_store)
```

### 4. Test it:

```bash
python -m app.cli.main serve
>>> stats
Total vectors: 1500
Dimension: 512
```

## Command Function Signatures

Command functions receive dependencies from the Container. Common patterns:

```python
# Simple command - uses one dependency
def run_cache(cache, *, clear: bool = False, list_keys: bool = False) -> None:
    ...

# Command with multiple dependencies
def run_query(recommender, vector_store, img_path: str) -> None:
    ...

# Command with optional flag
def run_classify(embedding, cache, img_path: str, use_trained: bool = False) -> None:
    ...
```

## Best Practices

1. **Keep commands focused** - Each command should do one thing well
2. **Use type hints** - Helps with IDE autocomplete and documentation
3. **Validate inputs** - Check file paths, required arguments before executing
4. **Print clear output** - Users should understand what happened
5. **Handle errors gracefully** - Try/except where appropriate, print helpful messages
6. **Use the Container** - Don't instantiate services inside commands
7. **Document with docstrings** - Explain what the command does and its parameters

## Testing Your Command

After adding a new command:

```bash
# Test help
python -m app.cli.main --help

# Test non-interactive (if supported)
python -m app.cli.main export --output data/export.json

# Test interactive
python -m app.cli.main serve
>>> export --output data/export.json
```

## Common Patterns

### Command with file I/O:

```python
import os

def run_export(vector_store, output_path: str) -> None:
    if os.path.exists(output_path):
        print(f"File {output_path} already exists.")
        return

    # ... perform export ...
    print(f"Exported to {output_path}")
```

### Command with subcommands (like `cache`):

```python
def run_cache(cache, *, clear: bool = False, list_keys: bool = False) -> None:
    if clear:
        cache.clear()
        print("All caches cleared.")
    elif list_keys:
        keys = cache.keys()
        print(f"Listing {len(keys)} keys:")
        for k in keys:
            print(" -", k)
    else:
        print("Cache is live.")
```

## Troubleshooting

**ImportError when adding new command:**
- Make sure you added the import in `main.py`
- Check that `commands/__init__.py` exists (even if empty)

**Command not recognized in interactive mode:**
- Verify the command name matches in `parser.py` and `main.py`
- Check the `cmd_args` dict includes your command's arguments

**Container dependency missing:**
- Add the dependency to `container.py`
- Pass it from `container.<dependency>` when calling your command

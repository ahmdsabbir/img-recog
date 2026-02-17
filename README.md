# Image Recognition & Recommendation System

A visual product recommendation system using CLIP embeddings and FAISS vector search. Supports background removal and image preprocessing for improved matching accuracy.

## Features

- **CLIP-based embeddings** - OpenAI's CLIP model for semantic image understanding
- **FAISS vector search** - Fast similarity search for product recommendations
- **Zero-shot classification** - Classify images into categories and attributes without training using CLIP
- **Trainable attribute classifiers** - Train custom attribute classifiers using your own labeled data
- **Background removal** - Optional rembg-based preprocessing to isolate products
- **Preprocessed image saving** - Save processed images for inspection/debugging
- **Dual-mode preprocessing** - Choose between background removal or passthrough mode

## Project Structure

```
img-recog/
├── app/
│   ├── cli/                     # CLI module
│   │   ├── main.py              # CLI entry point (serve, rebuild, train)
│   │   ├── parser.py            # Command parser for interactive mode
│   │   ├── query.py             # Query command handler
│   │   ├── classify.py          # Classify command handler
│   │   ├── rebuild.py           # Rebuild command handler
│   │   ├── train.py             # Train command handler
│   │   └── cache.py             # Cache command handler
│   ├── config.py                # Configuration settings
│   ├── container.py             # Dependency injection container
│   ├── domain/                  # Domain entities
│   │   ├── entities.py          # Core domain entities
│   │   └── types.py             # Type definitions
│   ├── infrastructure/          # External dependencies (DB, models, etc.)
│   │   ├── cache/               # In-memory caching layer
│   │   │   ├── providers        # Cache providers
│   │   │       ├── memory_cache # In memory cache implementation
│   │   │   ├── cache.py         # Cache api, uses one of the providers
│   │   │   ├── cache_keys.py    # Cache key generation utilities
│   │   ├── database/            # Database repositories
│   │   │   ├── pg_repository.py # PostgreSQL repository
│   │   │   └── sqlite_repository.py # SQLite repository
│   │   ├── embedding/           # CLIP embedding model
│   │   │   ├── clip_model.py    # CLIP implementation
│   │   │   └── dummy_model.py   # Dummy model for testing
│   │   ├── preprocessing/       # Image preprocessors
│   │   │   ├── factory.py       # Preprocessor factory
│   │   │   ├── passthrough_preprocessor.py
│   │   │   └── rembg_preprocessor.py
│   │   └── vector_store/        # FAISS vector store
│   │       ├── faiss_store.py   # FAISS implementation
│   │       └── in_memory_store.py
│   ├── interfaces/             # Abstract interfaces
│   │   ├── cache.py             # Cache interface
│   │   ├── embedding.py         # Embedding interface
│   │   ├── feedback.py          # Feedback interface
│   │   ├── preprocessor.py      # Preprocessor interface
│   │   ├── repository.py        # Repository interface
│   │   └── vectore_store.py     # Vector store interface
│   ├── models/                 # PyTorch model definitions
│   │   ├── attribute_head.py    # Attribute classifier head
│   │   ├── category_head.py     # Category classifier head
│   │   └── clip_model.py        # CLIP model wrapper
│   ├── services/               # Business logic
│   │   ├── category_classifier_service.py
│   │   ├── feedback_service.py
│   │   ├── product_attribute_service.py
│   │   ├── recommender.py       # Recommendation engine
│   │   └── zero_shot_attribute_service.py
│   └── training/               # Training scripts and utilities
│       ├── dataset_helpers.py
│       ├── train_attribute.py
│       └── train_category.py
├── data/
│   ├── products/               # Product images for indexing
│   ├── preprocessed/           # Saved preprocessed images
│   ├── training/               # Training data organized by category/attribute
│   └── faiss_index/            # FAISS index files
│       └── id_to_filename.json # ID to filename mapping
├── models/                     # Trained attribute classifier models
│   └── <category>/             # e.g., shoe, bag
│       └── <attribute>/        # e.g., color, gender
│           ├── model.pt        # PyTorch model weights
│           └── classes.json    # Class label mapping
├── scripts/                    # Utility scripts
│   ├── build_index.py          # Standalone index builder
│   └── retrain.py              # Retraining utilities
├── tests/                      # Test suite
│   ├── __init__.py
│   └── test_preprocessing.py
├── conftest.py                 # Pytest configuration
├── requirements-prod.txt       # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── .env                        # Environment configuration
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.12+
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
   ```bash
   cd img-recog
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # Production dependencies only
   pip install -r requirements-prod.txt

   # Or with development tools (pytest, ruff, etc.)
   pip install -r requirements-dev.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your settings:
   ```env
   DEVICE=cpu                           # or "cuda" for GPU support
   EMBEDDING_MODEL=openai/clip-vit-base-patch32
   TOP_K=5                              # Number of recommendations
   DATABASE_URL=sqlite:///./data/app.db
   ```

5. **Prepare product images**
   ```bash
   # Place your product images in the data/products directory
   mkdir -p data/products
   # Copy your product images here...
   ```

## Usage

The CLI can be invoked in two ways:

### Running the CLI

**As a module:**
```bash
python -m app.cli.main <command>
```

**Direct execution:**
```bash
python app/cli/main.py <command>
```

The CLI offers two modes of operation:
- **Direct commands**: `rebuild` and `train` for one-time operations
- **Interactive serve mode**: For running `query`, `classify`, `rebuild`, and `cache` commands

### Interactive Serve Mode

Start the interactive shell:

```bash
python -m app.cli.main serve
```

Once inside the interactive shell, you can run the following commands:

#### Querying for Similar Products

```bash
# Basic query
>>> query --image path/to/query.jpg

# Query with trained model classification
>>> classify --image path/to/image.jpg --use-trained
```

#### Rebuilding the Index

```bash
# Rebuild with default products directory
>>> rebuild

# Rebuild with custom products directory
>>> rebuild --products_dir data/products
```

#### Managing the Cache

```bash
# List all cached keys
>>> cache list

# Clear all caches
>>> cache clear
```

#### Exiting

```bash
>>> exit
# or
>>> quit
```

### Building the Vector Index (Direct Command)

Before querying, you must build the FAISS index from your product images:

```bash
python -m app.cli.main rebuild --products_dir data/products
```

### Training Attribute Classifiers (Direct Command)

Train custom attribute classifiers using your labeled data:

```bash
# Train a specific attribute for a category
python -m app.cli.main train --category shoe --attribute color
python -m app.cli.main train --category shoe --attribute gender
python -m app.cli.main train --category shoe --attribute age_group
```

**Training Data Structure:**

Organize your training images in the following directory structure:

```
data/training/
└── <category>/           # e.g., shoe, bag
    └── <attribute>/      # e.g., color, gender, age_group
        ├── <class_1>/    # e.g., black, red, blue
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        ├── <class_2>/
        │   └── ...
        └── ...
```

**Example:**
```
data/training/shoe/color/
├── black/
│   ├── shoe1.jpg
│   ├── shoe2.jpg
│   └── shoe3.jpg
├── white/
│   ├── shoe4.jpg
│   └── ...
└── red/
    └── ...
```

**Training Parameters:**
- Batch size: 8
- Learning rate: 1e-4
- Epochs: 10
- Optimizer: Adam
- Loss: CrossEntropyLoss

**Model Storage:**
Trained models are saved to `models/<category>/<attribute>/`:
- `model.pt` - PyTorch model weights
- `classes.json` - Class label mapping

### Classifying Images (Interactive Mode)

Classify an image into categories and extract product attributes. First start the interactive shell:

```bash
python -m app.cli.main serve
```

Then run the classify command:

#### Zero-Shot Classification (Default)

Uses CLIP's text prompts to classify without any training data:

```bash
>>> classify --image path/to/image.jpg
```

#### Trained Model Classification

Uses your trained attribute classifiers for improved accuracy:

```bash
>>> classify --image path/to/image.jpg --use-trained
```

If trained models aren't found, the system automatically falls back to zero-shot classification.

**How it works:**

1. **Category Classification** - Always uses zero-shot CLIP to identify the product category (e.g., "shoe", "bag")
2. **Attribute Classification** - Either:
   - **Zero-shot**: Uses CLIP text prompts to extract attributes (color, gender, style, etc.)
   - **Trained models**: Uses your custom-trained classifiers for attributes (more accurate on your data)

**Example output:**
```
Category: shoe (confidence 0.92)

Attributes:
 - color: red (confidence 0.85)
 - gender: male (confidence 0.78)
 - age_group: adult (confidence 0.92)
```

**Customizing labels:**
- **Zero-shot labels**: Edit `app/services/zero_shot_attribute_service.py`
- **Category labels**: Edit `app/services/category_classifier_service.py`

### CLI Options

#### Direct Commands

```bash
python -m app.cli.main {serve,rebuild,train} [options]
# or
python app/cli/main.py {serve,rebuild,train} [options]
```

**Options:**
- `--products_dir DIR` - Directory of product images (for rebuild command, default: `data/products`)
- `--category CATEGORY` - Product category for training (e.g., shoe, bag)
- `--attribute ATTRIBUTE` - Attribute to train (e.g., color, gender, age_group)

#### Interactive Serve Commands

Once inside `serve` mode, the following commands are available:

**`query`** - Find similar products
- `--image IMAGE` - Path to query image

**`classify`** - Classify image and extract attributes
- `--image IMAGE` - Path to image to classify
- `--use-trained` - Use trained models for attribute classification (fallback to zero-shot if not found)

**`rebuild`** - Rebuild the FAISS index
- `--products_dir DIR` - Directory of product images (default: `data/products`)

**`cache`** - Manage the in-memory cache
- `cache list` - List all cached keys
- `cache clear` - Clear all caches

**`exit` / `quit`** - Exit the interactive shell

## Preprocessing

The system includes two preprocessing modes:

### RembgPreprocessor (Default)

Removes background using the rembg library (U2Net model), crops to the product, and pads to a square canvas.

**Pipeline:**
1. Background removal (rembg/U2Net)
2. Tight crop to foreground
3. Square padding (center product on canvas)
4. Resize to 224×224 (CLIP input size)

**First run:** Downloads U2Net model weights (~170MB) to `~/.u2net/`.

### PassthroughPreprocessor

Lightweight alternative that skips background removal. Still performs:
1. RGB conversion
2. Square padding (preserves aspect ratio)
3. Resize to 224×224

Use this when:
- Images are already clean (studio shots on white backgrounds)
- Running tests without rembg dependency
- Resource-constrained environments

### Configuration

Preprocessing behavior is controlled by environment variables in `.env`:

```env
USE_BG_REMOVAL=true          # Enable/disable background removal
PREPROCESS_SIZE=224,224      # Target size for CLIP (width,height)
PREPROCESS_PADDING=0.1       # Padding fraction (0.1 = 10% margin)
PREPROCESS_BG_COLOR=255,255,255  # Background fill color (RGB)
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_preprocessing.py -v
```

### Code Quality

```bash
# Format code
ruff format app/ tests/

# Lint code
ruff check app/ tests/

# Type checking
mypy app/
```

## How It Works

1. **Index Building (rebuild)**
   - Loads product images from `data/products`
   - Preprocesses each image (background removal, resize)
   - Generates CLIP embeddings (512-dimension vectors)
   - Stores vectors in FAISS index for fast search
   - Saves index to `data/faiss_index/index.bin`

2. **Interactive Serve Mode (serve)**
   - Starts an interactive shell for running query, classify, rebuild, and cache commands
   - Initializes CLIP embedding model, FAISS vector store, and in-memory cache once
   - Allows rapid execution of multiple commands without reinitializing models

3. **Querying (query)**
   - Loads and preprocesses the query image
   - Generates CLIP embedding
   - Searches FAISS index for nearest neighbors
   - Returns top-K similar product IDs with distance scores

4. **Training (train)**
   - Loads training images from `data/training/<category>/<attribute>/<class>/`
   - Generates CLIP embeddings for all training images
   - Trains a linear classifier (AttributeHead) on top of CLIP embeddings
   - Saves model weights and class mappings to `models/<category>/<attribute>/`
   - Uses transfer learning - CLIP features are frozen, only the classifier head is trained

5. **Classification (classify)**
   - Loads and preprocesses the query image
   - **Category**: Uses CLIP zero-shot classification to determine product type
   - **Attributes**:
     - **Zero-shot mode**: Uses CLIP text prompts to predict attributes
     - **Trained mode**: Uses trained AttributeHead models for predictions (cached for performance)
   - Returns category and attributes with confidence scores
   - **Caching**: Trained models are cached in memory for faster repeated classifications

6. **Cache Management (cache)**
   - **List**: View all currently cached items (models, embeddings, etc.)
   - **Clear**: Clear all cached items to free memory
   - Automatically speeds up repeated queries/classifications by storing results

7. **Preprocessing Pipeline**
   - Ensures consistent input format (RGB, square, 224×224)
   - Removes background noise for better matching
   - Preserves product aspect ratio via square padding

## Troubleshooting

### FAISS index not found
```
Error: FAISS index not found. Rebuild index first using 'rebuild' command.
```
**Solution:** Run the rebuild command to create the index.

### Rembg not installed
```
EnvironmentError: rembg is not installed.
```
**Solution:** Install rembg or set `USE_BG_REMOVAL=false` in `.env`.

### Out of memory
**Solution:** Use a smaller embedding model or process images in batches.

### No trained models found
```
Error: No trained models found for category: shoe
```
**Solution:** Train the attribute models first using the `train` command, or use zero-shot classification without `--use-trained`.

### Training data not found
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/training/shoe/color'
```
**Solution:** Organize your training images in the correct directory structure under `data/training/<category>/<attribute>/<class>/`.

### Low classification accuracy
**Solutions:**
- For zero-shot: Improve prompt text in `app/services/zero_shot_attribute_service.py`
- For trained models: Add more training data, ensure balanced classes, or adjust training hyperparameters in `app/training/train_attribute.py`

## License

MIT

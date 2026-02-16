# Image Recognition & Recommendation System

A visual product recommendation system using CLIP embeddings and FAISS vector search. Supports background removal and image preprocessing for improved matching accuracy.

## Features

- **CLIP-based embeddings** - OpenAI's CLIP model for semantic image understanding
- **FAISS vector search** - Fast similarity search for product recommendations
- **Zero-shot classification** - Classify images into categories and extract attributes without training
- **Background removal** - Optional rembg-based preprocessing to isolate products
- **Preprocessed image saving** - Save processed images for inspection/debugging
- **Dual-mode preprocessing** - Choose between background removal or passthrough mode

## Project Structure

```
img-recog/
├── app/
│   ├── cli.py                   # CLI entry point
│   ├── config.py                # Configuration settings
│   ├── domain/                  # Domain entities
│   ├── infrastructure/          # External dependencies (DB, models, etc.)
│   │   ├── embedding/          # CLIP embedding model
│   │   ├── preprocessing/      # Image preprocessors
│   │   └── vector_store/       # FAISS vector store
│   ├── interfaces/             # Abstract interfaces
│   └── services/               # Business logic
├── data/
│   ├── products/               # Product images for indexing
│   ├── preprocessed/           # Saved preprocessed images
│   └── faiss_index/            # FAISS index files
├── tests/                      # Test suite
├── requirements-prod.txt       # Production dependencies
├── requirements-dev.txt        # Development dependencies
└── .env                        # Environment configuration
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

### Building the Vector Index

Before querying, you must build the FAISS index from your product images:

```bash
# Basic rebuild (no preprocessing saved)
python -m app.cli rebuild --products_dir data/products

# Rebuild and save preprocessed images
python -m app.cli rebuild --products_dir data/products --save_preprocessed

# Rebuild with custom output directory
python -m app.cli rebuild \
  --products_dir data/products \
  --save_preprocessed \
  --preprocessed_dir data/my_preprocessed
```

### Querying for Similar Products

```bash
# Basic query
python -m app.cli query --image path/to/query.jpg

# Query with preprocessed image saved for inspection
python -m app.cli query --image path/to/query.jpg --save_preprocessed

# Query with custom output directory
python -m app.cli query \
  --image path/to/query.jpg \
  --save_preprocessed \
  --preprocessed_dir data/query_results
```

### Classifying Images

Classify an image into categories and extract product attributes using CLIP's zero-shot classification:

```bash
# Basic classification
python -m app.cli classify --image path/to/image.jpg
```

The classifier performs a two-step classification:

1. **Category Classification** - Identifies the product category (e.g., "shoe", "bag", "clothing")
2. **Attribute Classification** - Extracts relevant attributes based on the category (e.g., color, style, material)

**Example output:**
```
Category: shoe (confidence 0.92)
Attributes:
 - type: sneaker (confidence 0.88)
 - color: red (confidence 0.85)
 - material: leather (confidence 0.72)
```

The classification uses predefined category and attribute labels. To customize:
- Edit `app/services/category_classifier_service.py` for category labels
- Edit `app/services/product_attribute_service.py` for attribute labels

### CLI Options

```
positional arguments:
  {query,rebuild,classify}  Command to run

options:
  -h, --help            Show help message
  --image IMAGE         Path to query/classify image
  --products_dir DIR    Directory of product images (for rebuild command)
  --save_preprocessed   Save preprocessed images to data/preprocessed
  --preprocessed_dir DIR
                        Custom directory for preprocessed images
                        (default: data/preprocessed)
```

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

2. **Querying (query)**
   - Loads and preprocesses the query image
   - Generates CLIP embedding
   - Searches FAISS index for nearest neighbors
   - Returns top-K similar product IDs with distance scores

3. **Classification (classify)**
   - Loads and preprocesses the query image
   - Classifies into category using CLIP zero-shot classification
   - Extracts category-specific attributes (color, style, material, etc.)
   - Returns category and attributes with confidence scores

4. **Preprocessing Pipeline**
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

## License

MIT

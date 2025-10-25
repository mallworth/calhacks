# PDF to Text Conversion

## Setup

1. **Activate virtual environment:**
   ```bash
   cd tools
   source .venv/bin/activate
   ```

2. **Install dependencies (if not already installed):**
   ```bash
   pip install pdfplumber tqdm
   ```

## Usage

1. **Add your PDFs:**
   ```bash
   # Place PDF files in the pdfs/ directory
   cp /path/to/your/*.pdf tools/pdfs/
   ```

2. **Run the conversion script:**
   ```bash
   python pdf_to_text.py
   ```

3. **Output:**
   - Text files will be created in `tools/kb/`
   - Each PDF becomes a `.txt` file with the same name
   - Page numbers are preserved in the output

## Example

```bash
# Structure before:
tools/
├── pdfs/
│   ├── survival_guide.pdf
│   └── first_aid.pdf
└── kb/

# Run conversion:
python pdf_to_text.py

# Structure after:
tools/
├── pdfs/
│   ├── survival_guide.pdf
│   └── first_aid.pdf
└── kb/
    ├── survival_guide.txt
    └── first_aid.txt
```

## Output Format

Each text file contains:
```
--- Page 1 ---
[text from page 1]

--- Page 2 ---
[text from page 2]

...
```

## Next Steps

After converting PDFs to text:
1. Use `extract_and_chunk.py` to create chunks for embedding
2. Generate embeddings using the iOS app
3. Build vector search index for RAG

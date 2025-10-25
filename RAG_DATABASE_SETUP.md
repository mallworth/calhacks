# RAG Database Setup - Complete! ✅

## What Was Created

### Database: `rag_database.db`
- **534 text chunks** from 8 knowledge base files
- **384-dimensional embeddings** for each chunk using your ONNX BERT model
- **SQLite Vec** for fast vector similarity search

### Files Created:
1. **`tools/build_rag_db.py`** - Builds the database from txt files
2. **`tools/query_rag.py`** - Query the database
3. **`vec0.dylib`** - SQLite vector extension
4. **`rag_database.db`** - Your RAG database

## Knowledge Base Sources (534 chunks total)
- 2018-First-Aid-Pocket-Guide_1.txt (3 chunks)
- How to Make Water Safe in an Emergency (4 chunks)
- USMC-Summer-Survival-Course-Handbook.txt (99 chunks)
- USMC-Winter-Survival-Course-Handbook.txt (102 chunks)
- WHO-ICRC-Basic-Emergency-Care.txt (166 chunks)
- cold-weather-survival.txt (17 chunks)
- AHA/Red Cross First Aid Guidelines (130 chunks)
- until-help-arrives-web-tutorial.txt (13 chunks)

## How to Use

### Query the database:
```bash
cd /Users/gavinlynch04/Desktop/calhacks
source .venv/bin/activate
python tools/query_rag.py "your question here"
```

### Example queries:
```bash
python tools/query_rag.py "How do I stop bleeding?"
python tools/query_rag.py "What to do for hypothermia?"
python tools/query_rag.py "How to purify water?"
python tools/query_rag.py "How do I treat a snake bite?"
python tools/query_rag.py "What to do for a broken bone?"
```

## How It Works

1. **Text is chunked** into ~500 word segments with 80 word overlap
2. **Each chunk is embedded** using your ONNX BERT model (384 dimensions)
3. **Embeddings are stored** in SQLite with the vec extension
4. **When you query**, your question is embedded and compared to all chunks
5. **Returns top 5** most similar chunks by cosine distance

## Database Schema

```sql
-- Documents table
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    source TEXT,          -- filename
    chunk_id INTEGER,     -- chunk number within file
    text TEXT,            -- actual content
    created_at TIMESTAMP
);

-- Vector search table
CREATE VIRTUAL TABLE vec_documents USING vec0(
    document_id INTEGER PRIMARY KEY,
    embedding FLOAT[384]
);
```

## Test Results

Query: "How do I stop bleeding?"
- ✅ Found relevant passages about applying pressure
- ✅ Distance: ~5.5-6.0 (good similarity)

Query: "What to do for broken bone?"
- ✅ Found fracture management passages
- ✅ Distance: ~6.0-6.2 (good similarity)

Query: "How to purify water?"
- ✅ Found CDC water safety guide
- ✅ Distance: 5.5 (excellent match!)

## Next Steps

You can now:
1. **Integrate this into your Flutter app** - call Python script from Dart
2. **Use the embeddings directly** - query from Swift/Dart using the same model
3. **Add more documents** - just add .txt files to `tools/kb/` and rebuild
4. **Adjust chunk size** - modify the `chunk_text()` parameters

## Rebuild Database (if needed)

```bash
# If you add new files to tools/kb/
cd /Users/gavinlynch04/Desktop/calhacks
source .venv/bin/activate
python tools/build_rag_db.py
```

Note: This will recreate the database from scratch with all files in `tools/kb/`.

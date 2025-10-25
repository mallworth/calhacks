#!/usr/bin/env python3
"""
Build a SQLite Vec RAG database from text files in kb/ directory.
Uses ONNX embeddings for semantic search.
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
from transformers import AutoTokenizer

def chunk_text(text: str, size: int = 500, overlap: int = 80) -> List[str]:
    """
    Chunk text into overlapping segments by word count.
    
    Args:
        text: Input text to chunk
        size: Number of words per chunk
        overlap: Number of overlapping words between chunks
    
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i+size])
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
    return chunks


def load_onnx_model(model_path: str, tokenizer_path: str):
    """Load ONNX model and tokenizer."""
    print(f"Loading ONNX model from {model_path}")
    session = ort.InferenceSession(model_path)
    
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    
    return session, tokenizer


def embed_text(text: str, session: ort.InferenceSession, tokenizer, max_length: int = 512) -> np.ndarray:
    """
    Generate embeddings for text using ONNX model.
    
    Args:
        text: Input text to embed
        session: ONNX runtime session
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    
    Returns:
        384-dimensional embedding vector
    """
    # Tokenize
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="np"
    )
    
    # Run inference
    outputs = session.run(
        ["last_hidden_state"],
        {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
            "token_type_ids": inputs["token_type_ids"].astype(np.int64)
        }
    )
    
    # Mean pooling over sequence length
    last_hidden = outputs[0]  # Shape: (batch_size, seq_len, hidden_size)
    attention_mask = inputs["attention_mask"]
    
    # Expand attention mask for broadcasting
    mask_expanded = np.expand_dims(attention_mask, -1)
    
    # Apply mask and compute mean
    sum_embeddings = np.sum(last_hidden * mask_expanded, axis=1)
    sum_mask = np.sum(mask_expanded, axis=1)
    embedding = sum_embeddings / np.maximum(sum_mask, 1e-9)
    
    return embedding[0]  # Return first (and only) batch item


def serialize_f32(vector: np.ndarray) -> bytes:
    """Serialize float32 vector to bytes for sqlite-vec."""
    return vector.astype(np.float32).tobytes()


def create_database(db_path: str):
    """Create SQLite database with vec extension."""
    print(f"Creating database at {db_path}")
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    
    # Load sqlite-vec extension
    try:
        # Try local path first (downloaded extension)
        base_dir = Path(__file__).parent.parent
        local_vec = str(base_dir / "vec0.dylib")
        
        if Path(local_vec).exists():
            conn.load_extension(local_vec)
            print(f"✓ Loaded sqlite-vec extension from {local_vec}")
        else:
            # Try system paths
            paths = [
                "vec0",
                "/usr/local/lib/vec0.dylib",
                "/opt/homebrew/lib/vec0.dylib",
            ]
            
            loaded = False
            for path in paths:
                try:
                    conn.load_extension(path)
                    loaded = True
                    print(f"✓ Loaded sqlite-vec extension from {path}")
                    break
                except:
                    continue
            
            if not loaded:
                raise Exception("sqlite-vec extension not found")
            
    except Exception as e:
        print(f"Error loading extension: {e}")
        print("\nTo install sqlite-vec:")
        print("  Download from: https://github.com/asg017/sqlite-vec/releases")
        conn.close()
        raise
    
    conn.enable_load_extension(False)
    
    # Create tables
    conn.executescript("""
        -- Main documents table
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            chunk_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Virtual table for vector similarity search
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_documents USING vec0(
            document_id INTEGER PRIMARY KEY,
            embedding FLOAT[384]
        );
        
        -- Index for faster lookups
        CREATE INDEX IF NOT EXISTS idx_source ON documents(source);
    """)
    
    conn.commit()
    return conn


def process_knowledge_base(kb_dir: Path, session, tokenizer, conn: sqlite3.Connection):
    """
    Process all text files in knowledge base directory.
    
    Args:
        kb_dir: Path to knowledge base directory
        session: ONNX runtime session
        tokenizer: Tokenizer
        conn: Database connection
    """
    txt_files = sorted(kb_dir.glob("*.txt"))
    print(f"\nFound {len(txt_files)} text files to process")
    
    total_chunks = 0
    
    for txt_file in tqdm(txt_files, desc="Processing files"):
        try:
            # Read file
            text = txt_file.read_text(encoding='utf-8')
            
            # Chunk text
            chunks = chunk_text(text, size=500, overlap=80)
            
            print(f"  {txt_file.name}: {len(chunks)} chunks")
            
            # Process each chunk
            for chunk_id, chunk in enumerate(tqdm(chunks, desc=f"  Embedding", leave=False)):
                # Generate embedding
                embedding = embed_text(chunk, session, tokenizer)
                
                # Insert into database
                cursor = conn.execute(
                    "INSERT INTO documents (source, chunk_id, text) VALUES (?, ?, ?)",
                    (txt_file.name, chunk_id, chunk)
                )
                doc_id = cursor.lastrowid
                
                # Insert embedding
                conn.execute(
                    "INSERT INTO vec_documents (document_id, embedding) VALUES (?, ?)",
                    (doc_id, serialize_f32(embedding))
                )
                
                total_chunks += 1
            
            # Commit after each file
            conn.commit()
            
        except Exception as e:
            print(f"  ✗ Error processing {txt_file.name}: {e}")
            continue
    
    print(f"\n✓ Processed {total_chunks} total chunks from {len(txt_files)} files")


def test_search(conn: sqlite3.Connection, session, tokenizer, query: str = "How do I stop bleeding?"):
    """Test the vector search functionality."""
    print(f"\n{'='*60}")
    print(f"Testing search with query: '{query}'")
    print(f"{'='*60}\n")
    
    # Generate query embedding
    query_embedding = embed_text(query, session, tokenizer)
    
    # Search for similar documents using vec0 syntax
    cursor = conn.execute("""
        SELECT 
            d.source,
            d.chunk_id,
            d.text,
            distance
        FROM vec_documents v
        JOIN documents d ON v.document_id = d.id
        WHERE embedding MATCH ? AND k = 5
        ORDER BY distance
    """, (serialize_f32(query_embedding),))
    
    results = cursor.fetchall()
    
    print(f"Top {len(results)} results:\n")
    for i, (source, chunk_id, text, distance) in enumerate(results, 1):
        print(f"{i}. [{source} - chunk {chunk_id}] (distance: {distance:.4f})")
        print(f"   {text[:200]}...")
        print()


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    kb_dir = base_dir / "tools" / "kb"
    model_path = str(base_dir / "onnx-out" / "model.onnx")
    tokenizer_path = str(base_dir / "onnx-out")
    db_path = str(base_dir / "rag_database.db")
    
    print("="*60)
    print("Building RAG Database with SQLite Vec")
    print("="*60)
    print(f"Knowledge base: {kb_dir}")
    print(f"Model: {model_path}")
    print(f"Database: {db_path}")
    print()
    
    # Load model
    session, tokenizer = load_onnx_model(model_path, tokenizer_path)
    
    # Create database
    conn = create_database(db_path)
    
    # Process knowledge base
    process_knowledge_base(kb_dir, session, tokenizer, conn)
    
    # Test search
    test_search(conn, session, tokenizer, "How do I stop severe bleeding?")
    test_search(conn, session, tokenizer, "What should I do for a broken bone?")
    test_search(conn, session, tokenizer, "How to purify water in emergency?")
    
    conn.close()
    
    print(f"\n✓ Database created successfully at: {db_path}")
    print(f"  Total documents: {conn.execute('SELECT COUNT(*) FROM documents').fetchone()[0]}")


if __name__ == "__main__":
    main()

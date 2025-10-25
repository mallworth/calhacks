#!/usr/bin/env python3
"""
Query the RAG database for similar documents.
Usage: python query_rag.py "your search query"
"""

import sys
import sqlite3
import numpy as np
import onnxruntime as ort
from pathlib import Path
from transformers import AutoTokenizer


def embed_text(text: str, session: ort.InferenceSession, tokenizer, max_length: int = 512) -> np.ndarray:
    """Generate embeddings for text using ONNX model."""
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="np"
    )
    
    outputs = session.run(
        ["last_hidden_state"],
        {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
            "token_type_ids": inputs["token_type_ids"].astype(np.int64)
        }
    )
    
    last_hidden = outputs[0]
    attention_mask = inputs["attention_mask"]
    mask_expanded = np.expand_dims(attention_mask, -1)
    sum_embeddings = np.sum(last_hidden * mask_expanded, axis=1)
    sum_mask = np.sum(mask_expanded, axis=1)
    embedding = sum_embeddings / np.maximum(sum_mask, 1e-9)
    
    return embedding[0]


def serialize_f32(vector: np.ndarray) -> bytes:
    """Serialize float32 vector to bytes for sqlite-vec."""
    return vector.astype(np.float32).tobytes()


def search(query: str, top_k: int = 5):
    """Search the RAG database for similar documents."""
    # Paths
    base_dir = Path(__file__).parent.parent
    model_path = str(base_dir / "onnx-out" / "model.onnx")
    tokenizer_path = str(base_dir / "onnx-out")
    db_path = str(base_dir / "rag_database.db")
    vec_path = str(base_dir / "vec0.dylib")
    
    # Load model
    session = ort.InferenceSession(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(vec_path)
    conn.enable_load_extension(False)
    
    # Generate query embedding
    query_embedding = embed_text(query, session, tokenizer)
    
    # Search
    cursor = conn.execute("""
        SELECT 
            d.source,
            d.chunk_id,
            d.text,
            distance
        FROM vec_documents v
        JOIN documents d ON v.document_id = d.id
        WHERE embedding MATCH ? AND k = ?
        ORDER BY distance
    """, (serialize_f32(query_embedding), top_k))
    
    results = cursor.fetchall()
    conn.close()
    
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python query_rag.py 'your search query'")
        print("\nExample queries:")
        print("  python query_rag.py 'How do I stop bleeding?'")
        print("  python query_rag.py 'What to do for hypothermia?'")
        print("  python query_rag.py 'How to purify water?'")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"{'='*70}\n")
    
    results = search(query, top_k=5)
    
    if not results:
        print("No results found.")
        return
    
    for i, (source, chunk_id, text, distance) in enumerate(results, 1):
        print(f"{i}. [{source} - chunk {chunk_id}]")
        print(f"   Distance: {distance:.4f}")
        print(f"   {text[:300]}...")
        print()


if __name__ == "__main__":
    main()

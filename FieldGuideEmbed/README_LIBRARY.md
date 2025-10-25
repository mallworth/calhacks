# FieldGuideEmbed - RAG Library for iOS

Swift library for embedding generation and semantic search over a knowledge base using BERT embeddings and SQLite.

## Overview

This library provides:
- **Embedder**: Generate 384-dimensional embeddings from text using an ONNX BERT model
- **RAGDatabase**: Perform semantic search over a pre-built knowledge base

## Features

- ✅ Pure Swift implementation compatible with iOS 17+ and macOS 13+
- ✅ ONNX Runtime for efficient model inference
- ✅ Hugging Face tokenizers for text preprocessing
- ✅ Manual L2 distance calculation (no SQLite extensions required on iOS)
- ✅ Pre-built knowledge base with 534 chunks from medical/survival documents

## Installation

### As a Package Dependency

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "path/to/FieldGuideEmbed", from: "1.0.0")
]
```

### For iOS App Integration

1. Add the package to your Xcode project
2. Bundle required files:
   - `model.onnx` (BERT model)
   - `tokenizer.json` (tokenizer)
   - `rag_database.db` (pre-built database)

## Usage

### Basic Embedding

```swift
import FieldGuideEmbed

// Initialize embedder
let embedder = try await Embedder(
    modelPath: "path/to/model.onnx",
    tokenizerDir: "path/to/tokenizer"
)

// Generate embedding
let embedding = try embedder.encode("How do I treat hypothermia?", role: "query")
print("Embedding dimension: \(embedding.count)") // 384
```

### RAG Database Search

```swift
import FieldGuideEmbed

// Initialize embedder and database
let embedder = try await Embedder(
    modelPath: Bundle.main.path(forResource: "model", ofType: "onnx")!,
    tokenizerDir: Bundle.main.resourcePath!
)

let ragDB = try RAGDatabase(
    databasePath: Bundle.main.path(forResource: "rag_database", ofType: "db")!,
    embedder: embedder
)

// Search for similar chunks
let results = try ragDB.search(query: "How do I treat hypothermia?", topK: 5)

for result in results {
    print("Source: \(result.source)")
    print("Distance: \(result.distance)")
    print("Text: \(result.text)")
    print()
}
```

### Flutter/Dart Integration

Use platform channels to call from Flutter:

```swift
// In your iOS AppDelegate or ViewController
import FieldGuideEmbed

class RAGHandler {
    private var embedder: Embedder?
    private var ragDB: RAGDatabase?
    
    func initialize() async throws {
        let modelPath = Bundle.main.path(forResource: "model", ofType: "onnx")!
        let tokenizerDir = Bundle.main.resourcePath!
        let dbPath = Bundle.main.path(forResource: "rag_database", ofType: "db")!
        
        embedder = try await Embedder(modelPath: modelPath, tokenizerDir: tokenizerDir)
        ragDB = try RAGDatabase(databasePath: dbPath, embedder: embedder!)
    }
    
    func search(query: String, topK: Int = 5) throws -> [[String: Any]] {
        let results = try ragDB!.search(query: query, topK: topK)
        return results.map { result in
            [
                "source": result.source,
                "chunk_id": result.chunkId,
                "text": result.text,
                "distance": result.distance
            ]
        }
    }
}
```

```dart
// In your Flutter app
class NativeRAG {
  static const platform = MethodChannel('com.yourapp/rag');
  
  Future<List<Map<String, dynamic>>> search(String query) async {
    try {
      final results = await platform.invokeMethod('search', {'query': query});
      return List<Map<String, dynamic>>.from(results);
    } catch (e) {
      print('Error: $e');
      return [];
    }
  }
}
```

## Building

```bash
swift build
```

## Testing

Run the included test executable:

```bash
swift run Embed
```

This will test embedding generation and similarity calculations.

## Database Structure

The RAG database contains two tables:

**documents**
- `id`: INTEGER PRIMARY KEY
- `source`: TEXT (source filename)
- `chunk_id`: INTEGER (chunk number within source)
- `text`: TEXT (chunk content)

**vec_documents**
- `document_id`: INTEGER (foreign key to documents)
- `embedding`: BLOB (384 float32 values)

## Building the Knowledge Base

To rebuild the database from source documents:

```bash
cd tools
source ../.venv/bin/activate
python build_rag_db.py
```

This processes all `.txt` files in `tools/kb/` and creates `rag_database.db`.

## Performance

- Embedding generation: ~50-100ms per query on M1 Mac
- Database search: ~100-200ms for 534 chunks (all-pairs L2 distance)
- Memory usage: ~150MB for model + embeddings

## Requirements

- Swift 5.9+
- iOS 17+ / macOS 13+
- ONNX Runtime Swift Package
- swift-transformers (Tokenizers module)

## Architecture

```
FieldGuideEmbed/
├── Sources/
│   ├── FieldGuideEmbed/          # Library target
│   │   ├── Embedder.swift        # BERT embedding generation
│   │   ├── RAGDatabase.swift     # Semantic search
│   │   └── FieldGuideEmbed.swift # Module file
│   └── Embed/                    # Test executable
│       └── main.swift            # Embedding similarity test
├── onnx-out/                     # Model files
│   ├── model.onnx
│   └── tokenizer.json
└── rag_database.db               # Pre-built knowledge base
```

## License

[Your License]

## Credits

- ONNX Runtime: https://github.com/microsoft/onnxruntime
- swift-transformers: https://github.com/huggingface/swift-transformers
- SQLite: https://www.sqlite.org/

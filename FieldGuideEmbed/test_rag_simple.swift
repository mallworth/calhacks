#!/usr/bin/env swift

// Quick test of RAG database - compile with:
// swiftc -O -I.build/debug -L.build/debug -lFieldGuideEmbed test_rag_simple.swift -o test_rag

print("This is a placeholder. To properly test RAG:")
print("1. Create a new executable target in Package.swift")
print("2. Or write iOS/Flutter integration code")
print("")
print("The FieldGuideEmbed library is now ready with:")
print("  - Embedder class for generating embeddings")
print("  - RAGDatabase class for semantic search")
print("")
print("Example usage from Swift:")
print("""
import FieldGuideEmbed

let embedder = try await Embedder(
    modelPath: "path/to/model.onnx",
    tokenizerDir: "path/to/tokenizer"
)

let ragDB = try RAGDatabase(
    databasePath: "path/to/rag_database.db",
    embedder: embedder
)

let results = try ragDB.search(query: "How do I treat hypothermia?", topK: 5)
for result in results {
    print("\\(result.source): \\(result.text)")
}
""")

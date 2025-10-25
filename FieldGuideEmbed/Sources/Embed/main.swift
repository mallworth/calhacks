import Foundation

// Helper function to compute cosine similarity between two vectors
func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count else { return 0.0 }
    
    var dotProduct: Float = 0.0
    var normA: Float = 0.0
    var normB: Float = 0.0
    
    for i in 0..<a.count {
        dotProduct += a[i] * b[i]
        normA += a[i] * a[i]
        normB += b[i] * b[i]
    }
    
    let denominator = sqrt(normA) * sqrt(normB)
    return denominator > 0 ? dotProduct / denominator : 0.0
}

func runTest() async throws {
    // Use absolute paths
    let basePath = "/Users/gavinlynch04/Desktop/calhacks"
    let modelPath = "\(basePath)/onnx-out/model.onnx"
    let tokenizerDir = "\(basePath)/onnx-out"

    print("=== Real Embedding Test ===\n")
    print("Loading model from: \(modelPath)")
    print("Loading tokenizer from: \(tokenizerDir)\n")
    
    let embedder = try await Embedder(modelPath: modelPath, tokenizerDir: tokenizerDir)
    print("✓ Model and tokenizer loaded successfully!\n")

    // Test texts with varying similarity
    let texts = [
        ("Query 1", "How do I control severe bleeding in the field?", "query"),
        ("Passage 1", "Apply direct pressure to stop bleeding. Use a clean cloth or bandage.", "passage"),
        ("Passage 2", "To control bleeding, apply firm pressure directly on the wound.", "passage"),
        ("Passage 3", "First aid for fractures involves immobilizing the injured area.", "passage"),
        ("Query 2", "What should I do for a broken bone?", "query")
    ]
    
    print("=== Generating Embeddings ===\n")
    
    // Generate embeddings
    var embeddings: [(String, [Float])] = []
    for (label, text, role) in texts {
        let vec = try embedder.encode(text, role: role)
        embeddings.append((label, vec))
        print("\(label): \(text)")
        print("  Embedding dimension: \(vec.count)")
        print("  First 8 values: \(vec.prefix(8).map { String(format: "%.4f", $0) }.joined(separator: ", "))\n")
    }
    
    // Compare similarities
    print("=== Similarity Matrix ===\n")
    print("Comparing all pairs (cosine similarity):\n")
    
    for i in 0..<embeddings.count {
        for j in (i+1)..<embeddings.count {
            let (label1, vec1) = embeddings[i]
            let (label2, vec2) = embeddings[j]
            let similarity = cosineSimilarity(vec1, vec2)
            print("\(label1) ↔ \(label2): \(String(format: "%.4f", similarity))")
        }
    }
    
    print("\n=== Analysis ===")
    print("Expected high similarity:")
    print("  - Query 1 ↔ Passage 1 & 2 (all about bleeding)")
    print("  - Query 2 ↔ Passage 3 (both about fractures/broken bones)")
    print("\nExpected lower similarity:")
    print("  - Query 1 ↔ Passage 3 (bleeding vs fractures)")
    print("  - Query 2 ↔ Passage 1 & 2 (fractures vs bleeding)")
    
    print("\n✓ Embedding test completed successfully!")
}

// Run the async function
let semaphore = DispatchSemaphore(value: 0)

Task {
    do {
        try await runTest()
        exit(0)
    } catch {
        fputs("ERROR: \(error)\n", stderr)
        exit(1)
    }
}

dispatchMain()

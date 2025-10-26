import Foundation
import OnnxRuntimeBindings
import Tokenizers

// Small helpers to bridge Swift arrays <-> ORT tensors.
extension NSMutableData {
    static func fromArray<T>(_ array: [T]) -> NSMutableData {
        array.withUnsafeBytes { buf in
            NSMutableData(bytes: buf.baseAddress!, length: buf.count)
        }
    }
}

func dataToArray<T>(_ data: NSMutableData, _: T.Type) -> [T] {
    let count = data.length / MemoryLayout<T>.size
    let ptr = data.mutableBytes.bindMemory(to: T.self, capacity: count)
    return Array(UnsafeBufferPointer(start: ptr, count: count))
}

final class Embedder {
    let session: ORTSession
    let tokenizer: any Tokenizer
    let maxLen: Int
    let addPrefixes: Bool

    init(modelPath: String,
         tokenizerDir: String,
         maxLength: Int = 512,
         addInstructionPrefixes: Bool = false) async throws {

        let env = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
        let opts = try ORTSessionOptions()
        try opts.setIntraOpNumThreads(1)
        self.session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: opts)

        // Load tokenizer from local folder
        let folderURL = URL(fileURLWithPath: tokenizerDir)
        self.tokenizer = try await AutoTokenizer.from(modelFolder: folderURL)
        self.maxLen = maxLength
        self.addPrefixes = addInstructionPrefixes
    }

    /// Split text into paragraphs (by double newline or single newline)
    private func chunkIntoParagraphs(_ text: String) -> [String] {
        // First try splitting by double newlines (paragraph boundaries)
        var paragraphs = text.components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        
        // If no double newlines, fall back to single newlines
        if paragraphs.count == 1 {
            paragraphs = text.components(separatedBy: "\n")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
        }
        
        // If still just one chunk, return as-is
        return paragraphs.isEmpty ? [text] : paragraphs
    }
    
    /// Apply masked mean pooling using attention mask
    private func maskedMeanPooling(lastHidden: [Float], mask: [Int], seqLen: Int, dim: Int) -> [Float] {
        var pooled = [Float](repeating: 0, count: dim)
        var sumMask: Float = 0
        
        for i in 0..<seqLen {
            let maskValue = Float(mask[i])
            sumMask += maskValue
            let offset = i * dim
            for d in 0..<dim {
                pooled[d] += lastHidden[offset + d] * maskValue
            }
        }
        
        // Divide by sum of mask to get mean
        if sumMask > 0 {
            for d in 0..<dim {
                pooled[d] /= sumMask
            }
        }
        
        return pooled
    }
    
    /// Apply L2 normalization to embedding vector
    private func l2Normalize(_ vector: [Float]) -> [Float] {
        let norm = sqrt(vector.reduce(0) { $0 + $1 * $1 })
        guard norm > 0 else { return vector }
        return vector.map { $0 / norm }
    }
    
    /// Encode a single chunk of text
    private func encodeChunk(_ text: String, role: String?) throws -> [Float] {
        let prefixed = addPrefixes
          ? ((role?.lowercased() == "passage") ? "passage: " : "query: ") + text
          : text

        // swift-transformers returns ONLY ids here.
        var ids = tokenizer.encode(text: prefixed, addSpecialTokens: true)

        // Optional: truncate (manual, since encode(...) doesn't accept trunc/pad)
        if ids.count > maxLen { ids = Array(ids.prefix(maxLen)) }

        // Attention mask = all ones (no padding used)
        let mask = Array(repeating: 1, count: ids.count)
        
        // Token type IDs = all zeros (single sequence)
        let tokenTypeIds = Array(repeating: 0, count: ids.count)

        // Build ORT tensors
        let idsI64  = ids.map(Int64.init)
        let maskI64 = mask.map(Int64.init)
        let typeI64 = tokenTypeIds.map(Int64.init)
        let shape: [NSNumber] = [1, NSNumber(value: ids.count)]

        let idsT  = try ORTValue(tensorData: .fromArray(idsI64),  elementType: .int64, shape: shape)
        let maskT = try ORTValue(tensorData: .fromArray(maskI64), elementType: .int64, shape: shape)
        let typeT = try ORTValue(tensorData: .fromArray(typeI64), elementType: .int64, shape: shape)

        // Request the last_hidden_state output from BERT model
        let outs = try session.run(withInputs: ["input_ids": idsT, 
                                                 "attention_mask": maskT,
                                                 "token_type_ids": typeT],
                                   outputNames: ["last_hidden_state"], runOptions: nil)

        guard let first = outs.values.first else { return [] }
        let lastHidden: [Float] = dataToArray(try first.tensorData(), Float.self)

        // Apply masked mean pooling over sequence length
        let seqLen = ids.count
        let dim = lastHidden.count / max(seqLen, 1)
        let pooled = maskedMeanPooling(lastHidden: lastHidden, mask: mask, seqLen: seqLen, dim: dim)
        
        // Apply L2 normalization
        return l2Normalize(pooled)
    }

    func encode(_ text: String, role: String? = nil) throws -> [Float] {
        // Split text into paragraphs
        let paragraphs = chunkIntoParagraphs(text)
        
        // If only one paragraph, encode directly
        if paragraphs.count == 1 {
            return try encodeChunk(text, role: role)
        }
        
        // Encode each paragraph and average the embeddings
        var allEmbeddings: [[Float]] = []
        for paragraph in paragraphs {
            let embedding = try encodeChunk(paragraph, role: role)
            if !embedding.isEmpty {
                allEmbeddings.append(embedding)
            }
        }
        
        // Average all paragraph embeddings
        guard !allEmbeddings.isEmpty else { return [] }
        let dim = allEmbeddings[0].count
        var averaged = [Float](repeating: 0, count: dim)
        
        for embedding in allEmbeddings {
            for d in 0..<dim {
                averaged[d] += embedding[d]
            }
        }
        
        let count = Float(allEmbeddings.count)
        for d in 0..<dim {
            averaged[d] /= count
        }
        
        // Apply L2 normalization to final averaged embedding
        return l2Normalize(averaged)
    }
}

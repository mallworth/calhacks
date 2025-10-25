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

    func encode(_ text: String, role: String? = nil) throws -> [Float] {
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

        // Mean-pool over sequence length → sentence embedding
        let seqLen = ids.count
        let dim = lastHidden.count / max(seqLen, 1)
        var pooled = [Float](repeating: 0, count: dim)
        var idx = 0
        for _ in 0..<seqLen {
            for d in 0..<dim { pooled[d] += lastHidden[idx + d] }
            idx += dim
        }
        if seqLen > 0 {
            for d in 0..<dim { pooled[d] /= Float(seqLen) }
        }
        return pooled
    }
}

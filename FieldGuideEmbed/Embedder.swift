// Embedder.swift
import Foundation
import OnnxRuntime
import Tokenizers

extension Data {
  static func fromArray<T>(_ array: [T]) -> Data { array.withUnsafeBufferPointer { Data(buffer: $0) } }
}
func dataToArray<T>(_ data: Data, _: T.Type) -> [T] {
  data.withUnsafeBytes { Array($0.bindMemory(to: T.self)) }
}

func meanPoolAndNormalize(lastHidden: [Float], seqLen: Int, hidden: Int, mask: [Int64]) -> [Float] {
  var out = [Float](repeating: 0, count: hidden)
  var denom: Float = 0
  for t in 0..<seqLen where mask[t] == 1 {
    denom += 1
    let base = t * hidden
    for h in 0..<hidden { out[h] += lastHidden[base + h] }
  }
  if denom > 0 {
    let inv = 1.0 / denom
    for h in 0..<hidden { out[h] *= Float(inv) }
  }
  var n: Float = 0; for v in out { n += v*v }
  if n > 0 { let inv = 1/sqrtf(n); for i in 0..<out.count { out[i] *= inv } }
  return out
}

public final class Embedder {
  let session: ORTSession
  let tokenizer: Tokenizer
  let maxLen: Int
  let addPrefixes: Bool  // set true for E5/BGE, false for MiniLM

  public init(modelPath: String, tokenizerPath: String, maxLength: Int = 512, addInstructionPrefixes: Bool = false) throws {
    let env = try ORTEnv(loggingLevel: .warning)
    let opts = try ORTSessionOptions()
    try opts.setIntraOpNumThreads(1)
    self.session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: opts)
    self.tokenizer = try Tokenizer(fromFile: tokenizerPath)
    self.maxLen = maxLength
    self.addPrefixes = addInstructionPrefixes
  }

  /// role: "query" or "passage" (only matters if addPrefixes = true)
  public func embed(_ text: String, role: String = "query") throws -> [Float] {
    let prefixed: String = (addPrefixes
                             ? (role.lowercased() == "passage" ? "passage: " : "query: ") + text
                             : text)
    let enc = try tokenizer.encode(prefixed,
                                   addSpecialTokens: true,
                                   truncation: .longestFirst,
                                   maxLength: maxLen,
                                   padding: .maxLength(maxLen))
    let ids  = enc.ids.map(Int64.init)
    let mask = enc.attentionMask.map(Int64.init)
    let shape: [NSNumber] = [1, NSNumber(value: ids.count)]
    let idsT  = try ORTValue(tensorData: .fromArray(ids),  elementType: .int64, shape: shape)
    let maskT = try ORTValue(tensorData: .fromArray(mask), elementType: .int64, shape: shape)

    let outs = try session.run(withInputs: ["input_ids": idsT, "attention_mask": maskT],
                               outputNames: nil, runOptions: nil)
    let lastHidden: [Float] = dataToArray(try outs[0].tensorData(), Float.self)
    let hidden = lastHidden.count / ids.count
    return meanPoolAndNormalize(lastHidden: lastHidden, seqLen: ids.count, hidden: hidden, mask: mask)
  }
}

// tiny helpers
public func toCSV(_ v: [Float]) -> String { v.map { String(format: "%.6f", $0) }.joined(separator: ",") }
public func toBase64(_ v: [Float]) -> String {
  var vv = v; return vv.withUnsafeMutableBytes { Data($0) }.base64EncodedString()
}

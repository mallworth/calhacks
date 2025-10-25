// main.swift
import Foundation
import OnnxRuntime
import Tokenizers

// Reuse Embedder.swift types
// Build command:
// swift build -c release
// Usage:
// swift run -c release Embed query --model model.onnx --tokenizer tokenizer.json --text "..." [--fmt csv|base64] [--role query|passage]
// swift run -c release Embed docs  --model model.onnx --tokenizer tokenizer.json --in docs.jsonl --out embeds.jsonl [--fmt base64] [--role passage]

enum Mode { case query, docs }

struct Args {
  var mode: Mode = .query
  var model = "model.onnx"
  var tok   = "tokenizer.json"
  var text  = ""
  var input = ""      // docs.jsonl (lines: {"id":"...","text":"..."})
  var output = "embeds.jsonl"
  var fmt   = "csv"   // csv|base64
  var role  = "query" // query|passage (for E5/BGE)
  var maxLen = 512
  var prefixes = true // true for E5/BGE; false for MiniLM
}

func parseArgs() -> Args {
  var a = Args()
  var it = CommandLine.arguments.dropFirst().makeIterator()
  guard let cmd = it.next() else { return a }
  a.mode = (cmd == "docs") ? .docs : .query
  while let k = it.next() {
    switch k {
      case "--model": a.model = it.next() ?? a.model
      case "--tokenizer", "--tok": a.tok = it.next() ?? a.tok
      case "--text": a.text = it.next() ?? ""
      case "--in": a.input = it.next() ?? ""
      case "--out": a.output = it.next() ?? a.output
      case "--fmt": a.fmt = it.next() ?? a.fmt
      case "--role": a.role = it.next() ?? a.role
      case "--maxlen": a.maxLen = Int(it.next() ?? "") ?? a.maxLen
      case "--prefixes": a.prefixes = (it.next() ?? "true") != "false"
      default: break
    }
  }
  return a
}

struct Doc: Decodable { let id: String; let text: String }

let a = parseArgs()
let embedder = try Embedder(modelPath: a.model, tokenizerPath: a.tok, maxLength: a.maxLen, addInstructionPrefixes: a.prefixes)

switch a.mode {
case .query:
  guard !a.text.isEmpty else {
    fputs("Usage: Embed query --model model.onnx --tokenizer tokenizer.json --text \"...\" [--fmt csv|base64] [--role query|passage] [--prefixes true|false]\n", stderr)
    exit(2)
  }
  let v = try embedder.embed(a.text, role: a.role)
  print(a.fmt == "base64" ? toBase64(v) : toCSV(v))

case .docs:
  guard !a.input.isEmpty else {
    fputs("Usage: Embed docs --model model.onnx --tokenizer tokenizer.json --in docs.jsonl --out embeds.jsonl [--fmt csv|base64] [--role passage] [--prefixes true|false]\n", stderr)
    exit(2)
  }
  let h = try String(contentsOfFile: a.input, encoding: .utf8)
  guard let out = OutputStream(toFileAtPath: a.output, append: false) else { fatalError("open \(a.output) failed") }
  out.open(); defer { out.close() }
  func writeln(_ s: String) {
    let d = (s + "\n").data(using: .utf8)!
    _ = d.withUnsafeBytes { out.write($0.bindMemory(to: UInt8.self).baseAddress!, maxLength: d.count) }
  }
  var n = 0
  for line in h.split(whereSeparator: \.isNewline) {
    if line.trimmingCharacters(in: .whitespaces).isEmpty { continue }
    let d = try JSONDecoder().decode(Doc.self, from: Data(line.utf8))
    let v = try embedder.embed(d.text, role: a.role)
    let payload = (a.fmt == "base64" ? toBase64(v) : toCSV(v))
    writeln(#"{"id":"\#(d.id)","embedding":"\#(payload)"}"#)
    n += 1
  }
  print("Wrote \(n) embeddings to \(a.output)")
}

import Flutter
import Foundation
import SQLite3

final class RAGService {
  private enum RAGError: Error {
    case resourceMissing(String)
    case databaseOpen(String)
    case queryPrepare(String)
  }
  
  private let channel: FlutterMethodChannel
  private var db: OpaquePointer?
  private let queue = DispatchQueue(label: "survival.rag.queue")
  
  init?(binaryMessenger: FlutterBinaryMessenger) {
    channel = FlutterMethodChannel(name: "survival/rag", binaryMessenger: binaryMessenger)
    do {
      let databaseURL = try prepareResources()
      try openDatabase(at: databaseURL)
      channel.setMethodCallHandler(handleMethodCall)
    } catch {
      print("❌ RAGService init failed: \(error)")
      closeDatabase()
      return nil
    }
  }
  
  deinit {
    closeDatabase()
  }
  
  private func prepareResources() throws -> URL {
    guard let bundleDB = Bundle.main.url(forResource: "rag_database", withExtension: "db") else {
      throw RAGError.resourceMissing("rag_database.db not bundled")
    }
    let fileManager = FileManager.default
    let directory = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("RAG", isDirectory: true)
    try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
    
    let dbURL = directory.appendingPathComponent("rag_database.db")
    if fileManager.fileExists(atPath: dbURL.path) {
      try fileManager.removeItem(at: dbURL)
    }
    try fileManager.copyItem(at: bundleDB, to: dbURL)
    
    return dbURL
  }
  
  private func openDatabase(at url: URL) throws {
    var connection: OpaquePointer?
    if sqlite3_open(url.path, &connection) != SQLITE_OK {
      let message = connection.flatMap { String(cString: sqlite3_errmsg($0)) } ?? "Unknown"
      sqlite3_close(connection)
      throw RAGError.databaseOpen(message)
    }
    db = connection
  }
  
  private func handleMethodCall(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard call.method == "search" else {
      result(FlutterMethodNotImplemented)
      return
    }
    guard let args = call.arguments as? [String: Any],
          let embeddingValues = args["embedding"] as? [Double] else {
      result(FlutterError(code: "INVALID_ARGS",
                          message: "Expected { 'embedding': [double], 'topK': int? }",
                          details: nil))
      return
    }
    let topK = max(1, (args["topK"] as? Int) ?? 3)
    let embedding = embeddingValues.map(Float.init)
    queue.async { [weak self] in
      guard let self else { return }
      do {
        let rows = try self.search(embedding: embedding, topK: topK)
        DispatchQueue.main.async {
          result(rows)
        }
      } catch {
        DispatchQueue.main.async {
          result(FlutterError(code: "QUERY_FAILED",
                               message: error.localizedDescription,
                               details: nil))
        }
      }
    }
  }
  
  private func search(embedding: [Float], topK: Int) throws -> [[String: Any]] {
    guard let db else { throw RAGError.databaseOpen("Database not open") }
    let sql = """
      SELECT d.source, d.chunk_id, d.text, e.embedding
      FROM documents d
      JOIN document_embeddings e ON d.id = e.document_id
    """
    var statement: OpaquePointer?
    if sqlite3_prepare_v2(db, sql, -1, &statement, nil) != SQLITE_OK {
      let message = String(cString: sqlite3_errmsg(db))
      throw RAGError.queryPrepare(message)
    }
    defer { sqlite3_finalize(statement) }
    
    var rows: [(source: String, chunkId: Int, text: String, score: Double)] = []
    while sqlite3_step(statement) == SQLITE_ROW {
      let source = String(cString: sqlite3_column_text(statement, 0))
      let chunkId = Int(sqlite3_column_int(statement, 1))
      let text = String(cString: sqlite3_column_text(statement, 2))
      guard let blobPtr = sqlite3_column_blob(statement, 3) else { continue }
      let blobSize = Int(sqlite3_column_bytes(statement, 3))
      let vectorCount = blobSize / MemoryLayout<Float>.size
      let vector = Array(UnsafeBufferPointer(start: blobPtr.assumingMemoryBound(to: Float.self),
                                             count: vectorCount))
      let score = cosineSimilarity(embedding, vector)
      rows.append((source: source, chunkId: chunkId, text: text, score: Double(score)))
    }
    let top = rows.sorted { $0.score > $1.score }.prefix(topK)
    return top.map {
      [
        "source": $0.source,
        "chunkId": $0.chunkId,
        "text": $0.text,
        "score": $0.score
      ]
    }
  }

  private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count, !a.isEmpty else { return 0.0 }
    var dot: Float = 0.0
    var normA: Float = 0.0
    var normB: Float = 0.0
    for i in 0..<a.count {
      let av = a[i]
      let bv = b[i]
      dot += av * bv
      normA += av * av
      normB += bv * bv
    }
    let denom = (normA > 0 && normB > 0) ? sqrt(normA) * sqrt(normB) : 0.0
    guard denom > 0 else { return 0.0 }
    return max(-1.0, min(1.0, dot / denom))
  }
  
  private func closeDatabase() {
    if let db {
      sqlite3_close(db)
    }
    db = nil
  }
}

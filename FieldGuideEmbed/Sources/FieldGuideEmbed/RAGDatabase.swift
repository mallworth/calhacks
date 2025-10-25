import Foundation
import SQLite3

/// A result from the RAG database search
public struct RAGResult {
    public let source: String
    public let chunkId: Int
    public let text: String
    public let distance: Float
    
    public init(source: String, chunkId: Int, text: String, distance: Float) {
        self.source = source
        self.chunkId = chunkId
        self.text = text
        self.distance = distance
    }
}

/// RAG Database for semantic search using SQLite Vec
/// Note: For iOS, the database must be pre-built with embeddings using Python tools
/// This class performs manual distance calculations since SQLite extensions can't be loaded on iOS
public class RAGDatabase {
    private var db: OpaquePointer?
    private let embedder: Embedder
    
    /// Initialize RAG database
    /// - Parameters:
    ///   - databasePath: Path to the SQLite database
    ///   - embedder: Embedder instance for generating query embeddings
    public init(databasePath: String, embedder: Embedder) throws {
        self.embedder = embedder
        
        // Open database
        guard sqlite3_open(databasePath, &db) == SQLITE_OK else {
            let errmsg = String(cString: sqlite3_errmsg(db)!)
            sqlite3_close(db)
            throw NSError(domain: "RAGDatabase", code: 1, 
                         userInfo: [NSLocalizedDescriptionKey: "Failed to open database: \(errmsg)"])
        }
    }
    
    deinit {
        sqlite3_close(db)
    }
    
    /// Search the database for semantically similar chunks
    /// - Parameters:
    ///   - query: The search query text
    ///   - topK: Number of results to return (default: 5)
    /// - Returns: Array of RAGResult sorted by similarity
    public func search(query: String, topK: Int = 5) throws -> [RAGResult] {
        // Generate embedding for query
        let queryEmbedding = try embedder.encode(query, role: "query")
        
        // Fetch all documents with their embeddings
        let sql = """
            SELECT 
                d.id,
                d.source,
                d.chunk_id,
                d.text,
                v.embedding
            FROM documents d
            JOIN vec_documents v ON d.id = v.document_id
        """
        
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else {
            let errmsg = String(cString: sqlite3_errmsg(db)!)
            throw NSError(domain: "RAGDatabase", code: 4,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to prepare query: \(errmsg)"])
        }
        
        defer {
            sqlite3_finalize(statement)
        }
        
        // Calculate distances for all documents
        var candidates: [(source: String, chunkId: Int, text: String, distance: Float)] = []
        
        while sqlite3_step(statement) == SQLITE_ROW {
            let source = String(cString: sqlite3_column_text(statement, 1))
            let chunkId = Int(sqlite3_column_int(statement, 2))
            let text = String(cString: sqlite3_column_text(statement, 3))
            
            // Get embedding blob
            guard let blobPtr = sqlite3_column_blob(statement, 4) else { continue }
            let blobSize = Int(sqlite3_column_bytes(statement, 4))
            
            // Convert blob to float array
            let docEmbedding = Array(UnsafeBufferPointer(
                start: blobPtr.assumingMemoryBound(to: Float.self),
                count: blobSize / MemoryLayout<Float>.size
            ))
            
            // Calculate L2 distance
            let distance = Self.l2Distance(queryEmbedding, docEmbedding)
            
            candidates.append((source: source, chunkId: chunkId, text: text, distance: distance))
        }
        
        // Sort by distance and take top K
        let topResults = candidates.sorted { $0.distance < $1.distance }.prefix(topK)
        
        return topResults.map { candidate in
            RAGResult(
                source: candidate.source,
                chunkId: candidate.chunkId,
                text: candidate.text,
                distance: candidate.distance
            )
        }
    }
    
    /// Calculate L2 (Euclidean) distance between two vectors
    private static func l2Distance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return Float.infinity }
        
        var sum: Float = 0.0
        for i in 0..<a.count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }
    
    /// Get total number of chunks in the database
    public func getChunkCount() throws -> Int {
        let sql = "SELECT COUNT(*) FROM documents"
        var statement: OpaquePointer?
        
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else {
            let errmsg = String(cString: sqlite3_errmsg(db)!)
            throw NSError(domain: "RAGDatabase", code: 5,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to get count: \(errmsg)"])
        }
        
        defer {
            sqlite3_finalize(statement)
        }
        
        guard sqlite3_step(statement) == SQLITE_ROW else {
            return 0
        }
        
        return Int(sqlite3_column_int(statement, 0))
    }
}

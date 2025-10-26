import Flutter
import Foundation

class EmbedService {
  private let channel: FlutterMethodChannel
  private var embedder: Embedder?
  private var isInitializing = false
  private var initializationError: Error?
  
  init(binaryMessenger: FlutterBinaryMessenger) {
    self.channel = FlutterMethodChannel(
      name: "survival/embed",
      binaryMessenger: binaryMessenger
    )
    
    self.channel.setMethodCallHandler(handleMethodCall)
    
    // Initialize embedder asynchronously
    isInitializing = true
    Task {
      do {
        guard let modelPath = Bundle.main.path(forResource: "bge-small-en-v1.5", ofType: "onnx") else {
          throw NSError(domain: "EmbedService", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model file not found. Expected: Models/bge-small-en-v1.5.onnx"])
        }

        print("🔄 Loading embedder from:")
        print("   Model: \(modelPath)")
        
        // Try looking in the main bundle root first, then try subdirectory
        var tokenizerURL = Bundle.main.url(forResource: "tokenizer", withExtension: "json")
        if tokenizerURL == nil {
          tokenizerURL = Bundle.main.url(forResource: "tokenizer", withExtension: "json", subdirectory: "Models")
        }
        
        guard let tokenizerURL = tokenizerURL else {
          throw NSError(domain: "EmbedService", code: 2, userInfo: [NSLocalizedDescriptionKey: "Tokenizer file not found in bundle"])
        }
        let tokenizerDir = tokenizerURL.deletingLastPathComponent().path
        
        print("   Tokenizer: \(tokenizerDir)")
        
        self.embedder = try await Embedder(modelPath: modelPath, tokenizerDir: tokenizerDir)
        self.isInitializing = false
        print("✅ Embedder loaded successfully")
      } catch {
        print("❌ Failed to load embedder: \(error)")
        self.initializationError = error
        self.isInitializing = false
      }
    }
  }
  
  private func handleMethodCall(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard call.method == "embed" else {
      result(FlutterMethodNotImplemented)
      return
    }
    
    guard let args = call.arguments as? [String: Any],
          let text = args["text"] as? String else {
      result(FlutterError(code: "INVALID_ARGS", message: "Expected {'text': '<string>'}", details: nil))
      return
    }
    
    // Check for initialization error
    if let error = initializationError {
      result(FlutterError(code: "INIT_ERROR", message: "Failed to initialize: \(error.localizedDescription)", details: nil))
      return
    }
    
    // Wait for initialization if still loading
    if isInitializing {
      result(FlutterError(code: "NOT_READY", message: "Embedder is still loading. Please wait a moment and try again.", details: nil))
      return
    }
    
    guard let embedder = embedder else {
      result(FlutterError(code: "NOT_READY", message: "Embedder not initialized", details: nil))
      return
    }
    
    // Run inference asynchronously
    DispatchQueue.global(qos: .userInitiated).async {
      do {
        let embedding = try embedder.encode(text)
        DispatchQueue.main.async {
          result(embedding.map { Double($0) })
        }
      } catch {
        DispatchQueue.main.async {
          result(FlutterError(code: "INFERENCE_ERROR", message: error.localizedDescription, details: nil))
        }
      }
    }
  }
}

import Flutter
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXRandom
import Hub
import Metal

/// LLMService handles local LLM inference using MLX Swift
final class LLMService {

  // Toggle to use the real local LLM or a mock response
  private let USE_REAL_LLM = true

  private let channel: FlutterMethodChannel
  private var modelContainer: ModelContainer?
  private var isInitializing = false
  private var initializationError: Error?
  private var isReady = false
  // Status/progress for copy/load
  private var statusState: String = "idle" // idle|copying|loading|ready|error
  private var statusProgress: Double = 0.0  // 0.0..1.0 when copying
  private var statusMessage: String = ""

  // Local cache base for MLX Hub
  private lazy var appSupportMLXCacheURL: URL? = {
    do {
      let base = try FileManager.default.url(
        for: .applicationSupportDirectory,
        in: .userDomainMask,
        appropriateFor: nil,
        create: true
      ).appendingPathComponent("MLXModels", isDirectory: true)
      try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
      return base
    } catch {
      print("⚠️ Could not create MLX cache dir: \(error)")
      return nil
    }
  }()

  // System prompt for FieldGuide assistant
  private let systemPrompt = """
  You are "FieldGuide," an offline-first first-aid/survival assistant.
  Answer ONLY from CONTEXT. If key info is missing, say so and give universal, time-critical safety steps (e.g., call emergency services, scene safety, direct pressure).
  Be concise (<=200 words). Number the actions. Cite every actionable step with [D#]. No internal reasoning.
  Sections: Title / RED FLAGS / DO NOW - Step-by-step / When to Escalate / What to Avoid / Sources / confidence: <high|medium|low>.
  """

  // Model + generation defaults (tune as needed)
  // Using a tiny 4-bit quantized model to reduce memory usage
  private let modelID = "mlx-community/Qwen2.5-0.5B-4bit"
  private let maxTokens = 500
  private let temperature: Float = 0.7
  private let topP: Float = 0.9

  init(binaryMessenger: FlutterBinaryMessenger) {
    self.channel = FlutterMethodChannel(
      name: "survival/llm",
      binaryMessenger: binaryMessenger
    )
    self.channel.setMethodCallHandler(handleMethodCall)

    // Check Metal availability
    #if targetEnvironment(simulator)
    print("⚠️ LLMService: Running on simulator - MLX may not work without Metal")
    #else
    print("✅ LLMService: Running on physical device")
    #endif
    
    // Check if Metal is available
    if let device = MTLCreateSystemDefaultDevice() {
      print("✅ Metal device available: \(device.name)")
    } else {
      print("❌ Metal device NOT available - MLX will fail!")
    }
    
    // Log initial memory usage
    logMemoryUsage(label: "LLMService init")

    // Do NOT auto-initialize - wait for user to trigger download
    if USE_REAL_LLM {
      print("✅ LLMService initialized - model will download on user request")
      print("📦 Model: \(modelID) (smaller model for memory constraints)")
    } else {
      print("✅ LLMService ready (mock mode - enable USE_REAL_LLM for real model)")
      self.isReady = true
    }
  }
  
  // Memory monitoring helper
  private func logMemoryUsage(label: String) {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
    let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
      $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
        task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
      }
    }
    
    if kerr == KERN_SUCCESS {
      let usedMB = Double(info.resident_size) / 1024.0 / 1024.0
      print("💾 Memory [\(label)]: \(String(format: "%.1f", usedMB)) MB")
    }
  }

  // MARK: - Initialization

  private func initializeRealLLMIfNeeded() {
    // Prevent duplicate inits
    if isReady || isInitializing { return }
    isInitializing = true
    statusState = "loading"
    statusMessage = "Loading model…"

    Task.detached(priority: .userInitiated) { [weak self] in
      guard let self else { return }
      do {
        print("🔄 Loading MLX LLM model…")
        self.logMemoryUsage(label: "Before model load")
        
        // Check if cache directory exists and log contents
        if let base = self.appSupportMLXCacheURL {
          print("📦 Cache directory: \(base.path)")
          let modelPath = base.appendingPathComponent("models--\(self.modelID.replacingOccurrences(of: "/", with: "--"))")
          if FileManager.default.fileExists(atPath: modelPath.path) {
            print("✓ Found existing model cache at: \(modelPath.path)")
            if let contents = try? FileManager.default.contentsOfDirectory(atPath: modelPath.path) {
              print("📁 Cache contents: \(contents.joined(separator: ", "))")
            }
          } else {
            print("⚠️ No existing model cache found - will download from HuggingFace")
          }
        }

        let configuration = ModelConfiguration(id: self.modelID)

        // Prefer local cache path if available; falls back to default Hub path.
        let hub: HubApi
        if let base = self.appSupportMLXCacheURL {
          hub = HubApi(downloadBase: base, useBackgroundSession: false)
          print("📦 Using local Hub cache at: \(base.path)")
        } else {
          hub = HubApi()
        }

        var lastProgress: Double = 0.0
        let container = try await LLMModelFactory.shared.loadContainer(
          hub: hub,
          configuration: configuration
        ) { progress in
          let pct = Int(progress.fractionCompleted * 100.0)
          // Only log every 5% to avoid spam
          if progress.fractionCompleted - lastProgress >= 0.05 || progress.fractionCompleted == 1.0 {
            print("📥 Model download progress: \(pct)% (total: \(progress.totalUnitCount) bytes, completed: \(progress.completedUnitCount) bytes)")
            lastProgress = progress.fractionCompleted
          }
          self.statusState = "loading"
          self.statusProgress = progress.fractionCompleted
          self.statusMessage = "Downloading or verifying model (\(pct)%)"
        }

        print("🔍 Verifying model is usable...")
        self.logMemoryUsage(label: "After model load, before verify")
        // Test that we can access the model
        try await container.perform { (modelContext: ModelContext) in
          print("✓ Model context created successfully")
          print("✓ Model weights loaded: \(modelContext.model)")
        }
        self.logMemoryUsage(label: "After model verify")

        self.modelContainer = container
        self.isReady = true
        self.isInitializing = false
        self.initializationError = nil
        self.statusState = "ready"
        self.statusProgress = 1.0
        self.statusMessage = "Ready"
        print("✅ MLX LLM model loaded and verified successfully")
        self.logMemoryUsage(label: "Model ready")
      } catch {
        self.initializationError = error
        self.isInitializing = false
        self.isReady = false
        self.statusState = "error"
        self.statusMessage = "Init failed: \(error.localizedDescription)"
        print("❌ Failed to load MLX LLM: \(error)")
        print("❌ Error details: \(error)")
      }
    }
  }

  // MARK: - Flutter method handling

  private func handleMethodCall(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    if call.method == "status" {
      // Report current status
      let dict: [String: Any] = [
        "state": statusState,
        "progress": statusProgress,
        "message": statusMessage,
        "ready": isReady
      ]
      result(dict)
      return
    }

    if call.method == "downloadModel" {
      print("📲 downloadModel called from Flutter")
      // Start model download + initialization
      if isReady {
        print("⚠️ Model already ready, rejecting download request")
        result(FlutterError(code: "ALREADY_READY", message: "Model already downloaded", details: nil))
        return
      }
      if isInitializing {
        print("⚠️ Model already initializing, rejecting duplicate request")
        result(FlutterError(code: "ALREADY_DOWNLOADING", message: "Model download already in progress", details: nil))
        return
      }
      
      print("✅ Starting model download...")
      // Start the download
      initializeRealLLMIfNeeded()
      result(nil) // Success - download started
      return
    }

    guard call.method == "generate" else {
      result(FlutterMethodNotImplemented)
      return
    }

    guard
      let args = call.arguments as? [String: Any],
      let prompt = args["prompt"] as? String
    else {
      result(FlutterError(
        code: "INVALID_ARGS",
        message: "Expected {'prompt': '<string>', 'context': '<string>' (optional)}",
        details: nil
      ))
      return
    }

    let context = (args["context"] as? String) ?? ""

    // Check if model is ready
    if let error = initializationError {
      result(FlutterError(
        code: "INIT_ERROR",
        message: "Failed to initialize LLM: \(error.localizedDescription)",
        details: nil
      ))
      return
    }

    guard isReady else {
      result(FlutterError(
        code: "NOT_READY",
        message: "LLM is still loading. Try again shortly.",
        details: nil
      ))
      return
    }

    // Dispatch the generation without blocking the main thread.
    Task.detached(priority: .userInitiated) { [weak self] in
      guard let self else { return }
      let reply: String
      if self.USE_REAL_LLM, let container = self.modelContainer {
        do {
          print("🤖 Starting LLM generation...")
          print("📝 Prompt length: \(prompt.count) chars, Context length: \(context.count) chars")
          self.logMemoryUsage(label: "Before generation")
          reply = try await self.generateRealResponseAsync(prompt: prompt, context: context, container: container)
          print("✅ LLM generation completed, response length: \(reply.count) chars")
          self.logMemoryUsage(label: "After generation")
        } catch {
          print("❌ LLM generation error: \(error)")
          print("❌ Error type: \(type(of: error))")
          if let nsError = error as NSError? {
            print("❌ Error domain: \(nsError.domain), code: \(nsError.code)")
            print("❌ Error userInfo: \(nsError.userInfo)")
          }
          reply = "Error generating response: \(error.localizedDescription)"
        }
      } else {
        reply = self.generateMockResponse(prompt: prompt, context: context)
      }

      // Send the response back to Flutter (must hop to main)
      DispatchQueue.main.async {
        result(reply)
      }
    }
  }

  // MARK: - Real LLM generation (async)

  private func generateRealResponseAsync(
    prompt: String,
    context: String,
    container: ModelContainer
  ) async throws -> String {

    // Qwen2.5 chat template format
    let fullPrompt: String
    if context.isEmpty {
      fullPrompt = """
      <|im_start|>system
      \(systemPrompt)<|im_end|>
      <|im_start|>user
      \(prompt)<|im_end|>
      <|im_start|>assistant
      """
    } else {
      fullPrompt = """
      <|im_start|>system
      \(systemPrompt)
      
      CONTEXT:
      \(context)<|im_end|>
      <|im_start|>user
      \(prompt)<|im_end|>
      <|im_start|>assistant
      """
    }

    print("🎲 Setting random seed...")
    // Seed for controlled randomness (optional)
    MLXRandom.seed(UInt64(Date().timeIntervalSince1970 * 1000))

    print("⚙️ Creating generation parameters (maxTokens: \(maxTokens), temp: \(temperature), topP: \(topP))...")
    let parameters = GenerateParameters(
      maxTokens: maxTokens,
      temperature: temperature,
      topP: topP
    )

    var response = ""
    print("🔄 Starting model.perform...")
    try await container.perform { (modelContext: ModelContext) in
      print("✓ Model context obtained, preparing input...")
      let input = try await modelContext.processor.prepare(input: UserInput(prompt: fullPrompt))
      print("✓ Input prepared, starting generation stream...")
      let stream = try MLXLMCommon.generate(
        input: input,
        parameters: parameters,
        context: modelContext
      )

      print("📖 Reading generation stream...")
      var tokenCount = 0
      let stopTokens = ["<|im_end|>", "<|im_start|>"]
      var shouldStop = false
      
      for try await generation in stream {
        if shouldStop { break }
        
        if let chunk = generation.chunk {
          response += chunk
          tokenCount += 1
          
          // Check if we've hit a stop token
          for stopToken in stopTokens {
            if response.hasSuffix(stopToken) {
              // Remove the stop token from response
              response = String(response.dropLast(stopToken.count))
              print("🛑 Stop token detected, ending generation at \(tokenCount) tokens")
              shouldStop = true
              break
            }
          }
          
          if tokenCount % 10 == 0 {
            print("📝 Generated \(tokenCount) tokens so far...")
          }
        }
      }
      print("✅ Stream completed, total tokens: \(tokenCount)")
    }

    return response.trimmingCharacters(in: .whitespacesAndNewlines)
  }

  // MARK: - Mock response (offline demo)

  private func generateMockResponse(prompt: String, context: String) -> String {
    // Simulate processing time
    Thread.sleep(forTimeInterval: 1.0)

    // Parse context to extract [D#] sources anywhere in text
    let sources = extractSources(from: context)
    let sourceRefs = sources.isEmpty ? "" : "\n\nSources: " + sources.joined(separator: ", ")

    return """
      **Severe Bleeding Control**

      RED FLAGS:
      • Spurting or pulsating blood
      • Blood soaking through multiple bandages
      • Signs of shock (pale, cold, rapid pulse)
      • Bleeding that won't stop after 10 minutes of pressure

      DO NOW - Step-by-step:
      1. [D1] Ensure scene safety - wear gloves if available
      2. [D1] Apply firm, direct pressure with clean cloth/bandage
      3. [D1] If blood soaks through, add layers - DO NOT remove first dressing
      4. [D2] Elevate the wound above heart level if possible
      5. [D3] Use a tourniquet ONLY for life-threatening limb bleeding when pressure fails

      When to Escalate:
      • Call 911 immediately for severe bleeding
      • Bleeding doesn't slow after 10 min of pressure
      • Any signs of shock appear
      • Wound is very large or deep

      What to Avoid:
      • Don't remove the first bandage once applied
      • Don't use a tourniquet unless absolutely necessary
      • Don't probe or clean wound before bleeding is controlled
      \(sourceRefs)

      Confidence: high

      ---
      This is a DEMO response. Enable USE_REAL_LLM for real inference.
      """
  }

  // MARK: - Utilities

  /// Extract all bracketed citations like [D1], [D12], etc. occurring anywhere in the text.
  private func extractSources(from text: String) -> [String] {
    let pattern = #"\[D\d+\]"#
    guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else { return [] }
    let range = NSRange(text.startIndex..<text.endIndex, in: text)
    let matches = regex.matches(in: text, options: [], range: range)
    var tokens: [String] = []
    tokens.reserveCapacity(matches.count)
    for m in matches {
      if let r = Range(m.range, in: text) {
        tokens.append(String(text[r]))
      }
    }
    // Deduplicate, preserve order
    var seen = Set<String>()
    return tokens.filter { seen.insert($0).inserted }
  }
}

private extension FileManager {
  func removeItemIfExists(at url: URL) throws {
    if fileExists(atPath: url.path) {
      try removeItem(at: url)
    }
  }
}

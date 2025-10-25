import Flutter
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXRandom
import Hub

/// LLMService handles local LLM inference using MLX Swift
final class LLMService {
  private let USE_REAL_LLM = true
  
  private let channel: FlutterMethodChannel
  private var modelContainer: ModelContainer?
  private var isInitializing = false
  private var initializationError: Error?
  private var isReady = false
  
  // System prompt for FieldGuide assistant
  private let systemPrompt = """
You are "FieldGuide," an offline-first first-aid/survival assistant.
Answer ONLY from CONTEXT. If key info is missing, say so and give universal, time-critical safety steps (e.g., call emergency services, scene safety, direct pressure).
Be concise (≤200 words). Number the actions. Cite every actionable step with [D#]. No internal reasoning.
Sections: Title / RED FLAGS / DO NOW — Step-by-step / When to Escalate / What to Avoid / Sources / confidence: <high|medium|low>.
"""
  
  init(binaryMessenger: FlutterBinaryMessenger) {
    self.channel = FlutterMethodChannel(
      name: "survival/llm",
      binaryMessenger: binaryMessenger
    )
    
    self.channel.setMethodCallHandler(handleMethodCall)
    
    if USE_REAL_LLM {
      // Real MLC-LLM initialization
      isInitializing = true
      DispatchQueue.global(qos: .userInitiated).async { [weak self] in
        guard let self else { return }
        self.initializeRealLLM()
      }
    } else {
      // Mock mode - simulate initialization delay
      DispatchQueue.global(qos: .userInitiated).asyncAfter(deadline: .now() + 2.0) { [weak self] in
        self?.isReady = true
        print("✅ LLMService ready (mock mode - see LLMService.swift to enable real LLM)")
      }
    }
  }
  
  private func initializeRealLLM() {
    Task {
      do {
        print("🔄 Loading MLX LLM model...")
        
        // Use Phi-3 mini 4bit - good for medical/survival tasks
        let modelConfiguration = ModelConfiguration(
          id: "mlx-community/Phi-3-mini-4k-instruct-4bit"
        )
        
        // Load model using LLMModelFactory
        let factory = LLMModelFactory.shared
        let modelContainer = try await factory.loadContainer(
          hub: HubApi(),
          configuration: modelConfiguration
        ) { progress in
          print("📥 Model download progress: \(progress.fractionCompleted * 100)%")
        }
        
        self.modelContainer = modelContainer
        self.isInitializing = false
        self.isReady = true
        print("✅ MLX LLM model loaded successfully")
      } catch {
        print("❌ Failed to load MLX LLM: \(error)")
        self.initializationError = error
        self.isInitializing = false
      }
    }
  }
  
  private func handleMethodCall(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard call.method == "generate" else {
      result(FlutterMethodNotImplemented)
      return
    }
    
    guard let args = call.arguments as? [String: Any],
          let prompt = args["prompt"] as? String else {
      result(FlutterError(
        code: "INVALID_ARGS",
        message: "Expected {'prompt': '<string>', 'context': '<string>' (optional)}",
        details: nil
      ))
      return
    }
    
    let context = args["context"] as? String ?? ""
    
    // Check for initialization error
    if let error = initializationError {
      result(FlutterError(
        code: "INIT_ERROR",
        message: "Failed to initialize LLM: \(error.localizedDescription)",
        details: nil
      ))
      return
    }
    
    if !isReady {
      result(FlutterError(
        code: "NOT_READY",
        message: "LLM is still loading. Please wait a moment and try again.",
        details: nil
      ))
      return
    }
    
    // Generate response asynchronously
    DispatchQueue.global(qos: .userInitiated).async { [weak self] in
      guard let self else { return }
      
      let response: String
      if self.USE_REAL_LLM && self.modelContainer != nil {
        response = self.generateRealResponse(prompt: prompt, context: context)
      } else {
        response = self.generateMockResponse(prompt: prompt, context: context)
      }
      
      DispatchQueue.main.async {
        result(response)
      }
    }
  }
  
  /// Real LLM generation using MLX Swift
  private func generateRealResponse(prompt: String, context: String) -> String {
    guard let container = modelContainer else {
      return "Error: Model container not initialized"
    }
    
    // Build full prompt with context
    let fullPrompt: String
    if !context.isEmpty {
      fullPrompt = """
      \(systemPrompt)
      
      CONTEXT:
      \(context)
      
      USER:
      \(prompt)
      """
    } else {
      fullPrompt = """
      \(systemPrompt)
      
      USER:
      \(prompt)
      """
    }
    
    // Generate response using async/await
    var response = ""
    let semaphore = DispatchSemaphore(value: 0)
    
    Task {
      do {
        // Seed for randomness
        MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))
        
        // Generation parameters
        let parameters = GenerateParameters(
          maxTokens: 500,
          temperature: 0.2,
          topP: 0.9
        )
        
        // Create user input
        let userInput = UserInput(prompt: fullPrompt)
        
        // Generate
        try await container.perform { (modelContext: ModelContext) in
          let input = try await modelContext.processor.prepare(input: userInput)
          let stream = try MLXLMCommon.generate(
            input: input,
            parameters: parameters,
            context: modelContext
          )
          
          for try await generation in stream {
            if let chunk = generation.chunk {
              response += chunk
            }
          }
        }
        
        semaphore.signal()
      } catch {
        response = "Error generating response: \(error.localizedDescription)"
        semaphore.signal()
      }
    }
    
    // Wait for generation to complete
    semaphore.wait()
    return response
  }
  
  /// Mock response generator - demonstrates the expected format
  /// Replace this with actual MLC-LLM inference once integrated
  private func generateMockResponse(prompt: String, context: String) -> String {
    // Simulate processing time
    Thread.sleep(forTimeInterval: 2.0)
    
    // Parse context to extract sources
    let sources = extractSources(from: context)
    let sourceRefs = sources.isEmpty ? "" : "\n\nSources: " + sources.joined(separator: ", ")
    
    return """
**Severe Bleeding Control**

RED FLAGS:
• Spurting or pulsating blood
• Blood soaking through multiple bandages
• Signs of shock (pale, cold, rapid pulse)
• Bleeding that won't stop after 10 minutes of pressure

DO NOW — Step-by-step:
1. [D1] Ensure scene safety - wear gloves if available
2. [D1] Apply firm, direct pressure with clean cloth/bandage
3. [D1] If blood soaks through, add layers - DO NOT remove first dressing
4. [D2] Elevate the wound above heart level if possible
5. [D3] Use tourniquet ONLY for life-threatening limb bleeding when pressure fails

When to Escalate:
• Call 911 immediately for severe bleeding
• Bleeding doesn't slow after 10 min of pressure
• Any signs of shock appear
• Wound is very large or deep

What to Avoid:
• Don't remove the first bandage once applied
• Don't use a tourniquet unless absolutely necessary
• Don't probe or clean wound before bleeding is controlled\(sourceRefs)

Confidence: high

---
⚠️ This is a DEMO response. Integrate MLC-LLM for real inference.
See MLC_INTEGRATION_GUIDE.md for setup instructions.
"""
  }
  
  private func extractSources(from context: String) -> [String] {
    var sources: [String] = []
    let lines = context.components(separatedBy: "\n")
    for line in lines {
      if line.hasPrefix("[D") && line.contains("]") {
        if let endIndex = line.firstIndex(of: "]") {
          let source = String(line[..<endIndex]) + "]"
          sources.append(source)
        }
      }
    }
    return sources
  }
}

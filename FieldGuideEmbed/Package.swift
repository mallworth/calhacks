// swift-tools-version:5.9
import PackageDescription

let package = Package(
  name: "FieldGuideEmbed",
  platforms: [.macOS(.v13), .iOS(.v17)],
  products: [
    .executable(name: "Embed", targets: ["Embed"]),
  ],
  dependencies: [
    .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager.git", from: "1.20.0"),
    .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.1.0")
  ],
  targets: [
    .executableTarget(
      name: "Embed",
      dependencies: [
        // Product is named 'onnxruntime'; it exposes the 'OnnxRuntimeBindings' module.
        .product(name: "onnxruntime",  package: "onnxruntime-swift-package-manager"),
        // Product is named 'Transformers' (umbrella for AutoTokenizer, etc.).
        .product(name: "Transformers", package: "swift-transformers"),
      ],
      // Declare the model files as resources (removes the warning).
      resources: [
        .copy("Models/model.onnx"),
        .copy("Models/tokenizer.json")
      ]
    )
  ]
)

cat > Package.swift <<'SWIFT'
// swift-tools-version:5.9
import PackageDescription

let package = Package(
  name: "Embed",
  platforms: [.macOS(.v13), .iOS(.v16)],
  products: [ .executable(name: "Embed", targets: ["Embed"]) ],
  dependencies: [
    .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager.git", from: "1.18.0"),
    .package(url: "https://github.com/huggingface/swift-tokenizers.git", from: "0.14.0")
  ],
  targets: [
    .executableTarget(
      name: "Embed",
      dependencies: [
        .product(name: "OnnxRuntime", package: "onnxruntime-swift-package-manager"),
        .product(name: "Tokenizers",   package: "swift-tokenizers")
      ],
      resources: [
        .copy("Models/model.onnx"),
        .copy("Models/tokenizer.json")
      ]
    )
  ]
)
SWIFT


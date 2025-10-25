// swift-tools-version:5.9
import PackageDescription

let package = Package(
  name: "FieldGuideEmbed",
  platforms: [.macOS(.v13), .iOS(.v17)],
  products: [
    .executable(name: "Embed", targets: ["Embed"]),
    .library(name: "FieldGuideEmbed", targets: ["FieldGuideEmbed"]),
  ],
  dependencies: [
    .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager.git", from: "1.20.0"),
    .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.1.0")
  ],
  targets: [
    .target(
      name: "FieldGuideEmbed",
      dependencies: [
        .product(name: "onnxruntime",  package: "onnxruntime-swift-package-manager"),
        .product(name: "Tokenizers", package: "swift-transformers"),
      ]
    ),
    .executableTarget(
      name: "Embed",
      dependencies: [
        "FieldGuideEmbed",
        .product(name: "onnxruntime",  package: "onnxruntime-swift-package-manager"),
        .product(name: "Tokenizers", package: "swift-transformers"),
      ],
      resources: [
        .copy("Models/model.onnx"),
        .copy("Models/tokenizer.json")
      ]
    )
  ]
)

# iOS Integration Guide for FieldGuideEmbed

## What's Ready

✅ **FieldGuideEmbed Library** - Swift library with public API
  - `Embedder` class for text embedding
  - `RAGDatabase` class for semantic search
  - Built and tested on macOS

✅ **RAG Database** - Pre-built knowledge base
  - Location: `/Users/gavinlynch04/Desktop/calhacks/rag_database.db`
  - 534 chunks from 8 medical/survival documents
  - Ready to bundle with iOS app

✅ **Model Files**
  - ONNX model: `/Users/gavinlynch04/Desktop/calhacks/onnx-out/model.onnx`
  - Tokenizer: `/Users/gavinlynch04/Desktop/calhacks/onnx-out/tokenizer.json`
  - Config files in same directory

## Next Steps for iOS Integration

### 1. Add Library to iOS Project

#### Option A: Local Package (Recommended for Development)

In Xcode:
1. File → Add Package Dependencies
2. Click "Add Local..."
3. Select `/Users/gavinlynch04/Desktop/calhacks/FieldGuideEmbed`
4. Add to your iOS app target

#### Option B: Copy Source Files

Copy these files to your iOS project:
- `Sources/FieldGuideEmbed/Embedder.swift`
- `Sources/FieldGuideEmbed/RAGDatabase.swift`
- `Sources/FieldGuideEmbed/FieldGuideEmbed.swift`

And add dependencies in your `Package.swift` or Podfile:
- `onnxruntime-swift-package-manager`
- `swift-transformers`

### 2. Bundle Required Files

Add these files to your iOS app bundle:

1. **In Xcode**, add to your target:
   - `rag_database.db` (24MB)
   - `model.onnx` (132MB)
   - `tokenizer.json` (466KB)
   - `config.json`
   - `vocab.txt`

2. **Build Phase**: Ensure "Copy Bundle Resources" includes these files

3. **Access in code**:
```swift
let modelPath = Bundle.main.path(forResource: "model", ofType: "onnx")!
let dbPath = Bundle.main.path(forResource: "rag_database", ofType: "db")!
```

### 3. Create Native Channel Handler

Create a new Swift file in your iOS project (e.g., `RAGChannelHandler.swift`):

```swift
import Flutter
import FieldGuideEmbed

class RAGChannelHandler: NSObject {
    private var embedder: Embedder?
    private var ragDB: RAGDatabase?
    private let channel: FlutterMethodChannel
    
    init(binaryMessenger: FlutterBinaryMessenger) {
        self.channel = FlutterMethodChannel(
            name: "com.fieldguide/rag",
            binaryMessenger: binaryMessenger
        )
        super.init()
        
        channel.setMethodCallHandler { [weak self] (call, result) in
            self?.handleMethodCall(call, result: result)
        }
        
        // Initialize asynchronously
        Task {
            await self.initialize()
        }
    }
    
    private func initialize() async {
        do {
            let modelPath = Bundle.main.path(forResource: "model", ofType: "onnx")!
            let tokenizerDir = Bundle.main.resourcePath!
            let dbPath = Bundle.main.path(forResource: "rag_database", ofType: "db")!
            
            embedder = try await Embedder(
                modelPath: modelPath,
                tokenizerDir: tokenizerDir
            )
            ragDB = try RAGDatabase(
                databasePath: dbPath,
                embedder: embedder!
            )
            
            print("✓ RAG system initialized")
        } catch {
            print("❌ Failed to initialize RAG: \(error)")
        }
    }
    
    private func handleMethodCall(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "search":
            searchRAG(call, result: result)
        case "getChunkCount":
            getChunkCount(result: result)
        default:
            result(FlutterMethodNotImplemented)
        }
    }
    
    private func searchRAG(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let ragDB = ragDB else {
            result(FlutterError(
                code: "NOT_INITIALIZED",
                message: "RAG database not initialized",
                details: nil
            ))
            return
        }
        
        guard let args = call.arguments as? [String: Any],
              let query = args["query"] as? String else {
            result(FlutterError(
                code: "INVALID_ARGUMENT",
                message: "Missing 'query' parameter",
                details: nil
            ))
            return
        }
        
        let topK = args["topK"] as? Int ?? 5
        
        do {
            let results = try ragDB.search(query: query, topK: topK)
            let jsonResults = results.map { r in
                [
                    "source": r.source,
                    "chunk_id": r.chunkId,
                    "text": r.text,
                    "distance": r.distance
                ] as [String: Any]
            }
            result(jsonResults)
        } catch {
            result(FlutterError(
                code: "SEARCH_ERROR",
                message: error.localizedDescription,
                details: nil
            ))
        }
    }
    
    private func getChunkCount(result: @escaping FlutterResult) {
        guard let ragDB = ragDB else {
            result(FlutterError(
                code: "NOT_INITIALIZED",
                message: "RAG database not initialized",
                details: nil
            ))
            return
        }
        
        do {
            let count = try ragDB.getChunkCount()
            result(count)
        } catch {
            result(FlutterError(
                code: "COUNT_ERROR",
                message: error.localizedDescription,
                details: nil
            ))
        }
    }
}
```

### 4. Register Handler in AppDelegate

Update your `AppDelegate.swift`:

```swift
import UIKit
import Flutter

@main
@objc class AppDelegate: FlutterAppDelegate {
    private var ragHandler: RAGChannelHandler?
    
    override func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
        GeneratedPluginRegistrant.register(with: self)
        
        if let controller = window?.rootViewController as? FlutterViewController {
            ragHandler = RAGChannelHandler(binaryMessenger: controller.binaryMessenger)
        }
        
        return super.application(application, didFinishLaunchingWithOptions: launchOptions)
    }
}
```

### 5. Update Flutter Code

Update `app/lib/services/native_channels.dart`:

```dart
import 'package:flutter/services.dart';

class RAGService {
  static const _channel = MethodChannel('com.fieldguide/rag');
  
  /// Search the knowledge base for relevant information
  /// 
  /// Returns a list of search results with source, text, and distance
  static Future<List<RAGResult>> search(String query, {int topK = 5}) async {
    try {
      final results = await _channel.invokeMethod('search', {
        'query': query,
        'topK': topK,
      });
      
      return (results as List).map((r) => RAGResult.fromJson(r)).toList();
    } on PlatformException catch (e) {
      print('Error searching RAG: ${e.message}');
      rethrow;
    }
  }
  
  /// Get total number of chunks in the database
  static Future<int> getChunkCount() async {
    try {
      return await _channel.invokeMethod('getChunkCount');
    } on PlatformException catch (e) {
      print('Error getting chunk count: ${e.message}');
      return 0;
    }
  }
}

class RAGResult {
  final String source;
  final int chunkId;
  final String text;
  final double distance;
  
  RAGResult({
    required this.source,
    required this.chunkId,
    required this.text,
    required this.distance,
  });
  
  factory RAGResult.fromJson(Map<dynamic, dynamic> json) {
    return RAGResult(
      source: json['source'] as String,
      chunkId: json['chunk_id'] as int,
      text: json['text'] as String,
      distance: (json['distance'] as num).toDouble(),
    );
  }
}
```

### 6. Usage in Flutter

```dart
import 'package:flutter/material.dart';
import 'services/native_channels.dart';

class SearchScreen extends StatefulWidget {
  @override
  _SearchScreenState createState() => _SearchScreenState();
}

class _SearchScreenState extends State<SearchScreen> {
  String query = '';
  List<RAGResult> results = [];
  bool isLoading = false;
  
  Future<void> performSearch() async {
    if (query.isEmpty) return;
    
    setState(() {
      isLoading = true;
    });
    
    try {
      final searchResults = await RAGService.search(query, topK: 5);
      setState(() {
        results = searchResults;
        isLoading = false;
      });
    } catch (e) {
      print('Search error: $e');
      setState(() {
        isLoading = false;
      });
    }
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Field Guide Search')),
      body: Column(
        children: [
          Padding(
            padding: EdgeInsets.all(16),
            child: TextField(
              decoration: InputDecoration(
                hintText: 'Ask a question...',
                suffixIcon: IconButton(
                  icon: Icon(Icons.search),
                  onPressed: performSearch,
                ),
              ),
              onChanged: (value) => query = value,
              onSubmitted: (_) => performSearch(),
            ),
          ),
          Expanded(
            child: isLoading
                ? Center(child: CircularProgressIndicator())
                : ListView.builder(
                    itemCount: results.length,
                    itemBuilder: (context, index) {
                      final result = results[index];
                      return Card(
                        margin: EdgeInsets.all(8),
                        child: Padding(
                          padding: EdgeInsets.all(12),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                result.source,
                                style: TextStyle(
                                  fontWeight: FontWeight.bold,
                                  color: Colors.blue,
                                ),
                              ),
                              SizedBox(height: 4),
                              Text(
                                'Distance: ${result.distance.toStringAsFixed(4)}',
                                style: TextStyle(
                                  fontSize: 12,
                                  color: Colors.grey,
                                ),
                              ),
                              SizedBox(height: 8),
                              Text(result.text),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}
```

## Testing

### 1. Build iOS App

```bash
cd app
flutter build ios
```

### 2. Test on Simulator

```bash
flutter run
```

### 3. Test Queries

Try these example queries:
- "How do I treat hypothermia?"
- "What should I do for severe bleeding?"
- "How can I purify water in an emergency?"

## Performance Considerations

- **First Search Latency**: ~1-2 seconds (model loading + embedding generation)
- **Subsequent Searches**: ~100-200ms
- **Memory Usage**: ~150-200MB (model + database in memory)
- **Database Size**: 24MB (should be acceptable for app bundle)
- **Model Size**: 132MB (consider on-demand download for App Store)

## Optimization Tips

1. **Pre-initialize on App Launch**: Initialize embedder during splash screen
2. **Cache Embeddings**: For common queries, cache embeddings
3. **Background Thread**: Run search on background thread to avoid UI blocking
4. **Pagination**: Only show top 5 results initially, load more on demand

## Troubleshooting

### "Cannot find FieldGuideEmbed"
- Ensure package is added to iOS target in Xcode
- Check that `import FieldGuideEmbed` is in your Swift file

### "Model file not found"
- Verify files are in "Copy Bundle Resources" build phase
- Check file names match exactly (case-sensitive)

### "RAG database not initialized"
- Check that files are bundled correctly
- Look for initialization errors in Xcode console
- Ensure async initialization completes before first search

### Slow Performance
- Ensure you're building in Release mode for testing
- Check that Metal/CoreML acceleration is enabled in ONNX Runtime
- Consider reducing `topK` parameter

## File Sizes

Approximate sizes to bundle with app:
- `rag_database.db`: 24 MB
- `model.onnx`: 132 MB
- `tokenizer.json`: 466 KB
- `vocab.txt`: 232 KB
- `config.json`: 1 KB

**Total**: ~157 MB

For App Store submission, consider:
- Downloading model on first launch (reduces initial download)
- Using app thinning
- Compressing database (SQLite VACUUM)

## Complete!

You now have:
- ✅ Swift library ready for iOS
- ✅ Integration guide for Flutter
- ✅ Example code for all layers
- ✅ Pre-built knowledge base
- ✅ Tested embedding system

Next: Follow steps 1-6 above to integrate into your Flutter app!

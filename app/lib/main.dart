import 'package:flutter/material.dart';
import 'services/native_channels.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) => const MaterialApp(home: DemoPage());
}

class DemoPage extends StatefulWidget {
  const DemoPage({super.key});
  @override
  State<DemoPage> createState() => _DemoPageState();
}

class _DemoPageState extends State<DemoPage> {
  final _controller = TextEditingController();
  String _output = "Waiting for embedder to load...";
  bool _isLoading = false;
  
  // Hardcoded reference text
  static const String referenceText = "Apply direct pressure to stop bleeding. Use a clean cloth or bandage and maintain pressure for several minutes.";
  
  @override
  void initState() {
    super.initState();
    _checkEmbedderReady();
  }
  
  Future<void> _checkEmbedderReady() async {
    // Wait a bit for embedder to initialize
    await Future.delayed(const Duration(seconds: 2));
    try {
      // Try a test embedding
      await NativeChannels.embed("test");
      setState(() => _output = "Ready! Enter your query above.");
    } catch (e) {
      if (e.toString().contains("still loading")) {
        // Still loading, check again
        await Future.delayed(const Duration(seconds: 1));
        _checkEmbedderReady();
      } else {
        setState(() => _output = "Error initializing: $e");
      }
    }
  }

  Future<void> _calculateSimilarity() async {
    final query = _controller.text.trim();
    if (query.isEmpty) {
      setState(() => _output = "Please enter a query");
      return;
    }
    
    setState(() {
      _isLoading = true;
      _output = "Computing embeddings...";
    });
    
    try {
      // Get embeddings for both texts
      final refEmbed = await NativeChannels.embed(referenceText);
      final queryEmbed = await NativeChannels.embed(query);
      
      // Calculate cosine similarity (vectors are already normalized)
      double similarity = 0.0;
      for (int i = 0; i < refEmbed.length; i++) {
        similarity += refEmbed[i] * queryEmbed[i];
      }
      
      final percentage = (similarity * 100).toStringAsFixed(1);
      
      setState(() {
        _output = "Similarity: $percentage%";
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _output = "Error: $e";
        _isLoading = false;
      });
    }
  }

  Future<void> _testLLM() async {
    final res = await NativeChannels.generate(_controller.text);
    setState(() => _output = res);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Embedding Similarity Test")),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Reference text display
            const Text(
              "Reference Text:",
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.blue.shade50,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.blue.shade200),
              ),
              child: const Text(
                referenceText,
                style: TextStyle(fontSize: 14),
              ),
            ),
            const SizedBox(height: 24),
            
            // Query input
            const Text(
              "Your Query:",
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: _controller,
              maxLines: 3,
              decoration: InputDecoration(
                hintText: "Enter your query here...",
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
            ),
            const SizedBox(height: 16),
            
            // Submit button
            ElevatedButton(
              onPressed: _isLoading ? null : _calculateSimilarity,
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
              ),
              child: _isLoading
                  ? const SizedBox(
                      height: 20,
                      width: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Text("Calculate Similarity", style: TextStyle(fontSize: 16)),
            ),
            const SizedBox(height: 24),
            
            // Output
            if (_output.isNotEmpty)
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.green.shade50,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.green.shade200),
                ),
                child: Text(
                  _output,
                  style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                  textAlign: TextAlign.center,
                ),
              ),
          ],
        ),
      ),
    );
  }
}

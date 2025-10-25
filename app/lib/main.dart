import 'dart:math';

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
  List<Map<String, dynamic>> _ragResults = [];
  List<bool> _expanded = [];
  
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
      // _output = "Computing embeddings...";
      _ragResults = [];
      _expanded = [];
    });

    try {
      // Get embeddings for both texts
      final refEmbed = await NativeChannels.embed(referenceText);
      final queryEmbed = await NativeChannels.embed(query);

      final similarity = _cosineSimilarity(refEmbed, queryEmbed);

      final percentage = (similarity * 100).toStringAsFixed(1);

      final ragResults = await NativeChannels.ragSearch(queryEmbed, topK: 3);

      setState(() {
        // _output = "Similarity: $percentage%";
        _ragResults = ragResults;
        _expanded = List<bool>.filled(ragResults.length, false);
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _output = "Error: $e";
        _ragResults = [];
        _expanded = [];
        _isLoading = false;
      });
    }
  }

  Future<void> _testLLM() async {
    final res = await NativeChannels.generate(_controller.text);
    setState(() => _output = res);
  }

  double _cosineSimilarity(List<double> a, List<double> b) {
    if (a.length != b.length || a.isEmpty) {
      return 0.0;
    }

    double dot = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    for (var i = 0; i < a.length; i++) {
      final av = a[i];
      final bv = b[i];
      dot += av * bv;
      normA += av * av;
      normB += bv * bv;
    }

    final denom = normA > 0 && normB > 0 ? sqrt(normA) * sqrt(normB) : 0.0;
    if (denom == 0.0) {
      return 0.0;
    }
    return (dot / denom).clamp(-1.0, 1.0);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Embedding Similarity Test")),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
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
              const SizedBox(height: 16),

              if (_output.isNotEmpty) ...[
                Text(
                  _output,
                  style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 16),
              ],
              
              if (_ragResults.isNotEmpty) ...[
                const SizedBox(height: 8),
                const Text(
                  "Top Knowledge Base Matches:",
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 12),
                for (var i = 0; i < _ragResults.length && i < 3; i++)
                  _buildResultCard(i),
              ],

              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildResultCard(int index) {
    final result = _ragResults[index];
    final expanded = index < _expanded.length ? _expanded[index] : false;
    final score = (result['score'] as num?)?.toDouble() ?? 0.0;
    final scorePercent = (score.clamp(-1.0, 1.0) * 100).toStringAsFixed(1);

    return Card(
      elevation: 1,
      margin: const EdgeInsets.only(bottom: 12),
      child: InkWell(
        onTap: () {
          setState(() {
            if (index < _expanded.length) {
              _expanded[index] = !_expanded[index];
            }
          });
        },
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Expanded(
                    child: Text(
                      result['source']?.toString() ?? 'Unknown source',
                      style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                    ),
                  ),
                  Text(
                    "$scorePercent%",
                    style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w500),
                  ),
                  const SizedBox(width: 8),
                  Icon(expanded ? Icons.expand_less : Icons.expand_more, size: 20),
                ],
              ),
              if (expanded) ...[
                const SizedBox(height: 12),
                SizedBox(
                  height: 150,
                  child: Scrollbar(
                    child: SingleChildScrollView(
                      child: Text(
                        (result['text']?.toString() ?? '').trim(),
                        style: const TextStyle(fontSize: 14),
                      ),
                    ),
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

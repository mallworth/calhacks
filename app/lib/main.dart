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
  String _llmResponse = "";
  bool _isGenerating = false;
  // LLM status
  String _llmState = 'idle';
  double _llmProgress = 0.0;
  bool _llmReady = false;
  bool _pollingStatus = false;
  String _downloadError = "";
  bool _modelCached = false;
  String _cacheInfo = "";
  
  @override
  void initState() {
    super.initState();
    _checkModelCache();
    _startPollingLLMStatus();
    setState(() => _output = "Ready! Enter your query above.");
  }
  
  Future<void> _checkModelCache() async {
    try {
      final cacheStatus = await NativeChannels.checkModelCache();
      setState(() {
        _modelCached = (cacheStatus['cached'] as bool?) ?? false;
        if (_modelCached) {
          final files = cacheStatus['files'] as List? ?? [];
          _cacheInfo = "Model cached (${files.length} files)";
          print("✅ Model is cached with ${files.length} files");
        } else {
          _cacheInfo = "Model not cached";
        }
      });
    } catch (e) {
      print("Error checking cache: $e");
    }
  }

  void _startPollingLLMStatus() {
    if (_pollingStatus) return;
    _pollingStatus = true;
    () async {
      while (mounted) {
        try {
          final st = await NativeChannels.llmStatus();
          setState(() {
            _llmState = (st['state'] ?? 'idle').toString();
            _llmProgress = (st['progress'] as num?)?.toDouble() ?? 0.0;
            _llmReady = (st['ready'] as bool?) ?? false;
          });
        } catch (_) {}
        await Future.delayed(const Duration(milliseconds: 500));
      }
    }();
  }

  Future<void> _startModelDownload() async {
    print("🔵 Download button pressed");
    setState(() {
      _downloadError = "";
    });
    try {
      print("🔵 Calling NativeChannels.downloadModel()");
      await NativeChannels.downloadModel();
      print("🔵 downloadModel() completed successfully");
    } catch (e) {
      print("🔴 downloadModel() error: $e");
      setState(() {
        _downloadError = e.toString();
      });
    }
  }
  
  Future<void> _clearCacheAndRetry() async {
    print("🧹 Clear cache and retry");
    setState(() {
      _downloadError = "";
      _llmState = 'idle';
    });
    try {
      await NativeChannels.clearModelCache();
      print("✅ Cache cleared, starting download");
      await Future.delayed(const Duration(milliseconds: 500));
      await _startModelDownload();
    } catch (e) {
      print("🔴 clearCache error: $e");
      setState(() {
        _downloadError = "Failed to clear cache: ${e.toString()}";
      });
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _calculateSimilarity() async {
    final query = _controller.text.trim();
    if (query.isEmpty) {
      setState(() => _output = "Please enter a query");
      return;
    }
    
    setState(() {
      _isLoading = true;
      _isGenerating = false;
      _ragResults = [];
      _expanded = [];
      _llmResponse = "";
    });

    try {
      // Step 1: Get embedding for the query
      final queryEmbed = await NativeChannels.embed(query);

      // Step 2: Search RAG database for relevant context
      final ragResults = await NativeChannels.ragSearch(queryEmbed, topK: 3);

      setState(() {
        _ragResults = ragResults;
        _expanded = List<bool>.filled(ragResults.length, false);
        _isLoading = false;
        _isGenerating = true;
      });

      // Step 3: Format context from RAG results
      final context = _formatContext(ragResults);

      // Step 4: Generate LLM response with context
      final llmResponse = await NativeChannels.generate(query, context: context);

      setState(() {
        _llmResponse = llmResponse;
        _isGenerating = false;
      });
    } catch (e) {
      setState(() {
        _output = "Error: $e";
        _ragResults = [];
        _expanded = [];
        _llmResponse = "";
        _isLoading = false;
        _isGenerating = false;
      });
    }
  }

  String _formatContext(List<Map<String, dynamic>> results) {
    if (results.isEmpty) return "";
    
    final buffer = StringBuffer();
    for (var i = 0; i < results.length; i++) {
      final result = results[i];
      final source = result['source'] ?? 'Unknown';
      final text = result['text'] ?? '';
      buffer.writeln('[D${i + 1}] $source');
      buffer.writeln(text.trim());
      buffer.writeln();
    }
    return buffer.toString().trim();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("FieldGuide RAG Assistant")),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Model download section
              if (!_llmReady && _llmState != 'loading') ...[
                Card(
                  elevation: 2,
                  color: Colors.orange.shade50,
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          "⚠️ Model Not Downloaded",
                          style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          _modelCached 
                            ? "Model files found in cache. Click below to load the model."
                            : "Download the AI model (~600MB) to enable intelligent responses. This only needs to be done once.",
                          style: const TextStyle(fontSize: 14),
                        ),
                        if (_modelCached) ...[
                          const SizedBox(height: 4),
                          Text(
                            _cacheInfo,
                            style: TextStyle(fontSize: 12, color: Colors.grey.shade700),
                          ),
                        ],
                        const SizedBox(height: 12),
                        ElevatedButton.icon(
                          onPressed: (_llmState == 'loading' || _llmState == 'copying') ? null : _startModelDownload,
                          icon: Icon(_modelCached ? Icons.play_arrow : Icons.download),
                          label: Text(_modelCached ? "Load Cached Model" : "Download Model Now"),
                          style: ElevatedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(vertical: 12),
                          ),
                        ),
                        if (_downloadError.isNotEmpty) ...[
                          const SizedBox(height: 8),
                          Text(
                            "Error: $_downloadError",
                            style: const TextStyle(color: Colors.red, fontSize: 12),
                          ),
                          const SizedBox(height: 8),
                          ElevatedButton.icon(
                            onPressed: _clearCacheAndRetry,
                            icon: const Icon(Icons.delete_forever),
                            label: const Text("Clear Cache & Retry"),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.orange,
                              foregroundColor: Colors.white,
                            ),
                          ),
                        ],
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 16),
              ],
              
              // Error state with clear cache option
              if (_llmState == 'error' && !_llmReady) ...[
                Card(
                  elevation: 2,
                  color: Colors.red.shade50,
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          "❌ Model Download Failed",
                          style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.red),
                        ),
                        const SizedBox(height: 8),
                        const Text(
                          "The model download encountered an error. You can try clearing the cache and downloading again.",
                          style: TextStyle(fontSize: 14),
                        ),
                        const SizedBox(height: 12),
                        ElevatedButton.icon(
                          onPressed: _clearCacheAndRetry,
                          icon: const Icon(Icons.refresh),
                          label: const Text("Clear Cache & Retry"),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.orange,
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(vertical: 12),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 16),
              ],
              
              // Download/loading progress
              if (_llmState == 'loading') ...[
                Card(
                  elevation: 2,
                  color: Colors.blue.shade50,
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            const Expanded(
                              child: Text(
                                'Downloading model from Hugging Face…',
                                style: TextStyle(fontWeight: FontWeight.w600),
                              ),
                            ),
                            Text('${(_llmProgress * 100).toStringAsFixed(0)}%'),
                          ],
                        ),
                        const SizedBox(height: 8),
                        LinearProgressIndicator(value: _llmProgress > 0 ? _llmProgress.clamp(0.0, 1.0) : null),
                        const SizedBox(height: 8),
                        const Text(
                          "This may take several minutes depending on your connection.",
                          style: TextStyle(fontSize: 12, color: Colors.black54),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 16),
              ],

              if (_llmReady) ...[
                Card(
                  elevation: 1,
                  color: Colors.green.shade50,
                  child: const Padding(
                    padding: EdgeInsets.all(12),
                    child: Row(
                      children: [
                        Icon(Icons.check_circle, color: Colors.green, size: 20),
                        SizedBox(width: 8),
                        Text(
                          "Model ready!",
                          style: TextStyle(fontWeight: FontWeight.w600),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 16),
              ],

              // Query input
              const Text(
                "Your Emergency/Survival Query:",
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              TextField(
                controller: _controller,
                maxLines: 3,
                decoration: InputDecoration(
                  hintText: "e.g., How do I stop severe bleeding?",
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              
              // Submit button
              ElevatedButton(
                onPressed: (_isLoading || _isGenerating || !_llmReady) ? null : _calculateSimilarity,
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                ),
                child: _isLoading
                    ? const Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          SizedBox(
                            height: 20,
                            width: 20,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          ),
                          SizedBox(width: 8),
                          Text("Searching knowledge base..."),
                        ],
                      )
                    : _isGenerating
                        ? const Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              SizedBox(
                                height: 20,
                                width: 20,
                                child: CircularProgressIndicator(strokeWidth: 2),
                              ),
                              SizedBox(width: 8),
                              Text("Generating response..."),
                            ],
                          )
                        : const Text("Get Answer", style: TextStyle(fontSize: 16)),
              ),
              const SizedBox(height: 16),

              if (_output.isNotEmpty && !_isLoading && !_isGenerating && _llmResponse.isEmpty) ...[
                Text(
                  _output,
                  style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 16),
              ],
              
              // LLM Response Section
              if (_llmResponse.isNotEmpty) ...[
                const Text(
                  "FieldGuide Response:",
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 8),
                Card(
                  elevation: 2,
                  color: Colors.blue.shade50,
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Text(
                      _llmResponse,
                      style: const TextStyle(fontSize: 15, height: 1.5),
                    ),
                  ),
                ),
                const SizedBox(height: 24),
              ],
              
              if (_ragResults.isNotEmpty) ...[
                const Divider(),
                const SizedBox(height: 8),
                const Text(
                  "Source Documents (tap to expand):",
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

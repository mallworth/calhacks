import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'services/native_channels.dart';

const BorderRadius _radius12 = BorderRadius.all(Radius.circular(12));

const Color _olivePrimary = Color(0xFF87986A);
const Color _oliveAccent = Color(0xFF6E7C4A);
const Color _oliveMuted = Color(0xFFB5C99A);
const Color _oliveSurface = Color(0xFFE2E8C0);
const Color _ivoryBackground = Color(0xFFF6F1D3);
const Color _ivorySurface = Color(0xFFFBF7E6);
const Color _successTint = Color(0xFFDDE7C7);
const Color _warningTint = Color(0xFFE9DEBB);
const Color _errorTint = Color(0xFFE7C9BA);
const Color _successAccent = Color(0xFF4F6B34);
const Color _warningAccent = Color(0xFF9B7A1D);
const Color _errorAccent = Color(0xFF8C3B2A);
const Color _textPrimary = Color(0xFF2F3523);
const Color _textSecondary = Color(0xFF5C6144);

/// Map source file names from the RAG service to the display titles used in
/// the "Source Documents" cards. Update the placeholder values (string1,
/// string2, etc.) to control the text shown on each card header.
const Map<String, String> _sourceDisplayNames = {
  '2018-First-Aid-Pocket-Guide_1.txt': 'Canadian Red Cross First Aid Pocket Guider',
  'cold-weather-survival.txt': 'Cold Weather Survival',
  'hewett-brumberg-et-al-2024-2024-american-heart-association-and-american-red-cross-guidelines-for-first-aid.txt': 'Red Cross Guidelines for First Aid',
  'How to Make Water Safe in an Emergency _ Water, Sanitation, and Hygiene (WASH)-related Emergencies and Outbreaks _ CDC.txt': 'CDC Water Safety Guide',
  'until-help-arrives-web-tutorial.txt': 'American Red Cross Until Help Arrives',
  'USMC-Summer-Survival-Course-Handbook.txt': 'USMC Summer Survival Manual',
  'USMC-Winter-Survival-Course-Handbook.txt': 'USMC Winter Survival Manual',
  'WHO-ICRC-Basic-Emergency-Care.txt': 'WHO Basic Emergency Care',
};

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    final colorScheme = ColorScheme.fromSeed(
      seedColor: _olivePrimary,
      brightness: Brightness.light,
    ).copyWith(
      primary: _olivePrimary,
      onPrimary: _ivorySurface,
      secondary: _oliveMuted,
      onSecondary: _textPrimary,
      background: _ivoryBackground,
      onBackground: _textPrimary,
      surface: _ivorySurface,
      onSurface: _textPrimary,
      error: const Color(0xFFB8614B),
      outline: _oliveMuted,
    );

    final theme = ThemeData(
      colorScheme: colorScheme,
      scaffoldBackgroundColor: _ivoryBackground,
      appBarTheme: const AppBarTheme(
        backgroundColor: _olivePrimary,
        foregroundColor: _ivorySurface,
        elevation: 0,
        centerTitle: true,
        titleTextStyle: TextStyle(
          fontSize: 20,
          fontWeight: FontWeight.w600,
        ),
      ),
      cardTheme: const CardThemeData(
        color: _ivorySurface,
        elevation: 1,
        margin: EdgeInsets.zero,
        shape: RoundedRectangleBorder(borderRadius: _radius12),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: _olivePrimary,
          foregroundColor: _ivorySurface,
          shape: const RoundedRectangleBorder(borderRadius: _radius12),
          textStyle: const TextStyle(fontWeight: FontWeight.w600),
          padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 20),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: _olivePrimary,
          side: const BorderSide(color: _olivePrimary, width: 1.5),
          shape: const RoundedRectangleBorder(borderRadius: _radius12),
          padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 20),
        ),
      ),
      inputDecorationTheme: const InputDecorationTheme(
        filled: true,
        fillColor: _ivorySurface,
        hintStyle: TextStyle(color: _textSecondary),
        contentPadding: EdgeInsets.all(16),
        border: OutlineInputBorder(
          borderRadius: _radius12,
          borderSide: BorderSide(color: _oliveMuted),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: _radius12,
          borderSide: BorderSide(color: _oliveMuted),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: _radius12,
          borderSide: BorderSide(color: _oliveAccent, width: 1.5),
        ),
      ),
      progressIndicatorTheme: const ProgressIndicatorThemeData(
        color: _oliveAccent,
      ),
      dividerTheme: const DividerThemeData(
        color: _oliveSurface,
        thickness: 1,
      ),
      textTheme: ThemeData.light().textTheme.apply(
            bodyColor: _textPrimary,
            displayColor: _textPrimary,
          ),
    );

    return MaterialApp(
      theme: theme,
      home: const DemoPage(),
    );
  }
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
  bool _showLowRelevanceDisclaimer = false;
  double _maxSimilarity = 0.0;
  bool _hideModelUi = false;
  
  @override
  void initState() {
    super.initState();
    _checkModelCache();
    _startPollingLLMStatus();
    setState(() => _output = "");
  }
  
  Future<void> _checkModelCache() async {
    try {
      print("🔍 Checking model cache from Flutter...");
      final cacheStatus = await NativeChannels.checkModelCache();
      print("📦 Cache status received: $cacheStatus");
      
      final cached = (cacheStatus['cached'] as bool?) ?? false;
      final complete = (cacheStatus['complete'] as bool?) ?? cached;
      final files = cacheStatus['files'] as List? ?? [];
      final path = cacheStatus['path'] as String? ?? '';
      
      setState(() {
        _modelCached = complete; // Only mark as cached if complete
        _hideModelUi = complete;
        if (complete) {
          _cacheInfo = "Model ready (${files.length} files)";
          print("✅ Model is complete with ${files.length} files at: $path");
        } else if (cached && !complete) {
          _cacheInfo = "Model incomplete (${files.length} files) - re-download needed";
          print("⚠️ Model cache exists but incomplete: ${files.length} files at: $path");
        } else {
          _cacheInfo = "Model not downloaded";
          print("📥 Model not cached at: $path");
        }
      });
    } catch (e) {
      print("❌ Error checking cache: $e");
      setState(() {
        _modelCached = false;
        _cacheInfo = "Error checking cache";
        _hideModelUi = false;
      });
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
            if (_llmReady) {
              _hideModelUi = true;
            }
            if (!_hideModelUi && !_modelCached) {
              _startModelDownload();
            }
            // Update error message from status if in error state
            if (_llmState == 'error') {
              final statusMessage = (st['message'] ?? '').toString();
              if (statusMessage.isNotEmpty && statusMessage != _downloadError) {
                _downloadError = statusMessage;
              }
            }
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
      _hideModelUi = false;
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
      _showLowRelevanceDisclaimer = false;
      _maxSimilarity = 0.0;
    });

    try {
      // Step 1: Get embedding for the query
      final queryEmbed = await NativeChannels.embed(query);

      // Step 2: Search RAG database for relevant context
      final ragResults = await NativeChannels.ragSearch(queryEmbed, topK: 3);
      
      // Step 3: Check similarity scores (note: RAGService returns "score" not "similarity")
      double maxSimilarity = 0.0;
      for (final result in ragResults) {
        final similarity = (result['score'] as num?)?.toDouble() ?? 0.0;
        if (similarity > maxSimilarity) {
          maxSimilarity = similarity;
        }
      }
      
      // Similarity threshold: 0.25 (25%)
      const similarityThreshold = 0.25;
      final hasRelevantResults = maxSimilarity >= similarityThreshold;
      
      // Filter results to only include those above threshold
      final filteredResults = ragResults.where((result) {
        final similarity = (result['score'] as num?)?.toDouble() ?? 0.0;
        return similarity >= similarityThreshold;
      }).toList();

      setState(() {
        _ragResults = filteredResults;  // Only show relevant results
        _expanded = List<bool>.filled(filteredResults.length, false);
        _isLoading = false;
        _isGenerating = true;
        _maxSimilarity = maxSimilarity;
        _showLowRelevanceDisclaimer = !hasRelevantResults;
      });

      // Step 4: Format context from RAG results (only if relevant)
      final context = hasRelevantResults ? _formatContext(filteredResults) : "";
      
      // Log similarity for debugging
      print("📊 Max similarity: ${(maxSimilarity * 100).toStringAsFixed(1)}% (threshold: ${(similarityThreshold * 100)}%)");
      if (!hasRelevantResults) {
        print("⚠️ Low relevance - not passing context to LLM");
      }

      // Step 5: Generate LLM response (with or without context)
      final llmResponse = await NativeChannels.generate(query, context: context);

      // Filter out <think> tags from the response
      final filteredResponse = _removeThinkTags(llmResponse);

      setState(() {
        _llmResponse = filteredResponse;
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
        _showLowRelevanceDisclaimer = false;
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

  String _removeThinkTags(String text) {
    // Remove everything between <think> and </think> tags (including the tags)
    // This handles multiple think blocks and multiline content
    return text
        .replaceAll(RegExp(r'<think>.*?</think>', dotAll: true), '')
        .trim();
  }

  String _formatSnippetText(String? raw) {
    if (raw == null) return '';
    var text = raw.trim();
    if (text.isEmpty) return text;

    final splitSentences = text
        .split(RegExp(r'(?<=[.!?])\s+(?=[A-Z])'))
        .map((segment) => segment.trim())
        .where((segment) => segment.isNotEmpty)
        .toList();

    if (splitSentences.isEmpty) {
      return text;
    }

    final sentences = List<String>.from(splitSentences);
    if (sentences.length > 1 && !_looksLikeSentenceStart(sentences.first)) {
      sentences.removeAt(0);
    }

    if (sentences.isEmpty) {
      return text;
    }

    var result = sentences.join(' ');

    final matches = RegExp(r'[.!?]').allMatches(result).toList();
    if (matches.isNotEmpty) {
      final last = matches.last;
      if (last.end < result.length) {
        result = result.substring(0, last.end);
      }
    }

    return result.trim();
  }

  bool _looksLikeSentenceStart(String sentence) {
    if (sentence.isEmpty) return false;
    final first = sentence.characters.first;
    return RegExp(r'[A-Z0-9"\'"""()]').hasMatch(first);
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      // Dismiss keyboard when tapping outside
      onTap: () {
        FocusScope.of(context).unfocus();
      },
      child: Scaffold(
        appBar: AppBar(title: const Text("LifeLine")),
        body: Padding(
          padding: const EdgeInsets.all(16),
          child: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
              // Model download section
              if (!_hideModelUi && !_llmReady && _llmState != 'loading') ...[
                Card(
                  elevation: 2,
                  color: _warningTint,
                  shape: const RoundedRectangleBorder(borderRadius: _radius12),
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          "⚠️ Model Not Downloaded",
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: _warningAccent,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          _modelCached 
                            ? "Model files found in cache. Click below to load the model."
                            : "Download the AI model (~600MB) to enable intelligent responses. This only needs to be done once.",
                          style: const TextStyle(fontSize: 14, color: _textPrimary),
                        ),
                        if (_modelCached) ...[
                          const SizedBox(height: 4),
                          Text(
                            _cacheInfo,
                            style: const TextStyle(fontSize: 12, color: _textSecondary),
                          ),
                        ],
                        const SizedBox(height: 12),
                        ElevatedButton.icon(
                          onPressed: (_llmState == 'loading' || _llmState == 'copying' || _hideModelUi)
                              ? null
                              : _startModelDownload,
                          icon: Icon(_modelCached ? Icons.play_arrow : Icons.download),
                          label: Text(_modelCached ? "Load Cached Model" : "Download Model Now"),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: _oliveAccent,
                            foregroundColor: _ivorySurface,
                            padding: const EdgeInsets.symmetric(vertical: 12),
                            shape: const RoundedRectangleBorder(borderRadius: _radius12),
                          ),
                        ),
                        if (_downloadError.isNotEmpty) ...[
                          const SizedBox(height: 8),
                          Text(
                            "Error: $_downloadError",
                            style: const TextStyle(color: _errorAccent, fontSize: 12),
                          ),
                          const SizedBox(height: 8),
                          ElevatedButton.icon(
                            onPressed: _clearCacheAndRetry,
                            icon: const Icon(Icons.delete_forever),
                            label: const Text("Clear Cache & Retry"),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: _errorAccent,
                              foregroundColor: _ivorySurface,
                              shape: const RoundedRectangleBorder(borderRadius: _radius12),
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
              if (!_hideModelUi && _llmState == 'error' && !_llmReady) ...[
                Card(
                  elevation: 2,
                  color: _errorTint,
                  shape: const RoundedRectangleBorder(borderRadius: _radius12),
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          "❌ Model Download Failed",
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: _errorAccent,
                          ),
                        ),
                        const SizedBox(height: 8),
                        if (_downloadError.isNotEmpty) ...[
                          Text(
                            "Error: $_downloadError",
                            style: const TextStyle(fontSize: 13, color: _errorAccent),
                          ),
                          const SizedBox(height: 12),
                        ],
                        const Text(
                          "The model download encountered an error. Try the options below:",
                          style: TextStyle(fontSize: 14, color: _textPrimary),
                        ),
                        const SizedBox(height: 12),
                        ElevatedButton.icon(
                          onPressed: _clearCacheAndRetry,
                          icon: const Icon(Icons.refresh),
                          label: const Text("Clear Cache & Retry"),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: _errorAccent,
                            foregroundColor: _ivorySurface,
                            padding: const EdgeInsets.symmetric(vertical: 12),
                            shape: const RoundedRectangleBorder(borderRadius: _radius12),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 16),
              ],
              
              // Download/loading progress
              if (!_hideModelUi && _llmState == 'loading') ...[
                Card(
                  elevation: 2,
                  color: _oliveSurface,
                  shape: const RoundedRectangleBorder(borderRadius: _radius12),
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
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 16),
              ],

              if (_llmReady) ...[
                Card(
                  elevation: 1,
                  color: _successTint,
                  shape: const RoundedRectangleBorder(borderRadius: _radius12),
                  child: const Padding(
                    padding: EdgeInsets.all(12),
                    child: Row(
                      children: [
                        Icon(Icons.check_circle, color: _successAccent, size: 20),
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
              // const Text(
              //   "Your Emergency/Survival Query:",
              //   style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              // ),
              const SizedBox(height: 8),
              TextField(
                controller: _controller,
                maxLines: 3,
                decoration: const InputDecoration(
                  hintText: "e.g., How do I stop severe bleeding?",
                ),
              ),
              const SizedBox(height: 16),
              
              // Submit button
              ElevatedButton(
                onPressed: (_isLoading || _isGenerating || !_llmReady) 
                  ? null 
                  : () {
                      FocusScope.of(context).unfocus();  // Hide keyboard
                      _calculateSimilarity();
                    },
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
                  "LifeLine Response:",
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 8),
                Card(
                  elevation: 2,
                  color: _oliveSurface,
                  shape: const RoundedRectangleBorder(borderRadius: _radius12),
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: MarkdownBody(
                      data: _llmResponse,
                      styleSheet: MarkdownStyleSheet.fromTheme(Theme.of(context)).copyWith(
                        p: const TextStyle(fontSize: 15, height: 1.5),
                        listBullet: const TextStyle(fontSize: 15, height: 1.5),
                      ),
                    ),
                  ),
                ),
                
                // Low relevance disclaimer
                if (_showLowRelevanceDisclaimer) ...[
                  const SizedBox(height: 12),
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: _errorTint,
                      border: Border.all(color: _errorAccent, width: 1.5),
                      borderRadius: _radius12,
                    ),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Icon(Icons.warning, color: _errorAccent, size: 20),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                "⚠️ Low Relevance Warning",
                                style: TextStyle(
                                  fontSize: 14,
                                  fontWeight: FontWeight.bold,
                                  color: _errorAccent,
                                ),
                              ),
                              const SizedBox(height: 4),
                              Text(
                                "The retrieved information has low similarity to your query (${(_maxSimilarity * 100).toStringAsFixed(0)}% match). This response is NOT based on expert-verified sources and may not be accurate. For medical emergencies, always call 911 or consult a healthcare professional.",
                                style: TextStyle(
                                  fontSize: 12,
                                  color: _errorAccent,
                                  height: 1.4,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
                
                const SizedBox(height: 24),
              ],
              
              if (_ragResults.isNotEmpty) ...[
                const Divider(),
                const SizedBox(height: 8),
                const Text(
                  "Source Documents",
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
      ),  // GestureDetector
    );
  }

  Widget _buildResultCard(int index) {
    final result = _ragResults[index];
    final expanded = index < _expanded.length ? _expanded[index] : false;
    final source = result['source']?.toString() ?? 'Unknown source';
    final displayName = _sourceDisplayNames[source] ?? source;
    final snippet = _formatSnippetText(result['text']?.toString());

    return Card(
      elevation: 1,
      margin: const EdgeInsets.only(bottom: 12),
      color: _ivorySurface,
      shape: const RoundedRectangleBorder(borderRadius: _radius12),
      child: InkWell(
        borderRadius: _radius12,
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
                      displayName,
                      style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                    ),
                  ),
                  Icon(expanded ? Icons.expand_less : Icons.expand_more, size: 20, color: _oliveAccent),
                ],
              ),
              if (expanded) ...[
                const SizedBox(height: 12),
                SizedBox(
                  height: 150,
                  child: Scrollbar(
                    child: SingleChildScrollView(
                      child: Text(
                        snippet,
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

import 'package:flutter/services.dart';

class NativeChannels {
  static const _embed = MethodChannel('survival/embed');
  static const _llm = MethodChannel('survival/llm');
  static const _rag = MethodChannel('survival/rag');

  static Future<List<double>> embed(String text) async {
    try {
      final res = await _embed.invokeMethod<List<dynamic>>('embed', {
        'text': text,
      });
      if (res == null) {
        return [];
      }
      try {
        return res.map((e) => (e as num).toDouble()).toList();
      } catch (e) {
        print('NativeChannels.embed: Error mapping result: $e');
        return [];
      }
    } on PlatformException catch (e) {
      print('NativeChannels.embed: PlatformException: ${e.message}');
      return [];
    } on MissingPluginException catch (e) {
      print('NativeChannels.embed: MissingPluginException: ${e.message}');
      return [];
    } catch (e) {
      print('NativeChannels.embed: Unknown error: $e');
      return [];
    }
  }

  static Future<String> generate(String prompt, {String? context}) async {
    try {
      final args = <String, dynamic>{'prompt': prompt};
      if (context != null && context.isNotEmpty) {
        args['context'] = context;
      }
      final res = await _llm.invokeMethod<String>('generate', args);
      return res ?? "";
    } on PlatformException catch (e) {
      print('PlatformException in generate: ${e.message}');
      rethrow;
    } catch (e) {
      print('Unexpected error in generate: $e');
      rethrow;
    }
  }

  static Future<List<Map<String, dynamic>>> ragSearch(
    List<double> embedding, {
    int topK = 3,
  }) async {
    try {
      final res = await _rag.invokeMethod<List<dynamic>>('search', {
        'embedding': embedding,
        'topK': topK,
      });
      if (res == null) {
        return [];
      }
      return res
          .whereType<Map>()
          .map((row) => row.map((key, value) => MapEntry(key.toString(), value)))
          .toList();
    } on PlatformException catch (e) {
      print('NativeChannels.ragSearch: PlatformException: ${e.message}');
      return [];
    } on MissingPluginException catch (e) {
      print('NativeChannels.ragSearch: MissingPluginException: ${e.message}');
      return [];
    } catch (e) {
      print('NativeChannels.ragSearch: Unknown error: $e');
      return [];
    }
  }

  static Future<Map<String, dynamic>> llmStatus() async {
    try {
      final res = await _llm.invokeMethod<dynamic>('status');
      if (res is Map) {
        return res.map((k, v) => MapEntry(k.toString(), v));
      }
      return const { 'state': 'unknown', 'progress': 0.0, 'ready': false };
    } catch (e) {
      return const { 'state': 'error', 'progress': 0.0, 'ready': false };
    }
  }

  static Future<void> downloadModel() async {
    try {
      await _llm.invokeMethod<void>('downloadModel');
    } on PlatformException catch (e) {
      print('PlatformException in downloadModel: ${e.message}');
      rethrow;
    } catch (e) {
      print('Unexpected error in downloadModel: $e');
      rethrow;
    }
  }
  
  static Future<void> clearModelCache() async {
    try {
      await _llm.invokeMethod<void>('clearCache');
    } on PlatformException catch (e) {
      print('PlatformException in clearCache: ${e.message}');
      rethrow;
    } catch (e) {
      print('Unexpected error in clearCache: $e');
      rethrow;
    }
  }
  
  static Future<Map<String, dynamic>> checkModelCache() async {
    try {
      final res = await _llm.invokeMethod<dynamic>('checkCache');
      if (res is Map) {
        return res.map((k, v) => MapEntry(k.toString(), v));
      }
      return const { 'cached': false, 'path': '', 'files': [] };
    } catch (e) {
      print('Unexpected error in checkCache: $e');
      return const { 'cached': false, 'path': '', 'files': [] };
    }
  }
}

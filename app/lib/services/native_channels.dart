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

  static Future<String> generate(String prompt) async {
    try {
      final res = await _llm.invokeMethod<String>('generate', {'prompt': prompt});
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
}

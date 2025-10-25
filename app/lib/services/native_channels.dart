import 'package:flutter/services.dart';

class NativeChannels {
  static const _embed = MethodChannel('survival/embed');
  static const _llm = MethodChannel('survival/llm');

  static Future<List<double>> embed(String text) async {
    final res = await _embed.invokeMethod<List<dynamic>>('embed', {
      'text': text,
    });
    return res?.map((e) => (e as num).toDouble()).toList() ?? [];
  }

  static Future<String> generate(String prompt) async {
    final res = await _llm.invokeMethod<String>('generate', {'prompt': prompt});
    return res ?? "";
  }
}

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
  String _output = "";

  Future<void> _testEmbed() async {
    final v = await NativeChannels.embed("hypothermia checklist");
    setState(() => _output = "Got ${v.length} dims");
  }

  Future<void> _testLLM() async {
    final res = await NativeChannels.generate(_controller.text);
    setState(() => _output = res);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Offline Survival Assistant")),
      body: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          children: [
            TextField(
              controller: _controller,
              decoration: const InputDecoration(labelText: "Ask something"),
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                ElevatedButton(
                  onPressed: _testEmbed,
                  child: const Text("Test Embed"),
                ),
                const SizedBox(width: 12),
                ElevatedButton(
                  onPressed: _testLLM,
                  child: const Text("Test LLM"),
                ),
              ],
            ),
            const Divider(),
            Expanded(child: SingleChildScrollView(child: Text(_output))),
          ],
        ),
      ),
    );
  }
}

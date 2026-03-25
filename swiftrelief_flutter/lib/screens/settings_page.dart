import 'package:flutter/material.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:http/http.dart' as http;
import '../services/api_client.dart';

class SettingsPage extends StatefulWidget {
  final ApiClient api;
  const SettingsPage({super.key, required this.api});

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  final _controller = TextEditingController();
  bool _saving = false;
  final _storage = const FlutterSecureStorage();

  @override
  void initState() {
    super.initState();
    _controller.text = widget.api.baseUrl;
  }

  Future<void> _save() async {
    setState(() => _saving = true);
    await _storage.write(key: 'api_base_url', value: _controller.text.trim());
    setState(() => _saving = false);
    ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
      content: Text('Saved. Restart the app to apply the new server URL.'),
    ));
  }

  Future<void> _test() async {
    final url = _controller.text.trim();
    if (url.isEmpty) return;
    final uri = Uri.parse(url);
    final ping = uri.replace(path: '/');
    setState(() => _saving = true);
    try {
      final res = await http.get(ping).timeout(const Duration(seconds: 6));
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text('Response ${res.statusCode} from $url'),
      ));
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text('Connection failed: $e'),
      ));
    } finally {
      setState(() => _saving = false);
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('API Base URL', style: TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            TextField(
              controller: _controller,
              decoration: const InputDecoration(border: OutlineInputBorder()),
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                ElevatedButton(
                  onPressed: _saving ? null : _save,
                  child: const Text('Save'),
                ),
                const SizedBox(width: 12),
                OutlinedButton(
                  onPressed: _saving ? null : _test,
                  child: const Text('Test connection'),
                ),
              ],
            ),
            const SizedBox(height: 16),
            const Text('Note: For production apps host the backend under a stable domain. This setting is useful for development and local testing.'),
          ],
        ),
      ),
    );
  }
}

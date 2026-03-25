import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/auth_provider.dart';

class RegisterPage extends StatefulWidget {
  const RegisterPage({super.key});

  @override
  State<RegisterPage> createState() => _RegisterPageState();
}

class _RegisterPageState extends State<RegisterPage> {
  final _formKey = GlobalKey<FormState>();
  String _name = '';
  String _email = '';
  String _password = '';
  int _age = 18;
  String _gender = 'Other';
  final Map<String, bool> _conditions = {
    'asthma_copd': false,
    'diabetes': false,
    'heart_disease': false,
    'stroke_history': false,
    'epilepsy': false,
    'pregnant': false,
  };
  String _otherInfo = '';
  bool _loading = false;

  Widget _buildBrandLogo({double size = 68}) {
    final zoom = size * 1.65;
    return ClipRRect(
      borderRadius: BorderRadius.circular(size * 0.24),
      child: SizedBox(
        width: size,
        height: size,
        child: OverflowBox(
          minWidth: zoom,
          maxWidth: zoom,
          minHeight: zoom,
          maxHeight: zoom,
          child: Image.asset(
            'assets/icons/swiftrelief_cropped.png',
            width: zoom,
            height: zoom,
            fit: BoxFit.cover,
          ),
        ),
      ),
    );
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;
    _formKey.currentState!.save();
    setState(() => _loading = true);
    try {
      await context.read<AuthProvider>().register(
        _name.trim(),
        _email.trim(),
        _password,
        _age,
        gender: _gender,
        conditions: _conditions,
        otherInfo: _otherInfo,
          );
      if (!mounted) return;
      Navigator.of(context).pop();
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Register failed: $e')));
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
              colors: [Color(0xFFFFF5F5), Colors.white],
            ),
          ),
          child: Center(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 460),
                child: DecoratedBox(
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(28),
                    border: Border.all(color: const Color(0xFFFECACA)),
                    boxShadow: const [
                      BoxShadow(color: Color(0x1AF87171), blurRadius: 20, offset: Offset(0, 10)),
                    ],
                  ),
                  child: Padding(
                    padding: const EdgeInsets.fromLTRB(20, 22, 20, 20),
                    child: Form(
                      key: _formKey,
                      child: ListView(shrinkWrap: true, children: [
                        Center(child: _buildBrandLogo(size: 72)),
                        const SizedBox(height: 10),
                        const Center(
                          child: Text(
                            'Create your account',
                            style: TextStyle(fontSize: 24, fontWeight: FontWeight.w800, color: Color(0xFFDC2626)),
                          ),
                        ),
                        const SizedBox(height: 6),
                        const Center(
                          child: Text(
                            'Set up your profile for smarter recommendations',
                            style: TextStyle(color: Color(0xFF6B7280)),
                          ),
                        ),
                        const SizedBox(height: 16),
                        TextFormField(
                          decoration: const InputDecoration(
                            labelText: 'Name',
                            border: OutlineInputBorder(),
                            prefixIcon: Icon(Icons.person_outline_rounded),
                          ),
                          validator: (v) => (v == null || v.isEmpty) ? 'Enter name' : null,
                          onSaved: (v) => _name = v ?? '',
                        ),
                        const SizedBox(height: 10),
                        TextFormField(
                          decoration: const InputDecoration(
                            labelText: 'Email',
                            border: OutlineInputBorder(),
                            prefixIcon: Icon(Icons.email_outlined),
                          ),
                          keyboardType: TextInputType.emailAddress,
                          validator: (v) => (v == null || !v.contains('@')) ? 'Enter valid email' : null,
                          onSaved: (v) => _email = v ?? '',
                        ),
                        const SizedBox(height: 10),
                        TextFormField(
                          decoration: const InputDecoration(
                            labelText: 'Password',
                            border: OutlineInputBorder(),
                            prefixIcon: Icon(Icons.lock_outline_rounded),
                          ),
                          obscureText: true,
                          validator: (v) => (v == null || v.length < 6) ? 'Password too short' : null,
                          onSaved: (v) => _password = v ?? '',
                        ),
                        const SizedBox(height: 10),
                        TextFormField(
                          decoration: const InputDecoration(
                            labelText: 'Age',
                            border: OutlineInputBorder(),
                            prefixIcon: Icon(Icons.cake_outlined),
                          ),
                          keyboardType: TextInputType.number,
                          initialValue: '18',
                          validator: (v) {
                            final n = int.tryParse(v ?? '');
                            if (n == null || n < 0 || n > 120) return 'Enter valid age';
                            return null;
                          },
                          onSaved: (v) => _age = int.tryParse(v ?? '18') ?? 18,
                        ),
                        const SizedBox(height: 10),
                        DropdownButtonFormField<String>(
                          initialValue: _gender,
                          items: const [
                            DropdownMenuItem(value: 'Male', child: Text('Male')),
                            DropdownMenuItem(value: 'Female', child: Text('Female')),
                            DropdownMenuItem(value: 'Other', child: Text('Other')),
                          ],
                          onChanged: (v) {
                            if (v != null) setState(() => _gender = v);
                          },
                          decoration: const InputDecoration(
                            labelText: 'Gender',
                            border: OutlineInputBorder(),
                          ),
                        ),
                        const SizedBox(height: 10),
                        const Text('Health Profile (optional)', style: TextStyle(fontWeight: FontWeight.w600)),
                        CheckboxListTile(
                          contentPadding: EdgeInsets.zero,
                          title: const Text('Asthma / COPD'),
                          value: _conditions['asthma_copd'],
                          onChanged: (v) => setState(() => _conditions['asthma_copd'] = v ?? false),
                        ),
                        CheckboxListTile(
                          contentPadding: EdgeInsets.zero,
                          title: const Text('Diabetes'),
                          value: _conditions['diabetes'],
                          onChanged: (v) => setState(() => _conditions['diabetes'] = v ?? false),
                        ),
                        CheckboxListTile(
                          contentPadding: EdgeInsets.zero,
                          title: const Text('Heart disease'),
                          value: _conditions['heart_disease'],
                          onChanged: (v) => setState(() => _conditions['heart_disease'] = v ?? false),
                        ),
                        CheckboxListTile(
                          contentPadding: EdgeInsets.zero,
                          title: const Text('Stroke history'),
                          value: _conditions['stroke_history'],
                          onChanged: (v) => setState(() => _conditions['stroke_history'] = v ?? false),
                        ),
                        CheckboxListTile(
                          contentPadding: EdgeInsets.zero,
                          title: const Text('Epilepsy'),
                          value: _conditions['epilepsy'],
                          onChanged: (v) => setState(() => _conditions['epilepsy'] = v ?? false),
                        ),
                        CheckboxListTile(
                          contentPadding: EdgeInsets.zero,
                          title: const Text('Pregnant'),
                          value: _conditions['pregnant'],
                          onChanged: (v) => setState(() => _conditions['pregnant'] = v ?? false),
                        ),
                        TextFormField(
                          decoration: const InputDecoration(
                            labelText: 'Other health notes (optional)',
                            border: OutlineInputBorder(),
                          ),
                          minLines: 2,
                          maxLines: 4,
                          onSaved: (v) => _otherInfo = (v ?? '').trim(),
                        ),
                        const SizedBox(height: 16),
                        ElevatedButton.icon(
                          onPressed: _loading ? null : _submit,
                          icon: const Icon(Icons.person_add_alt_1_rounded, size: 18),
                          label: Text(_loading ? 'Registering...' : 'Register'),
                        ),
                      ]),
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

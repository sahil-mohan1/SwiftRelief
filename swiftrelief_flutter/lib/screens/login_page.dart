import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/auth_provider.dart';

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final _formKey = GlobalKey<FormState>();
  String _email = '';
  String _password = '';
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
      await context.read<AuthProvider>().login(_email.trim(), _password);
      if (!mounted) return;
      Navigator.of(context).pop();
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Login failed: $e')));
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
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(20),
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 430),
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
                    padding: const EdgeInsets.fromLTRB(22, 26, 22, 22),
                    child: Form(
                      key: _formKey,
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          Center(child: _buildBrandLogo(size: 72)),
                          const SizedBox(height: 12),
                          const Center(
                            child: Text(
                              'Welcome back',
                              style: TextStyle(fontSize: 24, fontWeight: FontWeight.w800, color: Color(0xFFDC2626)),
                            ),
                          ),
                          const SizedBox(height: 6),
                          const Center(
                            child: Text(
                              'Continue to smart emergency recommendations',
                              style: TextStyle(color: Color(0xFF6B7280)),
                            ),
                          ),
                          const SizedBox(height: 20),
                          TextFormField(
                            decoration: const InputDecoration(
                              labelText: 'Email',
                              prefixIcon: Icon(Icons.email_outlined),
                              border: OutlineInputBorder(),
                            ),
                            keyboardType: TextInputType.emailAddress,
                            validator: (v) => (v == null || v.isEmpty || !v.contains('@')) ? 'Enter a valid email' : null,
                            onSaved: (v) => _email = v ?? '',
                          ),
                          const SizedBox(height: 12),
                          TextFormField(
                            decoration: const InputDecoration(
                              labelText: 'Password',
                              prefixIcon: Icon(Icons.lock_outline_rounded),
                              border: OutlineInputBorder(),
                            ),
                            obscureText: true,
                            validator: (v) => (v == null || v.length < 6) ? 'Password too short' : null,
                            onSaved: (v) => _password = v ?? '',
                          ),
                          const SizedBox(height: 18),
                          ElevatedButton.icon(
                            onPressed: _loading ? null : _submit,
                            icon: const Icon(Icons.login_rounded, size: 18),
                            label: Text(_loading ? 'Logging in...' : 'Login'),
                          ),
                          const SizedBox(height: 8),
                          OutlinedButton.icon(
                            onPressed: () => Navigator.of(context).pushNamed('/register'),
                            icon: const Icon(Icons.person_add_alt_1_rounded, size: 18),
                            label: const Text('Create account'),
                          ),
                        ],
                      ),
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

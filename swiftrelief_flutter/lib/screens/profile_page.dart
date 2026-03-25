import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/user.dart';
import '../providers/auth_provider.dart';
import '../repositories/cache_repository.dart';
import '../services/api_client.dart';

class ProfilePage extends StatefulWidget {
  final ApiClient api;
  const ProfilePage({super.key, required this.api});

  @override
  State<ProfilePage> createState() => _ProfilePageState();
}

class _ProfilePageState extends State<ProfilePage> {
  final _formKey = GlobalKey<FormState>();

  final TextEditingController _nameController = TextEditingController();
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _ageController = TextEditingController();
  final TextEditingController _otherInfoController = TextEditingController();

  bool _loading = true;
  bool _saving = false;
  String _gender = 'Other';
  bool _preferCategorySelector = false;

  final Map<String, bool> _conditions = {
    'asthma_copd': false,
    'diabetes': false,
    'heart_disease': false,
    'stroke_history': false,
    'epilepsy': false,
    'pregnant': false,
  };

  @override
  void initState() {
    super.initState();
    _loadProfile();
  }

  bool _asBool(dynamic value) {
    if (value is bool) return value;
    if (value is num) return value != 0;
    if (value is String) {
      final text = value.toLowerCase().trim();
      return text == '1' || text == 'true' || text == 'yes';
    }
    return false;
  }

  Future<void> _loadProfile() async {
    final token = context.read<AuthProvider>().token;
    if (token == null || token.isEmpty) {
      if (!mounted) return;
      Navigator.of(context).pop();
      return;
    }

    try {
      final res = await widget.api.get('/api/profile', token: token);
      final profile = res['profile'];
      if (profile is! Map<String, dynamic>) {
        throw Exception('Invalid profile response');
      }

      final conditions = profile['conditions'];
      final conditionMap = conditions is Map ? Map<String, dynamic>.from(conditions) : <String, dynamic>{};

      _nameController.text = '${profile['name'] ?? ''}';
      _emailController.text = '${profile['email'] ?? ''}';
      _ageController.text = '${profile['age'] ?? ''}';
      _otherInfoController.text = '${profile['other_info'] ?? ''}';
      _gender = '${profile['gender'] ?? 'Other'}';

      _conditions['asthma_copd'] = _asBool(conditionMap['asthma_copd']);
      _conditions['diabetes'] = _asBool(conditionMap['diabetes']);
      _conditions['heart_disease'] = _asBool(conditionMap['heart_disease']);
      _conditions['stroke_history'] = _asBool(conditionMap['stroke_history']);
      _conditions['epilepsy'] = _asBool(conditionMap['epilepsy']);
      _conditions['pregnant'] = _asBool(conditionMap['pregnant']);

      _preferCategorySelector = CacheRepository.read(CacheRepository.useCategorySelectorKey) == true;
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Failed to load profile: $e')));
    } finally {
      if (mounted) {
        setState(() => _loading = false);
      }
    }
  }

  Future<void> _saveProfile() async {
    if (!_formKey.currentState!.validate()) return;

    final age = int.tryParse(_ageController.text.trim());
    if (age == null || age < 0 || age > 120) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Enter valid age (0-120)')));
      return;
    }

    final token = context.read<AuthProvider>().token;
    if (token == null || token.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Please login again')));
      return;
    }

    setState(() => _saving = true);
    try {
      await widget.api.put('/api/profile', {
        'name': _nameController.text.trim(),
        'age': age,
        'gender': _gender,
        'conditions': _conditions,
        'other_info': _otherInfoController.text.trim(),
      }, token: token);

      await CacheRepository.save(
        CacheRepository.useCategorySelectorKey,
        _preferCategorySelector,
      );

      final meRes = await widget.api.get('/api/auth/me', token: token);
      final rawUser = meRes['user'];
      if (rawUser is Map<String, dynamic>) {
        if (!mounted) return;
        final auth = context.read<AuthProvider>();
        final currentToken = auth.token;
        if (currentToken != null) {
          await auth.saveToStorage(currentToken, User.fromJson(rawUser));
        }
      }

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Profile updated')));
      Navigator.of(context).pop();
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Failed to save profile: $e')));
    } finally {
      if (mounted) {
        setState(() => _saving = false);
      }
    }
  }

  Future<void> _logout() async {
    await context.read<AuthProvider>().logout();
    if (!mounted) return;
    Navigator.of(context).popUntil((route) => route.isFirst);
  }

  Widget _sectionCard({required Widget child}) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: const Color(0xFFFECACA)),
        boxShadow: const [
          BoxShadow(color: Color(0x1AF87171), blurRadius: 16, offset: Offset(0, 8)),
        ],
      ),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: child,
      ),
    );
  }

  @override
  void dispose() {
    _nameController.dispose();
    _emailController.dispose();
    _ageController.dispose();
    _otherInfoController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Your Profile'),
        centerTitle: true,
        actions: [
          IconButton(
            onPressed: _logout,
            icon: const Icon(Icons.logout),
            tooltip: 'Logout',
          ),
        ],
      ),
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Color(0xFFFFF5F5), Colors.white],
          ),
        ),
        child: _loading
            ? const Center(child: CircularProgressIndicator())
            : Form(
                key: _formKey,
                child: ListView(
                  padding: const EdgeInsets.all(16),
                  children: [
                    _sectionCard(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          const Text('Personal details', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700)),
                          const SizedBox(height: 12),
                          TextFormField(
                            controller: _nameController,
                            decoration: const InputDecoration(
                              labelText: 'Name',
                              prefixIcon: Icon(Icons.person_outline_rounded),
                              border: OutlineInputBorder(),
                            ),
                          ),
                          const SizedBox(height: 12),
                          TextFormField(
                            controller: _emailController,
                            enabled: false,
                            decoration: const InputDecoration(
                              labelText: 'Email',
                              prefixIcon: Icon(Icons.email_outlined),
                              border: OutlineInputBorder(),
                            ),
                          ),
                          const SizedBox(height: 12),
                          TextFormField(
                            controller: _ageController,
                            keyboardType: TextInputType.number,
                            decoration: const InputDecoration(
                              labelText: 'Age',
                              prefixIcon: Icon(Icons.cake_outlined),
                              border: OutlineInputBorder(),
                            ),
                            validator: (v) {
                              final n = int.tryParse((v ?? '').trim());
                              if (n == null || n < 0 || n > 120) return 'Enter valid age (0-120)';
                              return null;
                            },
                          ),
                          const SizedBox(height: 12),
                          DropdownButtonFormField<String>(
                            initialValue: _gender,
                            items: const [
                              DropdownMenuItem(value: 'Male', child: Text('Male')),
                              DropdownMenuItem(value: 'Female', child: Text('Female')),
                              DropdownMenuItem(value: 'Other', child: Text('Other')),
                            ],
                            onChanged: (v) {
                              if (v != null) {
                                setState(() => _gender = v);
                              }
                            },
                            decoration: const InputDecoration(
                              labelText: 'Gender',
                              border: OutlineInputBorder(),
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 12),
                    _sectionCard(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          const Text('Medical complications', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700)),
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
                          const SizedBox(height: 8),
                          TextFormField(
                            controller: _otherInfoController,
                            minLines: 2,
                            maxLines: 4,
                            decoration: const InputDecoration(
                              labelText: 'Other notes',
                              border: OutlineInputBorder(),
                            ),
                          ),
                          const SizedBox(height: 12),
                          const Text('Search preference', style: TextStyle(fontWeight: FontWeight.w700)),
                          CheckboxListTile(
                            contentPadding: EdgeInsets.zero,
                            title: const Text('Prefer category selector (instead of symptom input)'),
                            subtitle: const Text(
                              'When enabled, home search shows category dropdown. Otherwise you can type free-text symptoms.',
                            ),
                            value: _preferCategorySelector,
                            onChanged: (v) => setState(() => _preferCategorySelector = v ?? false),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 16),
                    ElevatedButton.icon(
                      onPressed: _saving ? null : _saveProfile,
                      icon: const Icon(Icons.save_outlined),
                      label: Text(_saving ? 'Saving...' : 'Save changes'),
                    ),
                  ],
                ),
              ),
      ),
    );
  }
}

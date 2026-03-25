import 'dart:convert';

import 'package:crypto/crypto.dart';
import 'package:flutter/material.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import '../models/user.dart';
import '../services/api_client.dart';

class AuthProvider with ChangeNotifier {
  final FlutterSecureStorage _storage = const FlutterSecureStorage();
  final ApiClient api;

  static const String _offlineCredsKey = 'offline_login_profiles_v1';

  User? _user;
  String? _token;

  AuthProvider(this.api);

  User? get user => _user;
  String? get token => _token;

  Future<void> loadFromStorage() async {
    final t = await _storage.read(key: 'token');
    final u = await _storage.read(key: 'user');
    if (t != null) _token = t;
    if (u != null) {
      try {
        final decoded = json.decode(u);
        if (decoded is Map) {
          _user = User.fromJson(Map<String, dynamic>.from(decoded));
        }
      } catch (_) {
        await _storage.delete(key: 'user');
      }
    }
    notifyListeners();
  }

  Future<void> saveToStorage(String token, User user) async {
    _token = token;
    _user = user;
    await _storage.write(key: 'token', value: token);
    await _storage.write(key: 'user', value: json.encode(user.toJson()));
    notifyListeners();
  }

  Future<void> logout() async {
    _token = null;
    _user = null;
    await _storage.delete(key: 'token');
    await _storage.delete(key: 'user');
    notifyListeners();
  }

  Future<void> login(String email, String password) async {
    final normalizedEmail = email.trim().toLowerCase();
    try {
      final res = await api.post('/api/auth/login', {'email': email, 'password': password});
      final token = res['token'] as String?;
      final rawUser = res['user'];
      if (token == null || rawUser is! Map) {
        final message = res['message']?.toString() ?? 'Login failed';
        throw Exception(message);
      }
      final user = User.fromJson(Map<String, dynamic>.from(rawUser));
      await saveToStorage(token, user);
      await _saveOfflineCredentials(email: normalizedEmail, password: password, user: user);
    } catch (e) {
      final cachedUser = await _verifyOfflineCredentials(email: normalizedEmail, password: password);
      if (cachedUser != null) {
        await saveToStorage('', cachedUser);
        return;
      }

      if (_isConnectivityError(e)) {
        throw Exception('Offline login unavailable. Connect once online, then retry offline.');
      }
      rethrow;
    }
  }

  Future<void> register(
    String name,
    String email,
    String password,
    int age, {
    String gender = 'Other',
    Map<String, dynamic>? conditions,
    String otherInfo = '',
  }) async {
    final res = await api.post('/api/auth/register', {
      'name': name,
      'email': email,
      'password': password,
      'age': age,
      'gender': gender,
      'conditions': conditions ?? {},
      'other_info': otherInfo,
    });
    final token = res['token'] as String?;
    final rawUser = res['user'];
    if (token == null || rawUser is! Map) {
      final message = res['message']?.toString() ?? 'Registration failed';
      throw Exception(message);
    }
    final user = User.fromJson(Map<String, dynamic>.from(rawUser));
    await saveToStorage(token, user);
    await _saveOfflineCredentials(email: email.trim().toLowerCase(), password: password, user: user);
  }

  String _passwordHash(String email, String password) {
    final normalizedEmail = email.trim().toLowerCase();
    final input = '$normalizedEmail::$password';
    return sha256.convert(utf8.encode(input)).toString();
  }

  bool _isConnectivityError(Object error) {
    final msg = error.toString().toLowerCase();
    return msg.contains('network error') || msg.contains('request timed out') || msg.contains('socket');
  }

  Future<Map<String, dynamic>> _readOfflineProfiles() async {
    final raw = await _storage.read(key: _offlineCredsKey);
    if (raw == null || raw.trim().isEmpty) return <String, dynamic>{};
    try {
      final decoded = json.decode(raw);
      if (decoded is Map) return Map<String, dynamic>.from(decoded);
    } catch (_) {}
    return <String, dynamic>{};
  }

  Future<void> _writeOfflineProfiles(Map<String, dynamic> profiles) async {
    await _storage.write(key: _offlineCredsKey, value: json.encode(profiles));
  }

  Future<void> _saveOfflineCredentials({
    required String email,
    required String password,
    required User user,
  }) async {
    final profiles = await _readOfflineProfiles();
    profiles[email] = {
      'email': email,
      'password_hash': _passwordHash(email, password),
      'user': user.toJson(),
      'updated_at': DateTime.now().toIso8601String(),
    };
    await _writeOfflineProfiles(profiles);
  }

  Future<User?> _verifyOfflineCredentials({
    required String email,
    required String password,
  }) async {
    final profiles = await _readOfflineProfiles();
    final raw = profiles[email];
    if (raw is! Map) return null;
    final profile = Map<String, dynamic>.from(raw);
    final expected = '${profile['password_hash'] ?? ''}';
    if (expected.isEmpty) return null;
    if (_passwordHash(email, password) != expected) return null;
    final rawUser = profile['user'];
    if (rawUser is! Map) return null;
    return User.fromJson(Map<String, dynamic>.from(rawUser));
  }
}

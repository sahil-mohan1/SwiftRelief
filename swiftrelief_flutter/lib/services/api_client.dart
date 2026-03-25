import 'dart:async';
import 'dart:convert';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:http/http.dart' as http;

class ApiClient {
  // Default compile-time URL (can be overridden at runtime by saved value)
  static const String _compileTimeBase = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'http://172.25.5.135:5000',
  );

  static const Duration _defaultTimeout = Duration(seconds: 25);

  final String baseUrl;
  final http.Client _http;

  ApiClient([String? baseUrl, http.Client? client])
      : baseUrl = baseUrl ?? _compileTimeBase,
        _http = client ?? http.Client();

  /// Async constructor that reads an optional runtime `api_base_url` from
  /// secure storage and falls back to the compile-time value.
  static Future<ApiClient> create([http.Client? client]) async {
    final storage = const FlutterSecureStorage();
    final runtime = await storage.read(key: 'api_base_url');
    final base = runtime ?? _compileTimeBase;
    return ApiClient(base, client);
  }

  Map<String, String> _headers([String? token]) {
    final headers = {'Content-Type': 'application/json'};
    if (token != null && token.isNotEmpty) {
      headers['Authorization'] = 'Bearer $token';
    }
    return headers;
  }

  Future<Map<String, dynamic>> post(
    String path,
    Map body, {
    String? token,
    Duration? timeout,
  }) async {
    final uri = Uri.parse('$baseUrl$path');
    try {
      final res = await _http
          .post(uri, headers: _headers(token), body: json.encode(body))
          .timeout(timeout ?? _defaultTimeout);
      return _process(res);
    } on TimeoutException {
      throw Exception('Request timed out. Check server URL/network and try again.');
    } on http.ClientException catch (e) {
      throw Exception('Network error: ${e.message}. Check API_BASE_URL and connectivity.');
    }
  }

  Future<Map<String, dynamic>> get(
    String path, {
    Map<String, String>? params,
    String? token,
    Duration? timeout,
  }) async {
    final uri = Uri.parse('$baseUrl$path').replace(queryParameters: params);
    try {
      final res = await _http.get(uri, headers: _headers(token)).timeout(timeout ?? _defaultTimeout);
      return _process(res);
    } on TimeoutException {
      throw Exception('Request timed out. Check server URL/network and try again.');
    } on http.ClientException catch (e) {
      throw Exception('Network error: ${e.message}. Check API_BASE_URL and connectivity.');
    }
  }

  Future<Map<String, dynamic>> put(
    String path,
    Map body, {
    String? token,
    Duration? timeout,
  }) async {
    final uri = Uri.parse('$baseUrl$path');
    try {
      final res = await _http
          .put(uri, headers: _headers(token), body: json.encode(body))
          .timeout(timeout ?? _defaultTimeout);
      return _process(res);
    } on TimeoutException {
      throw Exception('Request timed out. Check server URL/network and try again.');
    } on http.ClientException catch (e) {
      throw Exception('Network error: ${e.message}. Check API_BASE_URL and connectivity.');
    }
  }

  Map<String, dynamic> _process(http.Response res) {
    final body = res.body.isEmpty ? '{}' : res.body;
    final decoded = json.decode(body);
    if (res.statusCode >= 200 && res.statusCode < 300) {
      return decoded as Map<String, dynamic>;
    }
    throw ApiException(res.statusCode, decoded);
  }
}

class ApiException implements Exception {
  final int statusCode;
  final dynamic body;
  ApiException(this.statusCode, this.body);

  @override
  String toString() => 'ApiException($statusCode): $body';
}

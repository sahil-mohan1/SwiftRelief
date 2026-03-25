import 'dart:convert';

import '../services/api_client.dart';
import 'cache_repository.dart';

class RecommendRepository {
  final ApiClient api;
  RecommendRepository(this.api);

  static const _cacheKeyPrefix = 'last_recommendations';
  static const _searchHistoryPrefix = 'search_history_v1';

  String _normEmergencyType(String emergencyType) => emergencyType.trim().toLowerCase();

  String _normSymptom(String symptom) => symptom.trim().toLowerCase();

  String _normModelSignature(String modelSignature) {
    final value = modelSignature.trim();
    if (value.isEmpty) return 'unknown-model';
    return value.toLowerCase();
  }

  String _coordKey(double value) => value.toStringAsFixed(6);

  String _lastRecommendationsKey(
    String userScope,
    double lat,
    double lon,
    String emergencyType,
    String modelSignature,
    String symptom,
  ) {
    final et = _normEmergencyType(emergencyType);
    final model = _normModelSignature(modelSignature);
    final symptomNorm = _normSymptom(symptom);
    return '$_cacheKeyPrefix::$userScope::${_coordKey(lat)},${_coordKey(lon)}::$et::$model::$symptomNorm';
  }

  String _searchHistoryKey(String userScope) => '$_searchHistoryPrefix::$userScope';

  String _normalizeQuery(String query) => query.trim().toLowerCase();

  Future<Map<String, dynamic>> fetchRecommendations(
    double lat,
    double lon, {
    String userScope = 'global',
    String? token,
    String emergencyType = 'All (No filter)',
    bool offlineMode = false,
    String symptom = '',
  }) async {
    final cleanSymptom = symptom.trim();
    final body = {
      'lat': lat,
      'lon': lon,
      'emergency_type': emergencyType,
      'offline_mode': offlineMode,
      'top_k': 5,
      'shortlist': 20,
    };
    if (cleanSymptom.isNotEmpty) {
      body['symptom'] = cleanSymptom;
    }
    final res = await api.post(
      '/api/recommend',
      body,
      token: token,
      timeout: const Duration(seconds: 120),
    );
    final payload = res['data'];
    List<dynamic> data = [];
    int? runId;
    Map<String, dynamic>? explainMeta;
    String modelSignature = 'unknown-model';
    if (payload is List) {
      data = payload;
    } else if (payload is Map<String, dynamic>) {
      data = payload['results'] as List<dynamic>? ?? [];
      final rawRunId = payload['run_id'];
      runId = rawRunId is int ? rawRunId : int.tryParse('${rawRunId ?? ''}');
      final rawExplain = payload['explain_meta'];
      if (rawExplain is Map<String, dynamic>) {
        explainMeta = rawExplain;
        final modelKey = '${rawExplain['model_key'] ?? 'unknown'}'.trim().toLowerCase();
        final modelDir = '${rawExplain['model_dir'] ?? ''}'.trim().toLowerCase();
        final modelEnabled = rawExplain['model_enabled'] == true ? '1' : '0';
        modelSignature = _normModelSignature('$modelKey|$modelDir|$modelEnabled');
      }
    }
    // Cache raw JSON
    try {
      await CacheRepository.save(
        _lastRecommendationsKey(
          userScope,
          lat,
          lon,
          emergencyType,
          modelSignature,
          cleanSymptom,
        ),
        json.encode(data),
      );
    } catch (_) {}
    return {
      'run_id': runId,
      'results': data,
      'model_signature': modelSignature,
      'explain_meta': explainMeta,
    };
  }

  Future<void> submitFeedback({
    required int runId,
    required int hospitalId,
    int? runCandidateId,
    required int thumbs,
    required String reasonCode,
    String? token,
  }) async {
    final body = {
      'run_id': runId,
      'hospital_id': hospitalId,
      'run_candidate_id': runCandidateId,
      'thumbs': thumbs,
      'reason_code': reasonCode,
    };
    await api.post('/api/feedback', body, token: token);
  }

  Future<Map<String, dynamic>> geocodePlace(
    String query, {
    bool offline = false,
    String? token,
  }) async {
    final res = await api.get(
      '/api/geocode',
      params: {
        'query': query,
        'region': 'in',
        'offline': offline ? '1' : '0',
      },
      token: token,
    );
    final data = res['data'];
    if (data is Map<String, dynamic>) {
      return data;
    }
    throw Exception('Invalid geocode response');
  }

  Future<String?> mapSymptomToCategory(
    String symptom, {
    String? token,
  }) async {
    final clean = symptom.trim();
    if (clean.isEmpty) return null;
    final res = await api.post('/api/map_symptom', {
      'symptom': clean,
      'use_llm': true,
    }, token: token);
    final category = '${res['category'] ?? ''}'.trim();
    if (category.isEmpty || category.toLowerCase() == 'general') {
      return null;
    }
    return category;
  }

  List<dynamic> loadCached({
    required String userScope,
    required double lat,
    required double lon,
    required String emergencyType,
    required String modelSignature,
    String symptom = '',
  }) {
    try {
      final raw = CacheRepository.read(
        _lastRecommendationsKey(
          userScope,
          lat,
          lon,
          emergencyType,
          modelSignature,
          symptom,
        ),
      ) as String?;
      if (raw == null) return [];
      return json.decode(raw) as List<dynamic>;
    } catch (_) {
      return [];
    }
  }

  Future<void> saveSearchHistoryEntry({
    required String userScope,
    required String query,
    required String emergencyType,
    required String modelSignature,
    required double lat,
    required double lon,
    required String placeLabel,
    required List<Map<String, dynamic>> results,
    String symptom = '',
  }) async {
    final normalizedQuery = _normalizeQuery(query);
    final normalizedSymptom = _normSymptom(symptom);
    if (normalizedQuery.isEmpty) return;

    final normalizedModel = _normModelSignature(modelSignature);
    final existing = loadSearchHistory(userScope: userScope);
    existing.removeWhere((entry) {
      final q = '${entry['query'] ?? ''}'.trim().toLowerCase();
      final e = _normEmergencyType('${entry['emergency_type'] ?? ''}');
      final m = _normModelSignature('${entry['model_signature'] ?? ''}');
      final s = _normSymptom('${entry['symptom'] ?? ''}');
      final latVal = entry['lat'];
      final lonVal = entry['lon'];
      final savedLat = (latVal is num) ? latVal.toDouble() : double.tryParse('$latVal');
      final savedLon = (lonVal is num) ? lonVal.toDouble() : double.tryParse('$lonVal');
      return q == normalizedQuery &&
          e == _normEmergencyType(emergencyType) &&
          m == normalizedModel &&
          s == normalizedSymptom &&
          savedLat != null &&
          savedLon != null &&
          _coordKey(savedLat) == _coordKey(lat) &&
          _coordKey(savedLon) == _coordKey(lon);
    });

    final now = DateTime.now().toIso8601String();
    existing.insert(0, {
      'query': normalizedQuery,
      'query_display': query.trim(),
      'emergency_type': emergencyType,
      'model_signature': normalizedModel,
      'symptom': normalizedSymptom,
      'lat': lat,
      'lon': lon,
      'place_label': placeLabel,
      'results': results,
      'saved_at': now,
    });

    final trimmed = existing.take(30).toList();
    await CacheRepository.save(_searchHistoryKey(userScope), json.encode(trimmed));
  }

  List<Map<String, dynamic>> loadSearchHistory({required String userScope}) {
    try {
      final raw = CacheRepository.read(_searchHistoryKey(userScope)) as String?;
      if (raw == null || raw.trim().isEmpty) return <Map<String, dynamic>>[];
      final decoded = json.decode(raw);
      if (decoded is! List) return <Map<String, dynamic>>[];
      return decoded.whereType<Map>().map((entry) => Map<String, dynamic>.from(entry)).toList();
    } catch (_) {
      return <Map<String, dynamic>>[];
    }
  }

  Map<String, dynamic>? findCachedSearch({
    required String userScope,
    required String query,
    required String emergencyType,
    required String modelSignature,
    required double lat,
    required double lon,
    String symptom = '',
  }) {
    final normalizedQuery = _normalizeQuery(query);
    final normalizedSymptom = _normSymptom(symptom);
    if (normalizedQuery.isEmpty) return null;
    final history = loadSearchHistory(userScope: userScope);
    final normalizedEmergency = _normEmergencyType(emergencyType);
    final normalizedModel = _normModelSignature(modelSignature);

    for (final entry in history) {
      final q = '${entry['query'] ?? ''}'.trim().toLowerCase();
      final e = _normEmergencyType('${entry['emergency_type'] ?? ''}');
      final m = _normModelSignature('${entry['model_signature'] ?? ''}');
      final s = _normSymptom('${entry['symptom'] ?? ''}');
      final latVal = entry['lat'];
      final lonVal = entry['lon'];
      final savedLat = (latVal is num) ? latVal.toDouble() : double.tryParse('$latVal');
      final savedLon = (lonVal is num) ? lonVal.toDouble() : double.tryParse('$lonVal');

      if (q != normalizedQuery) continue;
      if (e != normalizedEmergency) continue;
      if (m != normalizedModel) continue;
      if (s != normalizedSymptom) continue;
      if (savedLat == null || savedLon == null) continue;
      if (_coordKey(savedLat) != _coordKey(lat)) continue;
      if (_coordKey(savedLon) != _coordKey(lon)) continue;
      return entry;
    }
    return null;
  }
}

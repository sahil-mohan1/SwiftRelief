import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:flutter/services.dart';
import 'package:geolocator/geolocator.dart';
import 'package:latlong2/latlong.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';

import 'services/api_client.dart';
import 'services/osm_cached_tile_provider.dart';
import 'repositories/cache_repository.dart';
import 'repositories/recommend_repository.dart';
import 'providers/auth_provider.dart';
import 'screens/login_page.dart';
import 'screens/profile_page.dart';
import 'screens/register_page.dart';
import 'screens/settings_page.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await CacheRepository.init();

  final api = await ApiClient.create();

  runApp(MultiProvider(
    providers: [ChangeNotifierProvider(create: (_) => AuthProvider(api))],
    child: MyApp(api: api),
  ));
}

class MyApp extends StatelessWidget {
  final ApiClient api;
  const MyApp({super.key, required this.api});

  @override
  Widget build(BuildContext context) {
    final colorScheme = ColorScheme.fromSeed(seedColor: Colors.red, brightness: Brightness.light);
    return MaterialApp(
      title: 'SwiftRelief',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: colorScheme,
        scaffoldBackgroundColor: Colors.white,
        appBarTheme: AppBarTheme(
          backgroundColor: Colors.white,
          foregroundColor: colorScheme.primary,
          elevation: 0,
          surfaceTintColor: Colors.transparent,
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.red,
            foregroundColor: Colors.white,
          ),
        ),
        outlinedButtonTheme: OutlinedButtonThemeData(
          style: OutlinedButton.styleFrom(
            foregroundColor: Colors.red,
            side: const BorderSide(color: Color(0xFFFCA5A5)),
          ),
        ),
      ),
      routes: {
        '/': (ctx) => HomePage(api: api),
        '/login': (ctx) => const LoginPage(),
        '/register': (ctx) => const RegisterPage(),
        '/profile': (ctx) => ProfilePage(api: api),
        '/settings': (ctx) => SettingsPage(api: api),
      },
      initialRoute: '/',
    );
  }
}

class HomePage extends StatefulWidget {
  final ApiClient api;
  const HomePage({super.key, required this.api});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  late final RecommendRepository _repo;
  final MapController _mapController = MapController();
  static const String _lastPlaceLabelCacheKey = 'last_place_label';

  List<Map<String, dynamic>> _items = [];
  bool _loading = false;
  String _status = 'Ready';
  final TextEditingController _placeController = TextEditingController();
  final TextEditingController _symptomController = TextEditingController();
  final FocusNode _searchFocusNode = FocusNode();
  String _selectedEmergency = 'All (No filter)';
  bool _useCategorySelector = false;
  bool _offlineMode = false;
  bool _searchExpanded = false;
  String _selectedPlaceLabel = 'your area';
  String? _lastSearchQuery;
  List<Map<String, dynamic>>? _pendingCachedResults;
  String _currentModelSignature = 'unknown-model';
  int? _currentRunId;
  final DraggableScrollableController _sheetController = DraggableScrollableController();
  static const double _sheetMinSize = 0.12;
  static const double _sheetInitialSize = 0.16;
  static const double _sheetMaxSize = 0.72;
  double _sheetSize = _sheetInitialSize;

  double? _lat;
  double? _lon;

  final List<String> _emergencyOptions = const [
    'All (No filter)',
    'Emergency / Trauma',
    'Cardiology (Heart)',
    'Neurology (Brain / Stroke)',
    'Orthopedics (Bones / Fracture)',
    'Ophthalmology (Eye)',
    'Gynecology (Maternity / Women’s Health)',
    'Pediatrics (Child Care)',
    'ENT',
  ];

  final List<Map<String, String>> _positiveReasons = const [
    {'code': 'closest', 'label': 'Closest'},
    {'code': 'matches_need', 'label': 'Matches my need'},
    {'code': 'trusted', 'label': 'Trusted/known hospital'},
    {'code': 'affordable', 'label': 'Affordable'},
  ];

  final List<Map<String, String>> _negativeReasons = const [
    {'code': 'too_far_or_slow', 'label': 'Too far / too slow'},
    {'code': 'not_relevant', 'label': 'Not relevant to my condition'},
    {'code': 'lacks_facilities', 'label': 'Lacks required facilities'},
    {'code': 'low_rating', 'label': 'Low rating / bad reviews'},
    {'code': 'seems_invalid_or_duplicate', 'label': 'Seems invalid / duplicate'},
    {'code': 'private_or_expensive', 'label': 'Private / expensive'},
  ];

  @override
  void initState() {
    super.initState();
    _repo = RecommendRepository(widget.api);
    _searchFocusNode.addListener(() {
      if (_searchFocusNode.hasFocus && mounted) {
        setState(() => _searchExpanded = true);
      }
    });
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      await context.read<AuthProvider>().loadFromStorage();
      if (!mounted) return;
      await _loadSearchPreference();
      _loadCached();
    });
  }

  Future<void> _loadSearchPreference() async {
    final useCategory = CacheRepository.read(CacheRepository.useCategorySelectorKey) == true;
    if (!mounted) return;
    setState(() => _useCategorySelector = useCategory);
  }

  Future<void> _openProfile() async {
    await Navigator.of(context).pushNamed('/profile');
    if (!mounted) return;
    await _loadSearchPreference();
  }

  void _loadCached() {
    final cachedLabel = CacheRepository.read(_lastPlaceLabelCacheKey);
    setState(() {
      _items = <Map<String, dynamic>>[];
      if (cachedLabel is String && cachedLabel.trim().isNotEmpty) {
        _selectedPlaceLabel = cachedLabel.trim();
      }
    });
  }

  String _userScope() {
    final auth = context.read<AuthProvider>();
    final user = auth.user;
    if (user != null) {
      return 'u_${user.email.toLowerCase()}';
    }
    return 'anonymous';
  }

  String _formatPlaceLabel(String rawLabel, {String? fallback}) {
    final cleaned = rawLabel.trim();
    if (cleaned.isEmpty) {
      return (fallback == null || fallback.trim().isEmpty) ? 'your area' : fallback.trim();
    }
    final parts = cleaned
        .split(',')
        .map((part) => part.trim())
        .where((part) => part.isNotEmpty)
        .toList();
    if (parts.isEmpty) {
      return (fallback == null || fallback.trim().isEmpty) ? 'your area' : fallback.trim();
    }

    final city = parts.first;
    final pincode = RegExp(r'\b\d{6}\b').firstMatch(cleaned)?.group(0);
    return pincode == null ? city : '$city, $pincode';
  }

  Future<bool> _searchPlace() async {
    final q = _placeController.text.trim();
    _pendingCachedResults = null;
    _lastSearchQuery = q;
    if (q.isEmpty) {
      setState(() => _status = 'Enter a place first.');
      return false;
    }
    try {
      final token = context.read<AuthProvider>().token;
      final data = await _repo.geocodePlace(q, offline: _offlineMode, token: token);
      final latVal = data['lat'];
      final lonVal = data['lon'];
      final parsedLat = (latVal is String) ? double.tryParse(latVal) : (latVal as num?)?.toDouble();
      final parsedLon = (lonVal is String) ? double.tryParse(lonVal) : (lonVal as num?)?.toDouble();
      if (parsedLat == null || parsedLon == null) {
        setState(() => _status = 'Place found but coordinates are invalid.');
        return false;
      }
      final rawLabel = '${data['display_name'] ?? data['name'] ?? q}';
      final compactLabel = _formatPlaceLabel(rawLabel, fallback: q);
      setState(() {
        _lat = parsedLat;
        _lon = parsedLon;
        _selectedPlaceLabel = compactLabel;
        _status = 'Place found: $_selectedPlaceLabel';
      });
      await CacheRepository.save(_lastPlaceLabelCacheKey, compactLabel);
      _mapController.move(LatLng(parsedLat, parsedLon), 13);
      return true;
    } catch (e) {
      setState(() => _status = 'Place search failed: $e');
      return false;
    }
  }

  Future<void> _sosRecommendFromCurrentLocation() async {
    FocusScope.of(context).unfocus();
    setState(() => _loading = true);
    try {
      final serviceEnabled = await Geolocator.isLocationServiceEnabled();
      if (!serviceEnabled) {
        setState(() => _status = 'Location services are disabled. Enable GPS and try again.');
        return;
      }

      var permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }
      if (permission == LocationPermission.denied || permission == LocationPermission.deniedForever) {
        setState(() => _status = 'Location permission denied. Allow location access for SOS.');
        return;
      }

      final position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
        timeLimit: const Duration(seconds: 20),
      );

      setState(() {
        _lat = position.latitude;
        _lon = position.longitude;
        _selectedPlaceLabel = 'Current location';
        _status = 'Using your current location for SOS search.';
      });
      await CacheRepository.save(_lastPlaceLabelCacheKey, _selectedPlaceLabel);
      _lastSearchQuery = null;
      _pendingCachedResults = null;
      _mapController.move(LatLng(position.latitude, position.longitude), 13);

      await _fetch();
      if (mounted) {
        setState(() => _searchExpanded = false);
      }
    } catch (e) {
      if (mounted) {
        setState(() => _status = 'Unable to get current location: $e');
      }
    } finally {
      if (mounted) {
        setState(() => _loading = false);
      }
    }
  }

  Future<void> _searchAndRecommend() async {
    FocusScope.of(context).unfocus();

    if (_placeController.text.trim().isNotEmpty) {
      final ok = await _searchPlace();
      if (!ok) return;
      if (_pendingCachedResults != null) {
        final cachedResults = _pendingCachedResults!;
        _pendingCachedResults = null;
        setState(() {
          _items = cachedResults;
          _status = 'Showing cached result(s): ${cachedResults.length}.';
        });
          _focusMap(_lat ?? 10.0, _lon ?? 76.0, cachedResults);
        if (mounted) {
          setState(() => _searchExpanded = false);
        }
        return;
      }
    } else if (_lat == null || _lon == null) {
      setState(() => _status = 'Enter a place to search.');
      return;
    }

    await _applySymptomToCategoryFilter();

    await _fetch();
    if (mounted) {
      setState(() => _searchExpanded = false);
    }
  }

  Future<void> _fetch() async {
    setState(() => _loading = true);
    final symptomText = _symptomController.text.trim();
    final requestEmergencyType = _useCategorySelector ? _selectedEmergency : 'All (No filter)';
    try {
      final token = context.read<AuthProvider>().token;
      final useLat = _lat ?? 10.0;
      final useLon = _lon ?? 76.0;
      final response = await _repo.fetchRecommendations(
        useLat,
        useLon,
        userScope: _userScope(),
        token: token,
        emergencyType: requestEmergencyType,
        offlineMode: _offlineMode,
        symptom: symptomText,
      );
      final runId = _asInt(response['run_id']);
      final modelSignature = '${response['model_signature'] ?? ''}'.trim();
      final list = response['results'] as List<dynamic>? ?? [];

      final normalized = list
          .map((item) => item is Map<String, dynamic> ? item : <String, dynamic>{})
          .where((item) => item.isNotEmpty)
          .take(5)
          .toList();

      setState(() {
        _currentRunId = runId;
        if (modelSignature.isNotEmpty) {
          _currentModelSignature = modelSignature;
        }
        _items = normalized;
        _status = 'Fetched ${normalized.length} result(s).';
      });

      final query = _lastSearchQuery;
      if (query != null && query.trim().isNotEmpty) {
        await _repo.saveSearchHistoryEntry(
          userScope: _userScope(),
          query: query,
          emergencyType: requestEmergencyType,
          modelSignature: _currentModelSignature,
          lat: useLat,
          lon: useLon,
          placeLabel: _selectedPlaceLabel,
          results: normalized,
          symptom: symptomText,
        );
      }

      _focusMap(useLat, useLon, normalized);
    } catch (e) {
      if (mounted) {
        final query = _lastSearchQuery;
        if (query != null && query.trim().isNotEmpty) {
          final cached = _repo.findCachedSearch(
            userScope: _userScope(),
            query: query,
            emergencyType: requestEmergencyType,
            modelSignature: _currentModelSignature,
            lat: _lat ?? 0.0,
            lon: _lon ?? 0.0,
            symptom: _symptomController.text.trim(),
          );
          final rawResults = cached?['results'];
          if (rawResults is List) {
            final cachedResults = rawResults
                .map((item) => item is Map<String, dynamic> ? item : <String, dynamic>{})
                .where((item) => item.isNotEmpty)
                .take(5)
                .toList();
            setState(() {
              _items = cachedResults;
              _status = 'Network unavailable. Showing cached results for "$query".';
            });
            _focusMap(_lat ?? 10.0, _lon ?? 76.0, cachedResults);
            return;
          }
        }

        final curLat = _lat;
        final curLon = _lon;
        if (curLat != null && curLon != null) {
          final strictCached = _repo
              .loadCached(
                userScope: _userScope(),
                lat: curLat,
                lon: curLon,
                emergencyType: requestEmergencyType,
                modelSignature: _currentModelSignature,
                symptom: _symptomController.text.trim(),
              )
              .map((item) => item is Map<String, dynamic> ? item : <String, dynamic>{})
              .where((item) => item.isNotEmpty)
              .take(5)
              .toList();
          if (strictCached.isNotEmpty) {
            setState(() {
              _items = strictCached;
              _status = 'Network unavailable. Showing exact-match cached recommendations.';
            });
            _focusMap(curLat, curLon, strictCached);
            return;
          }
        }

        final msg = 'Fetch failed. Ensure Flask is running at http://10.0.2.2:5000. Error: $e';
        setState(() => _status = msg);
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
      }
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  void dispose() {
    _placeController.dispose();
    _symptomController.dispose();
    _searchFocusNode.dispose();
    _sheetController.dispose();
    super.dispose();
  }

  String _normalizeEmergencyLabel(String label) {
    return label
        .trim()
        .toLowerCase()
        .replaceAll('women’s', "women's")
        .replaceAll('’', "'");
  }

  String? _matchEmergencyOption(String category) {
    final normalized = _normalizeEmergencyLabel(category);
    for (final option in _emergencyOptions) {
      if (_normalizeEmergencyLabel(option) == normalized) {
        return option;
      }
    }
    for (final option in _emergencyOptions) {
      final optionNorm = _normalizeEmergencyLabel(option);
      if (optionNorm.contains(normalized) || normalized.contains(optionNorm)) {
        return option;
      }
    }
    return null;
  }

  Future<void> _applySymptomToCategoryFilter() async {
    if (!_useCategorySelector) return;

    final symptom = _symptomController.text.trim();
    if (symptom.isEmpty) return;

    final current = _normalizeEmergencyLabel(_selectedEmergency);
    if (current != _normalizeEmergencyLabel('All (No filter)')) {
      return;
    }

    try {
      final token = context.read<AuthProvider>().token;
      final mapped = await _repo.mapSymptomToCategory(symptom, token: token);
      if (mapped == null) return;

      final matched = _matchEmergencyOption(mapped);
      if (matched != null && matched != _selectedEmergency && mounted) {
        setState(() {
          _selectedEmergency = matched;
          _status = 'Category set from symptom: $matched';
        });
      }
    } catch (_) {
      // Non-blocking: recommendation call still sends symptom for backend-side inference.
    }
  }

  double? _asDouble(dynamic value) {
    if (value is num) return value.toDouble();
    if (value is String) return double.tryParse(value);
    return null;
  }

  int? _asInt(dynamic value) {
    if (value is int) return value;
    if (value is num) return value.toInt();
    if (value is String) return int.tryParse(value);
    return null;
  }

  String _itemName(Map<String, dynamic> item) {
    return '${item['name'] ?? item['hospital_name'] ?? 'Unknown hospital'}';
  }

  String _itemAddress(Map<String, dynamic> item) {
    return '${item['address'] ?? item['vicinity'] ?? 'Address unavailable'}';
  }

  String? _itemPhone(Map<String, dynamic> item) {
    final value = item['phone'] ?? item['phone_number'] ?? item['contact'];
    final phone = value == null ? '' : '$value'.trim();
    return phone.isEmpty ? null : phone;
  }

  String? _itemWebsite(Map<String, dynamic> item) {
    final value = item['website'] ?? item['site'] ?? item['url'];
    final website = value == null ? '' : '$value'.trim();
    if (website.isEmpty) return null;
    if (website.startsWith('http://') || website.startsWith('https://')) {
      return website;
    }
    return 'https://$website';
  }

  double? _itemScore(Map<String, dynamic> item) {
    final value = item['score'];
    if (value is num) return value.toDouble();
    if (value is String) return double.tryParse(value);
    return null;
  }

  bool _isCriticalEmergencyType() {
    final et = _selectedEmergency.trim().toLowerCase();
    return et == 'emergency / trauma' ||
        et == 'cardiology (heart)' ||
        et == 'neurology (brain / stroke)' ||
        et == 'orthopedics (bones / fracture)';
  }

  String? _friendlyExplainLabel(String label, {required bool positive}) {
    final low = label.trim().toLowerCase();
    if (low.isEmpty) return null;

    const hidden = <String>{
      'baseline score',
      'source',
      'data confidence',
      'candidate source',
      'gender',
      'age',
    };
    if (hidden.contains(low)) return null;

    if (low == 'distance') {
      return positive ? 'Shorter Distance' : 'Longer Distance';
    }
    if (low == 'rating') {
      return positive ? 'High Average Rating' : 'Low Average Rating';
    }
    if (low == 'rating count') {
      return positive ? 'High Review Volume' : 'Low Review Volume';
    }
    if (low == 'need match') {
      return null;
    }
    if (low == 'emergency type') {
      return null;
    }
    if (low == 'critical emergency type') {
      return null;
    }

    const map = <String, String>{
      'offline mode': 'Offline results mode',
      'penalty': 'Lower confidence signals',
      'asthma/copd profile': 'Matches your health profile',
      'diabetes profile': 'Matches your health profile',
      'heart-disease profile': 'Matches your health profile',
      'stroke-history profile': 'Matches your health profile',
      'epilepsy profile': 'Matches your health profile',
      'pregnancy profile': 'Matches your health profile',
    };

    final mapped = map[low] ?? label;
    if (!positive && low == 'offline mode') {
      return 'Offline mode may reduce result freshness';
    }
    return mapped;
  }

  List<String> _itemExplainPosLabels(Map<String, dynamic> item) {
    final explain = item['explain'];
    if (explain is! Map) return const <String>[];
    final pos = explain['top_positive'];
    if (pos is! List) return const <String>[];

    const blocked = <String>{'baseline score', 'source', 'data confidence'};
    final ratingCountVal = _asDouble(item['user_rating_count']) ?? 0.0;
    final ratingVal = _asDouble(item['rating']) ?? 0.0;
    final distanceKmVal = _asDouble(item['distance_km']) ?? 0.0;
    final hideRatingCount = ratingCountVal < 500.0;
    // Absolute threshold rules for positive cards
    final forceHighRating = ratingVal > 0.0 && ratingVal > 4.4;
    final forceShortDistance = distanceKmVal > 0.0 && distanceKmVal < 9.0;
    final forceReviewVolumeChip = ratingCountVal >= 1000.0;

    final out = <String>[];
    for (final p in pos) {
      if (p is! Map) continue;
      final label = '${p['label'] ?? p['feature'] ?? ''}'.trim();
      final low = label.toLowerCase();
      if (blocked.contains(low)) continue;
      if (hideRatingCount && low == 'rating count') continue;
      // Only show "Shorter Distance" when distance is genuinely short (< 9 km)
      if (low == 'distance' && distanceKmVal >= 9.0) continue;
      // Only show "High Average Rating" when rating is genuinely high (> 4.4)
      if (low == 'rating' && ratingVal <= 4.4) continue;
      final friendly = _friendlyExplainLabel(label, positive: true);
      if (friendly != null && friendly.isNotEmpty && !out.contains(friendly)) {
        out.add(friendly);
      }
      if (out.length >= 3) break;
    }

    if (forceHighRating && !out.contains('High Average Rating')) {
      if (out.length >= 3) out.removeLast();
      out.add('High Average Rating');
    }

    if (forceShortDistance && !out.contains('Shorter Distance')) {
      if (out.length >= 3) out.removeLast();
      out.add('Shorter Distance');
    }

    final hasReviewVolume = out.any((label) => label.toLowerCase().contains('review volume'));
    if (forceReviewVolumeChip && !hasReviewVolume) {
      if (out.length >= 3) out.removeLast();
      out.add('High Review Volume');
    }

    return out;
  }

  List<String> _itemExplainNegLabels(Map<String, dynamic> item, {int maxItems = 1}) {
    final explain = item['explain'];
    if (explain is! Map) return const <String>[];
    final neg = explain['top_negative'];

    const blocked = <String>{'baseline score', 'source', 'data confidence'};
    final ratingCountVal = _asDouble(item['user_rating_count']) ?? 0.0;
    final ratingVal = _asDouble(item['rating']) ?? 0.0;
    final distanceKmVal = _asDouble(item['distance_km']) ?? 0.0;
    final out = <String>[];
    if (neg is List) {
      for (final n in neg) {
        if (n is! Map) continue;
        final label = '${n['label'] ?? n['feature'] ?? ''}'.trim();
        final low = label.toLowerCase();
        if (label.isEmpty || blocked.contains(low)) continue;
        // Don't show "Low Review Volume" for hospitals that actually have many reviews
        if (low == 'rating count' && ratingCountVal >= 200.0) continue;
        // Only show "Low Average Rating" when rating is genuinely low (< 3.8)
        if (low == 'rating' && (ratingVal == 0.0 || ratingVal >= 3.8)) continue;
        // Only show "Longer Distance" when distance is genuinely far (> 15 km)
        if (low == 'distance' && distanceKmVal <= 15.0) continue;
        final friendly = _friendlyExplainLabel(label, positive: false);
        if (friendly == null || friendly.isEmpty || out.contains(friendly)) continue;
        out.add(friendly);
        if (out.length >= maxItems) break;
      }
    }
    // Absolute threshold rules: always show these when conditions are met
    if (ratingVal > 0.0 && ratingVal < 3.8 && !out.contains('Low Average Rating')) {
      out.add('Low Average Rating');
    }
    if (distanceKmVal > 15.0 && !out.contains('Longer Distance')) {
      out.add('Longer Distance');
    }
    return out;
  }

  Future<void> _openWebsite(String url) async {
    final uri = Uri.tryParse(url);
    if (uri == null) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Invalid website URL')),
      );
      return;
    }
    final launched = await launchUrl(uri, mode: LaunchMode.externalApplication);
    if (!launched && mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Unable to open website')),
      );
    }
  }

  Future<void> _openDirections(Map<String, dynamic> item) async {
    final destLat = _asDouble(item['lat']);
    final destLon = _asDouble(item['lon']);
    if (destLat == null || destLon == null) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Destination coordinates unavailable.')),
      );
      return;
    }

    final originLat = _lat;
    final originLon = _lon;
    final placeId = '${item['place_id'] ?? ''}'.trim();

    Uri uri;
    if (originLat != null && originLon != null) {
      final originParam = (_lastSearchQuery != null) ? _selectedPlaceLabel : '$originLat,$originLon';
      final params = <String, String>{
        'api': '1',
        'origin': originParam,
        'destination': '$destLat,$destLon',
        'travelmode': 'driving',
      };
      if (placeId.isNotEmpty) {
        params['destination_place_id'] = placeId;
      }
      uri = Uri.https('www.google.com', '/maps/dir/', params);
    } else {
      uri = Uri.https('www.google.com', '/maps/search/', {
        'api': '1',
        'query': '$destLat,$destLon',
      });
    }

    final launched = await launchUrl(uri, mode: LaunchMode.externalApplication);
    if (!launched && mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Unable to open route in Google Maps')),
      );
    }
  }

  Future<void> _copyPhone(String phone) async {
    await Clipboard.setData(ClipboardData(text: phone));
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Phone copied: $phone')),
    );
  }

  Future<void> _openFeedbackDialog(Map<String, dynamic> item, int thumbs) async {
    final runId = _currentRunId;
    final hospitalId = _asInt(item['hospital_id']);
    final runCandidateId = _asInt(item['run_candidate_id']);

    if (runId == null || hospitalId == null) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Feedback unavailable for this recommendation.')),
      );
      return;
    }

    final reasons = thumbs == 1 ? _positiveReasons : _negativeReasons;
    String? selectedReason;
    bool submitting = false;

    await showDialog<void>(
      context: context,
      builder: (dialogContext) {
        return StatefulBuilder(
          builder: (context, setDialogState) {
            return AlertDialog(
              title: const Text('Feedback'),
              content: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    _itemName(item),
                    style: const TextStyle(fontSize: 13, color: Color(0xFF4B5563)),
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                  ),
                  const SizedBox(height: 12),
                  DropdownButtonFormField<String>(
                    initialValue: selectedReason,
                    items: reasons
                        .map(
                          (reason) => DropdownMenuItem<String>(
                            value: reason['code'],
                            child: Text(reason['label'] ?? ''),
                          ),
                        )
                        .toList(),
                    onChanged: submitting
                        ? null
                        : (value) {
                            setDialogState(() => selectedReason = value);
                          },
                    decoration: const InputDecoration(
                      labelText: 'Reason',
                      border: OutlineInputBorder(),
                      isDense: true,
                    ),
                  ),
                ],
              ),
              actions: [
                TextButton(
                  onPressed: submitting ? null : () => Navigator.of(dialogContext).pop(),
                  child: const Text('Cancel'),
                ),
                ElevatedButton(
                  onPressed: (submitting || selectedReason == null)
                      ? null
                      : () async {
                          setDialogState(() => submitting = true);
                          try {
                            final token = context.read<AuthProvider>().token;
                            await _repo.submitFeedback(
                              runId: runId,
                              hospitalId: hospitalId,
                              runCandidateId: runCandidateId,
                              thumbs: thumbs,
                              reasonCode: selectedReason!,
                              token: token,
                            );
                            if (!mounted) return;
                            Navigator.of(dialogContext).pop();
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(content: Text('Feedback saved.')),
                            );
                          } catch (e) {
                            if (!mounted) return;
                            setDialogState(() => submitting = false);
                            ScaffoldMessenger.of(context).showSnackBar(
                              SnackBar(content: Text('Failed to save feedback: $e')),
                            );
                          }
                        },
                  child: Text(submitting ? 'Saving...' : 'Submit'),
                ),
              ],
            );
          },
        );
      },
    );
  }

  void _focusMap(double userLat, double userLon, List<Map<String, dynamic>> items) {
    if (items.isEmpty) {
      _mapController.move(LatLng(userLat, userLon), 13);
      return;
    }
    final first = items.first;
    final firstLat = _asDouble(first['lat']) ?? userLat;
    final firstLon = _asDouble(first['lon']) ?? userLon;
    final midLat = (firstLat + userLat) / 2;
    final midLon = (firstLon + userLon) / 2;
    _mapController.move(LatLng(midLat, midLon), 12.5);
  }

  void _setSheetSize(double nextSize) {
    final clamped = nextSize.clamp(_sheetMinSize, _sheetMaxSize);
    if ((clamped - _sheetSize).abs() < 0.0001) return;
    if (_sheetController.isAttached) {
      _sheetController.jumpTo(clamped);
    }
    if (mounted) {
      setState(() => _sheetSize = clamped);
    }
  }

  void _onSheetHandleDragUpdate(DragUpdateDetails details) {
    final height = MediaQuery.of(context).size.height;
    if (height <= 0) return;
    final delta = (details.primaryDelta ?? 0) / height;
    _setSheetSize(_sheetSize - delta);
  }

  List<Marker> _buildMarkers() {
    final markers = <Marker>[];

    final userLat = _lat ?? 10.0;
    final userLon = _lon ?? 76.0;
    markers.add(
      Marker(
        width: 34,
        height: 34,
        point: LatLng(userLat, userLon),
        child: const Icon(Icons.my_location, color: Colors.blue, size: 30),
      ),
    );

    for (var i = 0; i < _items.length; i++) {
      final item = _items[i];
      final lat = _asDouble(item['lat']);
      final lon = _asDouble(item['lon']);
      if (lat == null || lon == null) continue;

      markers.add(
        Marker(
          width: 38,
          height: 38,
          point: LatLng(lat, lon),
          child: GestureDetector(
            onTap: () => _mapController.move(LatLng(lat, lon), 15),
            child: CircleAvatar(
              radius: 17,
              backgroundColor: Colors.red,
              child: Text(
                '${i + 1}',
                style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
              ),
            ),
          ),
        ),
      );
    }
    return markers;
  }

  Widget _buildTopSearchControls() {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: const Color(0xFFFECACA)),
        boxShadow: const [
          BoxShadow(color: Color(0x1AF87171), blurRadius: 14, offset: Offset(0, 6)),
        ],
      ),
      child: Padding(
        padding: const EdgeInsets.all(10),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                Expanded(
                  child: TextField(
                    focusNode: _searchFocusNode,
                    controller: _placeController,
                    textInputAction: TextInputAction.search,
                    onTap: () => setState(() => _searchExpanded = true),
                    onSubmitted: (_) => _searchAndRecommend(),
                    decoration: InputDecoration(
                      hintText: 'Search location...',
                      prefixIcon: const Icon(Icons.search),
                      isDense: true,
                      filled: true,
                      fillColor: const Color(0xFFF9FAFB),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(14),
                        borderSide: BorderSide.none,
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                SizedBox(
                  width: 52,
                  height: 52,
                  child: ElevatedButton(
                    onPressed: _loading ? null : _sosRecommendFromCurrentLocation,
                    style: ElevatedButton.styleFrom(
                      shape: const CircleBorder(),
                      padding: EdgeInsets.zero,
                      backgroundColor: Colors.red,
                      foregroundColor: Colors.white,
                      elevation: 1,
                    ),
                    child: const Text(
                      'SOS',
                      style: TextStyle(fontSize: 12, fontWeight: FontWeight.w700),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                IconButton.filled(
                  onPressed: _openProfile,
                  icon: const Icon(Icons.person_rounded),
                  tooltip: 'Profile',
                  style: IconButton.styleFrom(
                    backgroundColor: const Color(0xFFDC2626),
                    foregroundColor: Colors.white,
                  ),
                ),
              ],
            ),
            AnimatedCrossFade(
              crossFadeState: _searchExpanded ? CrossFadeState.showSecond : CrossFadeState.showFirst,
              duration: const Duration(milliseconds: 180),
              firstChild: const SizedBox.shrink(),
              secondChild: Padding(
                padding: const EdgeInsets.only(top: 10),
                child: Column(
                  children: [
                    if (_useCategorySelector)
                      DropdownButtonFormField<String>(
                        initialValue: _selectedEmergency,
                        items: _emergencyOptions
                            .map((e) => DropdownMenuItem<String>(value: e, child: Text(e)))
                            .toList(),
                        onChanged: (v) {
                          if (v != null) setState(() => _selectedEmergency = v);
                        },
                        decoration: InputDecoration(
                          labelText: 'Category',
                          filled: true,
                          fillColor: const Color(0xFFF9FAFB),
                          border: OutlineInputBorder(borderRadius: BorderRadius.circular(14)),
                          isDense: true,
                        ),
                      )
                    else
                      TextField(
                        controller: _symptomController,
                        decoration: InputDecoration(
                          labelText: 'Symptom',
                          hintText: 'e.g. chest pain, blurred vision',
                          filled: true,
                          fillColor: const Color(0xFFF9FAFB),
                          border: OutlineInputBorder(borderRadius: BorderRadius.circular(14)),
                          isDense: true,
                        ),
                      ),
                    const SizedBox(height: 8),
                    Container(
                      decoration: BoxDecoration(
                        color: const Color(0xFFF9FAFB),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: SwitchListTile(
                        value: _offlineMode,
                        onChanged: (v) => setState(() => _offlineMode = v),
                        title: const Text('Offline mode'),
                        contentPadding: const EdgeInsets.symmetric(horizontal: 8),
                      ),
                    ),
                    const SizedBox(height: 8),
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton.icon(
                        onPressed: _loading ? null : _searchAndRecommend,
                        icon: const Icon(Icons.search_rounded, size: 18),
                        label: Text(_loading ? 'Searching...' : 'Search'),
                      ),
                    ),
                    if (_status.isNotEmpty) ...[
                      const SizedBox(height: 6),
                      Text(
                        _status,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(color: Color(0xFF6B7280), fontSize: 12),
                      ),
                    ]
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildRecommendationSheet() {
    return NotificationListener<DraggableScrollableNotification>(
      onNotification: (notification) {
        _sheetSize = notification.extent;
        return false;
      },
      child: DraggableScrollableSheet(
        controller: _sheetController,
        minChildSize: _sheetMinSize,
        maxChildSize: _sheetMaxSize,
        initialChildSize: _sheetInitialSize,
        builder: (context, scrollController) {
          return Container(
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.surface,
              borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
              boxShadow: const [
                BoxShadow(color: Color(0x24000000), blurRadius: 14, offset: Offset(0, -4)),
              ],
            ),
            child: Column(
              children: [
                const SizedBox(height: 12),
                GestureDetector(
                  behavior: HitTestBehavior.opaque,
                  onVerticalDragUpdate: _onSheetHandleDragUpdate,
                  child: SizedBox(
                    height: 24,
                    child: Center(
                      child: Container(
                        height: 6,
                        width: 52,
                        decoration: BoxDecoration(
                          color: const Color(0xFFD1D5DB),
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                    ),
                  ),
                ),
              const SizedBox(height: 12),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 14),
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                  decoration: BoxDecoration(
                    color: const Color(0xFFF9FAFB),
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(color: const Color(0xFFF3F4F6)),
                  ),
                  child: Row(
                    children: [
                      Container(
                        width: 32,
                        height: 32,
                        decoration: BoxDecoration(
                          color: const Color(0xFFFEE2E2),
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: const Icon(Icons.local_hospital, color: Color(0xFFDC2626), size: 18),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Text(
                          'Most suitable hospitals near $_selectedPlaceLabel',
                          style: const TextStyle(fontWeight: FontWeight.w700),
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 8),
              Expanded(
                child: _items.isEmpty
                    ? ListView(
                        controller: scrollController,
                        children: const [
                          SizedBox(height: 30),
                          Center(child: Text('No recommendations yet. Search a place and fetch.')),
                        ],
                      )
                    : ListView.separated(
                        controller: scrollController,
                        padding: const EdgeInsets.fromLTRB(12, 0, 12, 14),
                        itemCount: _items.length,
                      separatorBuilder: (_, index) => const SizedBox(height: 8),
                        itemBuilder: (_, index) {
                          final it = _items[index];
                          final lat = _asDouble(it['lat']);
                          final lon = _asDouble(it['lon']);
                          final distance = _asDouble(it['distance_km']);
                          final phone = _itemPhone(it);
                          final website = _itemWebsite(it);
                          final score = _itemScore(it);
                          final explainPos = _itemExplainPosLabels(it);
                          final allowTwoNegatives = index >= (_items.length - 3);
                          final explainNeg = _itemExplainNegLabels(
                            it,
                            maxItems: allowTwoNegatives ? 2 : 1,
                          );
                          return Material(
                            color: Colors.white,
                            borderRadius: BorderRadius.circular(14),
                            child: InkWell(
                              borderRadius: BorderRadius.circular(14),
                              onTap: (lat == null || lon == null)
                                  ? null
                                  : () => _mapController.move(LatLng(lat, lon), 15),
                              child: Padding(
                                padding: const EdgeInsets.all(10),
                                child: Row(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    CircleAvatar(
                                      backgroundColor: Colors.red,
                                      child: Text(
                                        '${index + 1}',
                                        style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                                      ),
                                    ),
                                    const SizedBox(width: 10),
                                    Expanded(
                                      child: Column(
                                        crossAxisAlignment: CrossAxisAlignment.start,
                                        children: [
                                          Text(
                                            _itemName(it),
                                            maxLines: 1,
                                            overflow: TextOverflow.ellipsis,
                                            style: const TextStyle(fontWeight: FontWeight.w700),
                                          ),
                                          const SizedBox(height: 2),
                                          Text(
                                            '${_itemAddress(it)}${distance != null ? ' • ${distance.toStringAsFixed(2)} km' : ''}',
                                            maxLines: 2,
                                            overflow: TextOverflow.ellipsis,
                                          ),
                                          if (distance != null) ...[
                                            const SizedBox(height: 4),
                                            Text(
                                              '${distance.toStringAsFixed(2)} km',
                                              style: const TextStyle(
                                                fontSize: 13,
                                                fontWeight: FontWeight.w700,
                                                color: Color(0xFFDC2626),
                                              ),
                                            ),
                                          ],
                                          if (score != null) ...[
                                            const SizedBox(height: 4),
                                            Text(
                                              'Score: ${score.toStringAsFixed(2)}',
                                              style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w700, color: Color(0xFF374151)),
                                            ),
                                          ],
                                          const SizedBox(height: 6),
                                          const Text(
                                            'Why recommended',
                                            style: TextStyle(fontSize: 12, fontWeight: FontWeight.w700, color: Color(0xFF4B5563)),
                                          ),
                                          if (_isCriticalEmergencyType()) ...[
                                            const SizedBox(height: 2),
                                            const Text(
                                              'For critical emergencies, closer and better-matched hospitals are prioritized.',
                                              style: TextStyle(fontSize: 11, color: Color(0xFF6B7280)),
                                            ),
                                          ],
                                          const SizedBox(height: 4),
                                          if (explainPos.isNotEmpty)
                                            Wrap(
                                              spacing: 6,
                                              runSpacing: 6,
                                              children: explainPos
                                                  .map(
                                                    (label) => Container(
                                                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                                                      decoration: BoxDecoration(
                                                        color: const Color(0xFFECFDF3),
                                                        borderRadius: BorderRadius.circular(999),
                                                      ),
                                                      child: Text(
                                                        '+ $label',
                                                        style: const TextStyle(
                                                          fontSize: 11,
                                                          color: Color(0xFF047857),
                                                          fontWeight: FontWeight.w600,
                                                        ),
                                                      ),
                                                    ),
                                                  )
                                                  .toList(),
                                            )
                                          else
                                            const Text(
                                              'Explanation unavailable for this hospital.',
                                              style: TextStyle(fontSize: 11, color: Color(0xFF6B7280)),
                                            ),
                                          if (explainNeg.isNotEmpty) ...[
                                            const SizedBox(height: 4),
                                            Wrap(
                                              spacing: 6,
                                              runSpacing: 6,
                                              children: explainNeg
                                                  .map(
                                                    (label) => Container(
                                                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                                                      decoration: BoxDecoration(
                                                        color: const Color(0xFFFEE2E2),
                                                        borderRadius: BorderRadius.circular(999),
                                                      ),
                                                      child: Text(
                                                        '- $label',
                                                        style: const TextStyle(
                                                          fontSize: 11,
                                                          color: Color(0xFFB91C1C),
                                                          fontWeight: FontWeight.w600,
                                                        ),
                                                      ),
                                                    ),
                                                  )
                                                  .toList(),
                                            ),
                                          ],
                                          if (phone != null) ...[
                                            const SizedBox(height: 4),
                                            GestureDetector(
                                              onLongPress: () => _copyPhone(phone),
                                              child: Text(
                                                phone,
                                                style: const TextStyle(
                                                  color: Colors.black,
                                                  fontWeight: FontWeight.w700,
                                                ),
                                              ),
                                            ),
                                          ],
                                          const SizedBox(height: 4),
                                          Wrap(
                                            spacing: 6,
                                            runSpacing: 4,
                                            crossAxisAlignment: WrapCrossAlignment.center,
                                            children: [
                                              if (website != null)
                                                OutlinedButton.icon(
                                                  onPressed: () => _openWebsite(website),
                                                  icon: const Icon(Icons.public, size: 16),
                                                  label: const Text('Website'),
                                                  style: OutlinedButton.styleFrom(
                                                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 2),
                                                    minimumSize: const Size(0, 30),
                                                    tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                                                    visualDensity: VisualDensity.compact,
                                                  ),
                                                ),
                                              OutlinedButton.icon(
                                                onPressed: () => _openDirections(it),
                                                icon: const Icon(Icons.route, size: 16),
                                                label: const Text('Route'),
                                                style: OutlinedButton.styleFrom(
                                                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 2),
                                                  minimumSize: const Size(0, 30),
                                                  tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                                                  visualDensity: VisualDensity.compact,
                                                ),
                                              ),
                                              IconButton(
                                                onPressed: () => _openFeedbackDialog(it, 1),
                                                icon: const Icon(Icons.thumb_up_alt_rounded),
                                                color: Colors.green,
                                                tooltip: 'Thumbs up',
                                                visualDensity: VisualDensity.compact,
                                              ),
                                              IconButton(
                                                onPressed: () => _openFeedbackDialog(it, -1),
                                                icon: const Icon(Icons.thumb_down_alt_rounded),
                                                color: Colors.grey,
                                                tooltip: 'Thumbs down',
                                                visualDensity: VisualDensity.compact,
                                              ),
                                            ],
                                          ),
                                        ],
                                      ),
                                    ),
                                    const Icon(Icons.chevron_right),
                                  ],
                                ),
                              ),
                            ),
                          );
                        },
                      ),
              ),
              ],
            ),
          );
        },
      ),
    );
  }

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

  Widget _buildGuestView(AuthProvider auth) {
    return SafeArea(
      child: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Color(0xFFFFF5F5), Colors.white],
          ),
        ),
        child: Stack(
          children: [
            Positioned(
              top: 8,
              right: 8,
              child: IconButton(
                onPressed: () => Navigator.of(context).pushNamed('/settings'),
                icon: const Icon(Icons.settings, size: 20),
                tooltip: 'Server settings',
              ),
            ),
            Center(
              child: Padding(
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
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          Center(child: _buildBrandLogo(size: 72)),
                          const SizedBox(height: 14),
                          const Center(
                            child: Text(
                              'SwiftRelief',
                              style: TextStyle(fontSize: 28, fontWeight: FontWeight.w800, color: Color(0xFFDC2626)),
                            ),
                          ),
                          const SizedBox(height: 8),
                          const Center(
                            child: Text(
                              'Find the right emergency care, faster.',
                              style: TextStyle(color: Color(0xFF6B7280), fontSize: 14),
                            ),
                          ),
                          const SizedBox(height: 24),
                          ElevatedButton.icon(
                            onPressed: () => Navigator.of(context).pushNamed('/login'),
                            icon: const Icon(Icons.login_rounded, size: 18),
                            label: const Text('Login'),
                          ),
                          const SizedBox(height: 10),
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
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();
    return Scaffold(
      body: auth.user == null
          ? _buildGuestView(auth)
          : Stack(
              children: [
                GestureDetector(
                  onTap: () {
                    FocusScope.of(context).unfocus();
                    if (_searchExpanded) {
                      setState(() => _searchExpanded = false);
                    }
                  },
                  child: FlutterMap(
                    mapController: _mapController,
                    options: MapOptions(
                      initialCenter: LatLng(_lat ?? 10.0, _lon ?? 76.0),
                      initialZoom: 10,
                    ),
                    children: [
                      TileLayer(
                        urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                        userAgentPackageName: 'com.swiftrelief.app',
                        tileProvider: OsmCachedTileProvider(),
                      ),
                      MarkerLayer(markers: _buildMarkers()),
                    ],
                  ),
                ),
                SafeArea(
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: Column(
                      children: [
                        _buildTopSearchControls(),
                      ],
                    ),
                  ),
                ),
                _buildRecommendationSheet(),
              ],
            ),
    );
  }
}

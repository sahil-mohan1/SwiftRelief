import 'dart:async';
import 'dart:io';

import 'package:cached_network_image/cached_network_image.dart';
import 'package:flutter_cache_manager/flutter_cache_manager.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:flutter/widgets.dart';

import '../repositories/cache_repository.dart';

final CacheManager _osmTileCacheManager = CacheManager(
  Config(
    'osm_tile_cache',
    stalePeriod: const Duration(days: 30),
    maxNrOfCacheObjects: 20000,
  ),
);

class OsmCachedTileProvider extends TileProvider {
  OsmCachedTileProvider();

  static const String _tilePathPrefix = 'osm_tile_path_v1::';

  String _tilePathKey(String url) => '$_tilePathPrefix$url';

  void _primeTilePath(String url) {
    unawaited(() async {
      try {
        final file = await _osmTileCacheManager.getSingleFile(
          url,
          headers: const {'User-Agent': 'com.swiftrelief.app'},
        );
        await CacheRepository.save(_tilePathKey(url), file.path);
      } catch (_) {}
    }());
  }

  @override
  ImageProvider getImage(TileCoordinates coordinates, TileLayer options) {
    final url = getTileUrl(coordinates, options);

    final cachedPath = CacheRepository.read(_tilePathKey(url));
    if (cachedPath is String && cachedPath.isNotEmpty) {
      final file = File(cachedPath);
      if (file.existsSync()) {
        return FileImage(file);
      }
    }

    _primeTilePath(url);
    return CachedNetworkImageProvider(
      url,
      cacheManager: _osmTileCacheManager,
      headers: const {'User-Agent': 'com.swiftrelief.app'},
    );
  }
}

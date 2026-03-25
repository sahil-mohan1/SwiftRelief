import 'package:hive_flutter/hive_flutter.dart';

class CacheRepository {
  static const String boxName = 'swiftrelief_cache';
  static const String useCategorySelectorKey = 'use_category_selector';

  static Future<void> init() async {
    await Hive.initFlutter();
    await Hive.openBox(boxName);
  }

  static Box box() => Hive.box(boxName);

  static Future<void> save(String key, dynamic value) async {
    await box().put(key, value);
  }

  static dynamic read(String key) => box().get(key);

  static Future<void> clear() async => await box().clear();
}

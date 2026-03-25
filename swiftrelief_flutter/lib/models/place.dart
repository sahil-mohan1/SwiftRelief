class Place {
  final String name;
  final double lat;
  final double lon;

  Place({required this.name, required this.lat, required this.lon});

  factory Place.fromJson(Map<String, dynamic> j) => Place(
        name: j['name'] ?? j['display_name'] ?? '',
        lat: (j['lat'] is String) ? double.parse(j['lat']) : (j['lat'] ?? 0.0),
        lon: (j['lon'] is String) ? double.parse(j['lon']) : (j['lon'] ?? 0.0),
      );
  Map<String, dynamic> toJson() => {'name': name, 'lat': lat, 'lon': lon};
}

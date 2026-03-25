class Recommendation {
  final int id;
  final String name;
  final String address;

  Recommendation({required this.id, required this.name, required this.address});

  factory Recommendation.fromJson(Map<String, dynamic> j) => Recommendation(
        id: j['id'] ?? 0,
        name: j['name'] ?? '',
        address: j['address'] ?? '',
      );
  Map<String, dynamic> toJson() => {'id': id, 'name': name, 'address': address};
}

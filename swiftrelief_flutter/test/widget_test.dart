// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

import 'package:swiftrelief_flutter/main.dart';
import 'package:swiftrelief_flutter/providers/auth_provider.dart';
import 'package:swiftrelief_flutter/services/api_client.dart';

void main() {
  testWidgets('App renders guest branding', (WidgetTester tester) async {
    final api = ApiClient();
    await tester.pumpWidget(
      ChangeNotifierProvider(
        create: (_) => AuthProvider(api),
        child: MyApp(api: api),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('SwiftRelief'), findsOneWidget);
  });
}

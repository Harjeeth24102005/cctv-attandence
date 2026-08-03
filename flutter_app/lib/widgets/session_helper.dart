import 'package:flutter/material.dart';

import '../services/api_service.dart';
import '../screens/login_screen.dart';

/// Every screen's error handling funnels through here: an
/// [UnauthorizedException] (expired/invalid token) always means "log out
/// and go back to the login screen", while any other error is just shown
/// inline by the caller. Centralizing this means that behavior can't
/// silently diverge screen to screen.
Future<void> handleApiError(BuildContext context, Object error) async {
  if (error is UnauthorizedException) {
    await ApiService.instance.logout();
    if (!context.mounted) return;
    Navigator.of(context, rootNavigator: true).pushAndRemoveUntil(
      MaterialPageRoute(builder: (_) => const LoginScreen()),
      (route) => false,
    );
  }
}

String friendlyError(Object error) {
  return error.toString().replaceFirst('Exception: ', '');
}

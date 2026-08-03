import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

import '../models/models.dart';

/// Thrown when a call gets a 401 back - the token is missing/expired/invalid.
/// Screens catch this specifically to bounce the user back to the login
/// screen instead of showing a generic error.
class UnauthorizedException implements Exception {
  final String message;
  UnauthorizedException(this.message);
}

/// Thin wrapper around the /api/v1/* endpoints exposed by api.py.
/// Holds the server base URL and JWT in SharedPreferences so the app stays
/// logged in across restarts, and centralizes error handling so every
/// screen doesn't have to repeat status-code checks.
class ApiService {
  static const _kBaseUrlKey = 'base_url';
  static const _kTokenKey = 'jwt_token';
  static const _kUsernameKey = 'username';

  String? _baseUrl;
  String? _token;

  static final ApiService instance = ApiService._internal();
  ApiService._internal();

  Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    _baseUrl = prefs.getString(_kBaseUrlKey);
    _token = prefs.getString(_kTokenKey);
  }

  bool get isLoggedIn => _baseUrl != null && _token != null;
  String? get baseUrl => _baseUrl;
  String? get token => _token;

  Uri _uri(String path, [Map<String, String>? query]) {
    if (_baseUrl == null) {
      throw StateError('Server URL not set. Call login() first.');
    }
    final cleanBase = _baseUrl!.endsWith('/')
        ? _baseUrl!.substring(0, _baseUrl!.length - 1)
        : _baseUrl!;
    return Uri.parse('$cleanBase$path').replace(queryParameters: query);
  }

  Map<String, String> get _authHeaders => {
        'Content-Type': 'application/json',
        if (_token != null) 'Authorization': 'Bearer $_token',
      };

  /// Builds a live snapshot/stream URL with the token as a query param,
  /// since Image widgets can't attach an Authorization header themselves.
  String liveSnapshotUrl() {
    final base = _uri('/api/v1/live/snapshot', {
      'token': _token ?? '',
      // Cache-busting timestamp so Image widgets actually re-fetch.
      't': DateTime.now().millisecondsSinceEpoch.toString(),
    });
    return base.toString();
  }

  // ---------------------------------------------------------------------
  // AUTH
  // ---------------------------------------------------------------------
  Future<void> login({
    required String serverUrl,
    required String username,
    required String password,
  }) async {
    _baseUrl = serverUrl.trim();
    final resp = await http.post(
      _uri('/api/v1/auth/login'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'username': username, 'password': password}),
    );
    final body = _decodeOrThrow(resp);
    _token = body['token'] as String;

    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_kBaseUrlKey, _baseUrl!);
    await prefs.setString(_kTokenKey, _token!);
    await prefs.setString(_kUsernameKey, username);
  }

  Future<void> logout() async {
    _token = null;
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_kTokenKey);
  }

  // ---------------------------------------------------------------------
  // DASHBOARD
  // ---------------------------------------------------------------------
  Future<DashboardSummary> getDashboard() async {
    final resp = await http.get(_uri('/api/v1/dashboard'), headers: _authHeaders);
    final body = _decodeOrThrow(resp);
    return DashboardSummary.fromJson(body);
  }

  // ---------------------------------------------------------------------
  // ATTENDANCE / MOVEMENTS
  // ---------------------------------------------------------------------
  Future<List<AttendanceRecord>> getAttendance({String? date, String? employeeId}) async {
    final resp = await http.get(
      _uri('/api/v1/attendance', {
        if (date != null) 'date': date,
        if (employeeId != null) 'employee_id': employeeId,
      }),
      headers: _authHeaders,
    );
    final body = _decodeOrThrow(resp);
    return (body['records'] as List)
        .map((r) => AttendanceRecord.fromJson(r as Map<String, dynamic>))
        .toList();
  }

  Future<List<MovementRecord>> getMovements({String? date, String? employeeId}) async {
    final resp = await http.get(
      _uri('/api/v1/movements', {
        if (date != null) 'date': date,
        if (employeeId != null) 'employee_id': employeeId,
      }),
      headers: _authHeaders,
    );
    final body = _decodeOrThrow(resp);
    return (body['records'] as List)
        .map((r) => MovementRecord.fromJson(r as Map<String, dynamic>))
        .toList();
  }

  Future<List<LiveMatch>> getLiveMatches() async {
    final resp = await http.get(_uri('/api/v1/live/matches'), headers: _authHeaders);
    final body = _decodeOrThrow(resp);
    return (body['matches'] as List)
        .map((m) => LiveMatch.fromJson(m as Map<String, dynamic>))
        .toList();
  }

  // ---------------------------------------------------------------------
  // EMPLOYEES
  // ---------------------------------------------------------------------
  Future<List<Employee>> getEmployees() async {
    final resp = await http.get(_uri('/api/v1/employees'), headers: _authHeaders);
    final body = _decodeOrThrow(resp);
    return (body['employees'] as List)
        .map((e) => Employee.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  Future<Map<String, dynamic>> getEmployeeDetail(String personId) async {
    final resp = await http.get(_uri('/api/v1/employees/$personId'), headers: _authHeaders);
    return _decodeOrThrow(resp);
  }

  Future<void> upsertEmployee({
    required String personId,
    required String name,
    String department = '',
    String designation = '',
    String phone = '',
    String email = '',
  }) async {
    final resp = await http.post(
      _uri('/api/v1/employees'),
      headers: _authHeaders,
      body: jsonEncode({
        'person_id': personId,
        'name': name,
        'department': department,
        'designation': designation,
        'phone': phone,
        'email': email,
      }),
    );
    _decodeOrThrow(resp);
  }

  // ---------------------------------------------------------------------
  // ENROLLMENT
  // ---------------------------------------------------------------------
  Future<void> enrollDetails({
    required String personId,
    required String name,
    String department = '',
    String designation = '',
    String phone = '',
    String email = '',
  }) async {
    final resp = await http.post(
      _uri('/api/v1/enroll/details'),
      headers: _authHeaders,
      body: jsonEncode({
        'person_id': personId,
        'name': name,
        'department': department,
        'designation': designation,
        'phone': phone,
        'email': email,
      }),
    );
    _decodeOrThrow(resp);
  }

  Future<Map<String, dynamic>> enrollCapture({
    required String personId,
    required String pose,
    required String base64Jpeg,
  }) async {
    final resp = await http.post(
      _uri('/api/v1/enroll/capture'),
      headers: _authHeaders,
      body: jsonEncode({
        'person_id': personId,
        'pose': pose,
        'image': 'data:image/jpeg;base64,$base64Jpeg',
      }),
    );
    // Capture responses use ok:false for expected retry reasons (blur, pose
    // mismatch, cooldown, etc) - those aren't exceptions, the caller needs
    // the reason text to show the user, so don't throw here.
    return jsonDecode(resp.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> enrollStatus(String personId) async {
    final resp = await http.get(
      _uri('/api/v1/enroll/status', {'person_id': personId}),
      headers: _authHeaders,
    );
    return _decodeOrThrow(resp);
  }

  Future<Map<String, dynamic>> enrollFinish(String personId) async {
    final resp = await http.post(
      _uri('/api/v1/enroll/finish'),
      headers: _authHeaders,
      body: jsonEncode({'person_id': personId}),
    );
    return _decodeOrThrow(resp);
  }

  // ---------------------------------------------------------------------
  // INTERNAL
  // ---------------------------------------------------------------------
  Map<String, dynamic> _decodeOrThrow(http.Response resp) {
    Map<String, dynamic> body;
    try {
      body = jsonDecode(resp.body) as Map<String, dynamic>;
    } catch (_) {
      throw Exception('Unexpected response from server (HTTP ${resp.statusCode}).');
    }

    if (resp.statusCode == 401) {
      throw UnauthorizedException(body['reason'] ?? 'Session expired. Please log in again.');
    }
    if (resp.statusCode >= 400 || body['ok'] == false) {
      throw Exception(body['reason'] ?? 'Request failed (HTTP ${resp.statusCode}).');
    }
    return body;
  }
}

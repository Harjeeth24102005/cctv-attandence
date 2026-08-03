import 'dart:convert';
import 'package:flutter/material.dart';

import '../services/api_service.dart';
import '../widgets/session_helper.dart';

class EmployeeDetailScreen extends StatefulWidget {
  final String personId;
  const EmployeeDetailScreen({super.key, required this.personId});

  @override
  State<EmployeeDetailScreen> createState() => _EmployeeDetailScreenState();
}

class _EmployeeDetailScreenState extends State<EmployeeDetailScreen> {
  Map<String, dynamic>? _data;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final data = await ApiService.instance.getEmployeeDetail(widget.personId);
      if (!mounted) return;
      setState(() => _data = data);
    } catch (e) {
      await handleApiError(context, e);
      if (mounted) setState(() => _error = friendlyError(e));
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(widget.personId)),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? Center(child: Text(_error!))
              : _buildBody(),
    );
  }

  Widget _buildBody() {
    final employee = _data!['employee'] as Map<String, dynamic>;
    final history = (_data!['recent_attendance'] as List? ?? []);
    ImageProvider? avatar;
    final thumb = employee['thumbnail'] as String?;
    if (thumb != null && thumb.contains(',')) {
      try {
        avatar = MemoryImage(base64Decode(thumb.split(',').last));
      } catch (_) {}
    }

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Center(
          child: CircleAvatar(
            radius: 48,
            backgroundImage: avatar,
            child: avatar == null ? const Icon(Icons.person, size: 48) : null,
          ),
        ),
        const SizedBox(height: 12),
        Center(
          child: Text(employee['name'] ?? '', style: Theme.of(context).textTheme.headlineSmall),
        ),
        Center(
          child: Text(
            [employee['department'], employee['designation']]
                .where((s) => (s ?? '').toString().isNotEmpty)
                .join(' - '),
            style: Theme.of(context).textTheme.bodyMedium,
          ),
        ),
        const SizedBox(height: 24),
        Card(
          child: Column(
            children: [
              _infoRow(Icons.badge_outlined, 'Employee ID', employee['person_id']),
              _infoRow(Icons.phone_outlined, 'Phone', employee['phone']),
              _infoRow(Icons.email_outlined, 'Email', employee['email']),
              _infoRow(
                Icons.face_retouching_natural_outlined,
                'Face embeddings',
                '${employee['embeddings_count']} of ${employee['total_captured']} captured photos',
              ),
            ],
          ),
        ),
        const SizedBox(height: 24),
        Text('Attendance - last 30 days', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 8),
        if (history.isEmpty)
          const Padding(padding: EdgeInsets.all(8), child: Text('No attendance in the last 30 days.')),
        ...history.map((h) => Card(
              margin: const EdgeInsets.only(bottom: 6),
              child: ListTile(
                leading: const Icon(Icons.check_circle_outline, color: Colors.green),
                title: Text(h['date'] ?? ''),
                trailing: Text(h['time'] ?? ''),
              ),
            )),
      ],
    );
  }

  Widget _infoRow(IconData icon, String label, dynamic value) {
    final v = (value ?? '').toString();
    return ListTile(
      leading: Icon(icon),
      title: Text(label),
      subtitle: Text(v.isEmpty ? '-' : v),
    );
  }
}

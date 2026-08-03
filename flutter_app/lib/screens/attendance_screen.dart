import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../models/models.dart';
import '../services/api_service.dart';
import '../widgets/session_helper.dart';

class AttendanceScreen extends StatefulWidget {
  const AttendanceScreen({super.key});

  @override
  State<AttendanceScreen> createState() => _AttendanceScreenState();
}

class _AttendanceScreenState extends State<AttendanceScreen> {
  DateTime _date = DateTime.now();
  List<AttendanceRecord> _records = [];
  bool _loading = true;
  String? _error;

  static final _dateFmt = DateFormat('yyyy-MM-dd');
  static final _displayFmt = DateFormat('EEE, d MMM yyyy');

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final records = await ApiService.instance.getAttendance(date: _dateFmt.format(_date));
      if (!mounted) return;
      setState(() => _records = records);
    } catch (e) {
      await handleApiError(context, e);
      if (mounted) setState(() => _error = friendlyError(e));
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _pickDate() async {
    final picked = await showDatePicker(
      context: context,
      initialDate: _date,
      firstDate: DateTime(2020),
      lastDate: DateTime.now(),
    );
    if (picked != null) {
      setState(() => _date = picked);
      _load();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Attendance'),
        actions: [
          IconButton(icon: const Icon(Icons.calendar_month_outlined), onPressed: _pickDate),
        ],
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Row(
              children: [
                Text(_displayFmt.format(_date), style: Theme.of(context).textTheme.titleMedium),
                const Spacer(),
                Text('${_records.length} present', style: Theme.of(context).textTheme.bodyMedium),
              ],
            ),
          ),
          const Divider(height: 1),
          Expanded(child: _buildBody()),
        ],
      ),
    );
  }

  Widget _buildBody() {
    if (_loading) return const Center(child: CircularProgressIndicator());
    if (_error != null) return Center(child: Text(_error!));
    if (_records.isEmpty) return const Center(child: Text('No attendance recorded for this date.'));

    return RefreshIndicator(
      onRefresh: _load,
      child: ListView.builder(
        itemCount: _records.length,
        itemBuilder: (context, i) {
          final r = _records[i];
          return ListTile(
            leading: CircleAvatar(child: Text(r.name.isNotEmpty ? r.name[0].toUpperCase() : '?')),
            title: Text(r.name),
            subtitle: Text([r.department, r.designation].where((s) => s.isNotEmpty).join(' - ')),
            trailing: Text(r.time),
          );
        },
      ),
    );
  }
}

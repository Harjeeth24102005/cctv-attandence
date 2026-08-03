import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../models/models.dart';
import '../services/api_service.dart';
import '../widgets/session_helper.dart';

class MovementsScreen extends StatefulWidget {
  const MovementsScreen({super.key});

  @override
  State<MovementsScreen> createState() => _MovementsScreenState();
}

class _MovementsScreenState extends State<MovementsScreen> {
  DateTime _date = DateTime.now();
  List<MovementRecord> _records = [];
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
      final records = await ApiService.instance.getMovements(date: _dateFmt.format(_date));
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
        title: const Text('Movements'),
        actions: [
          IconButton(icon: const Icon(Icons.calendar_month_outlined), onPressed: _pickDate),
        ],
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Align(
              alignment: Alignment.centerLeft,
              child: Text(_displayFmt.format(_date), style: Theme.of(context).textTheme.titleMedium),
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
    if (_records.isEmpty) return const Center(child: Text('No entry/exit events for this date.'));

    return RefreshIndicator(
      onRefresh: _load,
      child: ListView.builder(
        itemCount: _records.length,
        itemBuilder: (context, i) {
          final r = _records[i];
          final entering = r.direction == 'ENTERING';
          return ListTile(
            leading: CircleAvatar(
              backgroundColor: entering ? Colors.green.shade100 : Colors.orange.shade100,
              child: Icon(
                entering ? Icons.login : Icons.logout,
                color: entering ? Colors.green.shade800 : Colors.orange.shade800,
              ),
            ),
            title: Text(r.name),
            subtitle: Text(r.department.isNotEmpty ? r.department : r.personId),
            trailing: Text(r.time),
          );
        },
      ),
    );
  }
}

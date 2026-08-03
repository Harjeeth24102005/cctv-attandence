import 'dart:convert';
import 'package:flutter/material.dart';

import '../models/models.dart';
import '../services/api_service.dart';
import '../widgets/session_helper.dart';
import 'employee_detail_screen.dart';
import 'enroll_screen.dart';

class EmployeesScreen extends StatefulWidget {
  const EmployeesScreen({super.key});

  @override
  State<EmployeesScreen> createState() => _EmployeesScreenState();
}

class _EmployeesScreenState extends State<EmployeesScreen> {
  List<Employee> _employees = [];
  List<Employee> _filtered = [];
  bool _loading = true;
  String? _error;
  final _searchController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _load();
    _searchController.addListener(_applyFilter);
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final employees = await ApiService.instance.getEmployees();
      if (!mounted) return;
      setState(() {
        _employees = employees;
        _applyFilter();
      });
    } catch (e) {
      await handleApiError(context, e);
      if (mounted) setState(() => _error = friendlyError(e));
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _applyFilter() {
    final q = _searchController.text.trim().toLowerCase();
    setState(() {
      _filtered = q.isEmpty
          ? _employees
          : _employees
              .where((e) =>
                  e.name.toLowerCase().contains(q) ||
                  e.personId.toLowerCase().contains(q) ||
                  e.department.toLowerCase().contains(q))
              .toList();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Employees'),
        actions: [
          IconButton(
            icon: const Icon(Icons.person_add_alt_1),
            tooltip: 'Enroll employee',
            onPressed: () async {
              await Navigator.of(context).push(
                MaterialPageRoute(builder: (_) => const EnrollScreen()),
              );
              _load();
            },
          ),
        ],
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: TextField(
              controller: _searchController,
              decoration: const InputDecoration(
                hintText: 'Search by name, ID, or department',
                prefixIcon: Icon(Icons.search),
                border: OutlineInputBorder(),
                isDense: true,
              ),
            ),
          ),
          Expanded(child: _buildBody()),
        ],
      ),
    );
  }

  Widget _buildBody() {
    if (_loading) return const Center(child: CircularProgressIndicator());
    if (_error != null) return Center(child: Text(_error!));
    if (_filtered.isEmpty) return const Center(child: Text('No employees found.'));

    return RefreshIndicator(
      onRefresh: _load,
      child: ListView.builder(
        itemCount: _filtered.length,
        itemBuilder: (context, i) {
          final e = _filtered[i];
          ImageProvider? avatar;
          if (e.thumbnail != null && e.thumbnail!.contains(',')) {
            try {
              avatar = MemoryImage(base64Decode(e.thumbnail!.split(',').last));
            } catch (_) {
              avatar = null;
            }
          }
          return ListTile(
            leading: CircleAvatar(
              backgroundImage: avatar,
              child: avatar == null ? Text(e.name.isNotEmpty ? e.name[0].toUpperCase() : '?') : null,
            ),
            title: Text(e.name),
            subtitle: Text([e.department, e.designation].where((s) => s.isNotEmpty).join(' - ')),
            trailing: Icon(
              e.embeddingsCount > 0 ? Icons.check_circle : Icons.error_outline,
              color: e.embeddingsCount > 0 ? Colors.green : Colors.orange,
              size: 20,
            ),
            onTap: () => Navigator.of(context).push(
              MaterialPageRoute(builder: (_) => EmployeeDetailScreen(personId: e.personId)),
            ),
          );
        },
      ),
    );
  }
}

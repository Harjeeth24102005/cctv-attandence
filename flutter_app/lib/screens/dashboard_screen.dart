import 'dart:async';
import 'package:flutter/material.dart';

import '../models/models.dart';
import '../services/api_service.dart';
import '../widgets/session_helper.dart';
import 'enroll_screen.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  DashboardSummary? _summary;
  String? _error;
  Timer? _timer;
  int _snapshotTick = 0;

  @override
  void initState() {
    super.initState();
    _load();
    // Refresh counts + live feed every 5s, and bump the snapshot image
    // every 2s - independent cadences because the image is much cheaper
    // to re-fetch than re-aggregating the whole dashboard.
    _timer = Timer.periodic(const Duration(seconds: 2), (t) {
      setState(() => _snapshotTick++);
      if (_snapshotTick % 3 == 0) _load(silent: true);
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> _load({bool silent = false}) async {
    try {
      final summary = await ApiService.instance.getDashboard();
      if (!mounted) return;
      setState(() {
        _summary = summary;
        _error = null;
      });
    } catch (e) {
      await handleApiError(context, e);
      if (!silent && mounted) setState(() => _error = friendlyError(e));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Dashboard'),
        actions: [
          IconButton(
            icon: const Icon(Icons.person_add_alt_1),
            tooltip: 'Enroll employee',
            onPressed: () => Navigator.of(context).push(
              MaterialPageRoute(builder: (_) => const EnrollScreen()),
            ),
          ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: _load,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            if (_error != null)
              Card(
                color: Theme.of(context).colorScheme.errorContainer,
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Text(_error!),
                ),
              ),
            _buildLiveSnapshot(context),
            const SizedBox(height: 16),
            _buildSummaryGrid(context),
            const SizedBox(height: 16),
            Text('Recent recognitions', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            ..._buildRecentMatches(context),
          ],
        ),
      ),
    );
  }

  Widget _buildLiveSnapshot(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(12),
      child: AspectRatio(
        aspectRatio: 16 / 9,
        child: Container(
          color: Colors.black,
          child: Image.network(
            ApiService.instance.liveSnapshotUrl(),
            key: ValueKey(_snapshotTick),
            fit: BoxFit.contain,
            errorBuilder: (_, __, ___) => const Center(
              child: Text('Live feed unavailable', style: TextStyle(color: Colors.white70)),
            ),
            loadingBuilder: (context, child, progress) {
              if (progress == null) return child;
              return const Center(child: CircularProgressIndicator());
            },
          ),
        ),
      ),
    );
  }

  Widget _buildSummaryGrid(BuildContext context) {
    final s = _summary;
    final cards = [
      ('Present today', s?.presentCount, Icons.check_circle_outline, Colors.green),
      ('Absent today', s?.absentCount, Icons.remove_circle_outline, Colors.red),
      ('Currently inside', s?.currentlyInsideCount, Icons.meeting_room_outlined, Colors.blue),
      ('Total employees', s?.totalEmployees, Icons.badge_outlined, Colors.orange),
    ];
    return GridView.count(
      crossAxisCount: 2,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      mainAxisSpacing: 12,
      crossAxisSpacing: 12,
      childAspectRatio: 1.6,
      children: cards
          .map((c) => Card(
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Icon(c.$3, color: c.$4),
                      Text(
                        '${c.$2 ?? '-'}',
                        style: Theme.of(context).textTheme.headlineSmall,
                      ),
                      Text(c.$1, style: Theme.of(context).textTheme.bodySmall),
                    ],
                  ),
                ),
              ))
          .toList(),
    );
  }

  List<Widget> _buildRecentMatches(BuildContext context) {
    final matches = _summary?.recentMatches ?? [];
    if (matches.isEmpty) {
      return [const Padding(padding: EdgeInsets.all(8), child: Text('No recent activity.'))];
    }
    return matches.map((m) => _matchTile(context, m)).toList();
  }

  Widget _matchTile(BuildContext context, LiveMatch m) {
    final statusColor = switch (m.status) {
      'present' => Colors.green,
      'already_marked' => Colors.blueGrey,
      'unknown' => Colors.red,
      'entry' => Colors.teal,
      'exit' => Colors.deepOrange,
      _ => Colors.grey,
    };
    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: ListTile(
        leading: CircleAvatar(
          backgroundImage: m.thumb != null ? NetworkImage(m.thumb!) : null,
          child: m.thumb == null ? const Icon(Icons.person) : null,
        ),
        title: Text(m.name),
        subtitle: Text([m.department, m.designation].where((s) => s.isNotEmpty).join(' - ')),
        trailing: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Text(m.time, style: Theme.of(context).textTheme.bodySmall),
            Text(
              m.status.replaceAll('_', ' '),
              style: TextStyle(color: statusColor, fontSize: 12, fontWeight: FontWeight.bold),
            ),
          ],
        ),
      ),
    );
  }
}

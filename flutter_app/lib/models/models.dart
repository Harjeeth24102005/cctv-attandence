
/// Data models mirroring the JSON shapes returned by api.py.
/// Every field is defensively parsed (`?? default`) because a person who
/// hasn't been fully enrolled yet, or a movement with no matched employee,
/// can legitimately come back with nulls for name/department/etc.

class Employee {
  final String personId;
  final String name;
  final String department;
  final String designation;
  final String phone;
  final String email;
  final int totalCaptured;
  final int embeddingsCount;
  final String? thumbnail; // data:image/jpeg;base64,... or null

  Employee({
    required this.personId,
    required this.name,
    required this.department,
    required this.designation,
    required this.phone,
    required this.email,
    required this.totalCaptured,
    required this.embeddingsCount,
    this.thumbnail,
  });

  factory Employee.fromJson(Map<String, dynamic> json) {
    return Employee(
      personId: json['person_id'] ?? '',
      name: json['name'] ?? json['person_id'] ?? 'Unknown',
      department: json['department'] ?? '',
      designation: json['designation'] ?? '',
      phone: json['phone'] ?? '',
      email: json['email'] ?? '',
      totalCaptured: json['total_captured'] ?? 0,
      embeddingsCount: json['embeddings_count'] ?? 0,
      thumbnail: json['thumbnail'],
    );
  }
}

class AttendanceRecord {
  final String personId;
  final String name;
  final String department;
  final String designation;
  final String time;
  final String? date;
  final double similarity;

  AttendanceRecord({
    required this.personId,
    required this.name,
    required this.department,
    required this.designation,
    required this.time,
    required this.similarity,
    this.date,
  });

  factory AttendanceRecord.fromJson(Map<String, dynamic> json) {
    return AttendanceRecord(
      personId: json['id'] ?? '',
      name: json['name'] ?? json['id'] ?? 'Unknown',
      department: json['department'] ?? '',
      designation: json['designation'] ?? '',
      time: json['time'] ?? '',
      date: json['date'],
      similarity: (json['similarity'] ?? 0).toDouble(),
    );
  }
}

class MovementRecord {
  final String personId;
  final String name;
  final String department;
  final String direction; // ENTERING / EXITING
  final String time;
  final String? date;

  MovementRecord({
    required this.personId,
    required this.name,
    required this.department,
    required this.direction,
    required this.time,
    this.date,
  });

  factory MovementRecord.fromJson(Map<String, dynamic> json) {
    return MovementRecord(
      personId: json['id'] ?? '',
      name: json['name'] ?? json['id'] ?? 'Unknown',
      department: json['department'] ?? '',
      direction: json['direction'] ?? '',
      time: json['time'] ?? '',
      date: json['date'],
    );
  }
}

class LiveMatch {
  final String personId;
  final String name;
  final String department;
  final String designation;
  final double similarity;
  final String time;
  final String? thumb;
  final String status; // present / already_marked / unknown / entry / exit

  LiveMatch({
    required this.personId,
    required this.name,
    required this.department,
    required this.designation,
    required this.similarity,
    required this.time,
    required this.status,
    this.thumb,
  });

  factory LiveMatch.fromJson(Map<String, dynamic> json) {
    return LiveMatch(
      personId: json['id'] ?? '',
      name: json['name'] ?? json['id'] ?? 'Unknown',
      department: json['department'] ?? '',
      designation: json['designation'] ?? '',
      similarity: (json['similarity'] ?? 0).toDouble(),
      time: json['time'] ?? '',
      thumb: json['thumb'],
      status: json['status'] ?? '',
    );
  }
}

class DashboardSummary {
  final String date;
  final int totalEmployees;
  final int presentCount;
  final int absentCount;
  final int currentlyInsideCount;
  final List<LiveMatch> recentMatches;

  DashboardSummary({
    required this.date,
    required this.totalEmployees,
    required this.presentCount,
    required this.absentCount,
    required this.currentlyInsideCount,
    required this.recentMatches,
  });

  factory DashboardSummary.fromJson(Map<String, dynamic> json) {
    final matches = (json['recent_matches'] as List? ?? [])
        .map((m) => LiveMatch.fromJson(m as Map<String, dynamic>))
        .toList();
    return DashboardSummary(
      date: json['date'] ?? '',
      totalEmployees: json['total_employees'] ?? 0,
      presentCount: json['present_count'] ?? 0,
      absentCount: json['absent_count'] ?? 0,
      currentlyInsideCount: json['currently_inside_count'] ?? 0,
      recentMatches: matches,
    );
  }
}

import 'dart:async';
import 'dart:convert';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

import '../services/api_service.dart';
import '../widgets/session_helper.dart';

/// Mirrors ENROLL_POSES in app.py / api.py - order and labels matter for
/// what's shown to the person being enrolled, but the wire value ("front",
/// "left"...) must match the backend's pose names exactly.
const List<(String value, String label, IconData icon)> kPoses = [
  ('front', 'Look straight at the camera', Icons.face_outlined),
  ('left', 'Turn your head to the left', Icons.arrow_back_rounded),
  ('right', 'Turn your head to the right', Icons.arrow_forward_rounded),
  ('up', 'Tilt your head up slightly', Icons.arrow_upward_rounded),
  ('down', 'Tilt your head down slightly', Icons.arrow_downward_rounded),
];

class EnrollScreen extends StatefulWidget {
  const EnrollScreen({super.key});

  @override
  State<EnrollScreen> createState() => _EnrollScreenState();
}

class _EnrollScreenState extends State<EnrollScreen> {
  final _formKey = GlobalKey<FormState>();
  final _idController = TextEditingController();
  final _nameController = TextEditingController();
  final _deptController = TextEditingController();
  final _designationController = TextEditingController();
  final _phoneController = TextEditingController();
  final _emailController = TextEditingController();

  bool _submittingDetails = false;
  String? _detailsError;
  bool _detailsSaved = false;

  Future<void> _saveDetails() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() {
      _submittingDetails = true;
      _detailsError = null;
    });
    try {
      await ApiService.instance.enrollDetails(
        personId: _idController.text.trim(),
        name: _nameController.text.trim(),
        department: _deptController.text.trim(),
        designation: _designationController.text.trim(),
        phone: _phoneController.text.trim(),
        email: _emailController.text.trim(),
      );
      setState(() => _detailsSaved = true);
    } catch (e) {
      await handleApiError(context, e);
      if (mounted) setState(() => _detailsError = friendlyError(e));
    } finally {
      if (mounted) setState(() => _submittingDetails = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_detailsSaved) {
      return CaptureFlowScreen(personId: _idController.text.trim());
    }

    return Scaffold(
      appBar: AppBar(title: const Text('Enroll new employee')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Text(
                'Step 1 of 2 - enter their details. You\'ll capture 5 face '
                'angles (10 photos each) with the camera next.',
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _idController,
                decoration: const InputDecoration(labelText: 'Employee ID', border: OutlineInputBorder()),
                validator: (v) => (v == null || v.trim().isEmpty) ? 'Required' : null,
              ),
              const SizedBox(height: 12),
              TextFormField(
                controller: _nameController,
                decoration: const InputDecoration(labelText: 'Full name', border: OutlineInputBorder()),
                validator: (v) => (v == null || v.trim().isEmpty) ? 'Required' : null,
              ),
              const SizedBox(height: 12),
              TextFormField(
                controller: _deptController,
                decoration: const InputDecoration(labelText: 'Department (optional)', border: OutlineInputBorder()),
              ),
              const SizedBox(height: 12),
              TextFormField(
                controller: _designationController,
                decoration: const InputDecoration(labelText: 'Designation (optional)', border: OutlineInputBorder()),
              ),
              const SizedBox(height: 12),
              TextFormField(
                controller: _phoneController,
                decoration: const InputDecoration(labelText: 'Phone (optional)', border: OutlineInputBorder()),
                keyboardType: TextInputType.phone,
              ),
              const SizedBox(height: 12),
              TextFormField(
                controller: _emailController,
                decoration: const InputDecoration(labelText: 'Email (optional)', border: OutlineInputBorder()),
                keyboardType: TextInputType.emailAddress,
              ),
              if (_detailsError != null) ...[
                const SizedBox(height: 12),
                Text(_detailsError!, style: const TextStyle(color: Colors.red)),
              ],
              const SizedBox(height: 24),
              FilledButton.icon(
                onPressed: _submittingDetails ? null : _saveDetails,
                icon: _submittingDetails
                    ? const SizedBox(height: 16, width: 16, child: CircularProgressIndicator(strokeWidth: 2))
                    : const Icon(Icons.camera_alt_outlined),
                label: const Text('Continue to camera capture'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Step 2: guided multi-pose capture. Auto-captures a frame every
/// [_captureInterval] while the current pose isn't yet full, matching the
/// server's own per-pose capture cooldown (ENROLL_CAPTURE_COOLDOWN) so
/// requests aren't wasted on frames the server would reject anyway.
class CaptureFlowScreen extends StatefulWidget {
  final String personId;
  const CaptureFlowScreen({super.key, required this.personId});

  @override
  State<CaptureFlowScreen> createState() => _CaptureFlowScreenState();
}

class _CaptureFlowScreenState extends State<CaptureFlowScreen> {
  static const _captureInterval = Duration(milliseconds: 700);
  static const _perPoseTarget = 10;

  CameraController? _controller;
  Timer? _timer;
  bool _busy = false;
  bool _cameraReady = false;
  bool _finished = false;
  bool _finishing = false;
  String? _error;
  String? _hint;

  int _poseIndex = 0;
  int _currentCount = 0;

  @override
  void initState() {
    super.initState();
    _setupCamera();
  }

  Future<void> _setupCamera() async {
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      setState(() => _error = 'Camera permission is required to enroll a face.');
      return;
    }
    try {
      final cameras = await availableCameras();
      final front = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );
      final controller = CameraController(front, ResolutionPreset.medium, enableAudio: false);
      await controller.initialize();
      if (!mounted) return;
      setState(() {
        _controller = controller;
        _cameraReady = true;
      });
      _timer = Timer.periodic(_captureInterval, (_) => _captureTick());
    } catch (e) {
      setState(() => _error = 'Could not start camera: $e');
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    _controller?.dispose();
    super.dispose();
  }

  (String, String, IconData) get _currentPose => kPoses[_poseIndex];

  Future<void> _captureTick() async {
    if (_busy || _finished || _controller == null || !_controller!.value.isInitialized) return;
    _busy = true;
    try {
      final file = await _controller!.takePicture();
      final bytes = await file.readAsBytes();
      final b64 = base64Encode(bytes);

      final result = await ApiService.instance.enrollCapture(
        personId: widget.personId,
        pose: _currentPose.$1,
        base64Jpeg: b64,
      );

      if (!mounted) return;

      if (result['ok'] == true) {
        setState(() {
          _currentCount = (result['count'] as num?)?.toInt() ?? _currentCount;
          _hint = null;
        });
        final poseComplete = result['pose_complete'] == true || result['already_complete'] == true;
        if (poseComplete) {
          _advancePose();
        }
      } else {
        // Expected, frequent "try again" reasons (blur, pose mismatch, no
        // face, cooldown) - shown as a transient hint, not an error banner.
        setState(() => _hint = result['reason']?.toString());
      }
    } catch (e) {
      await handleApiError(context, e);
    } finally {
      _busy = false;
    }
  }

  void _advancePose() {
    if (_poseIndex >= kPoses.length - 1) {
      setState(() => _finished = true);
      _timer?.cancel();
      _finishEnrollment();
    } else {
      setState(() {
        _poseIndex++;
        _currentCount = 0;
        _hint = null;
      });
    }
  }

  Future<void> _finishEnrollment() async {
    setState(() => _finishing = true);
    try {
      await ApiService.instance.enrollFinish(widget.personId);
    } catch (e) {
      await handleApiError(context, e);
      if (mounted) setState(() => _error = friendlyError(e));
    } finally {
      if (mounted) setState(() => _finishing = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Capturing: ${widget.personId}')),
      body: _error != null
          ? Center(child: Padding(padding: const EdgeInsets.all(24), child: Text(_error!)))
          : (_finished ? _buildDoneView(context) : _buildCaptureView(context)),
    );
  }

  Widget _buildCaptureView(BuildContext context) {
    if (!_cameraReady || _controller == null) {
      return const Center(child: CircularProgressIndicator());
    }
    final (_, label, icon) = _currentPose;

    return Column(
      children: [
        Expanded(
          child: Stack(
            fit: StackFit.expand,
            children: [
              CameraPreview(_controller!),
              Positioned(
                top: 16,
                left: 16,
                right: 16,
                child: Card(
                  color: Colors.black54,
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: Row(
                      children: [
                        Icon(icon, color: Colors.white),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(label, style: const TextStyle(color: Colors.white, fontSize: 16)),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
              if (_hint != null)
                Positioned(
                  bottom: 96,
                  left: 16,
                  right: 16,
                  child: Card(
                    color: Colors.orange.shade100,
                    child: Padding(
                      padding: const EdgeInsets.all(8),
                      child: Text(_hint!, textAlign: TextAlign.center),
                    ),
                  ),
                ),
            ],
          ),
        ),
        Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text('Pose ${_poseIndex + 1} of ${kPoses.length}'),
                  Text('$_currentCount / $_perPoseTarget'),
                ],
              ),
              const SizedBox(height: 8),
              LinearProgressIndicator(value: _currentCount / _perPoseTarget),
              const SizedBox(height: 12),
              Row(
                children: List.generate(kPoses.length, (i) {
                  final done = i < _poseIndex;
                  final active = i == _poseIndex;
                  return Expanded(
                    child: Container(
                      margin: const EdgeInsets.symmetric(horizontal: 2),
                      height: 6,
                      decoration: BoxDecoration(
                        color: done
                            ? Colors.green
                            : active
                                ? Theme.of(context).colorScheme.primary
                                : Colors.grey.shade300,
                        borderRadius: BorderRadius.circular(3),
                      ),
                    ),
                  );
                }),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildDoneView(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (_finishing) ...[
              const CircularProgressIndicator(),
              const SizedBox(height: 16),
              const Text('Building face embeddings...'),
            ] else ...[
              const Icon(Icons.check_circle, color: Colors.green, size: 64),
              const SizedBox(height: 16),
              Text(
                '${widget.personId} is enrolled and can now be recognized on the entrance camera.',
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 24),
              FilledButton(
                onPressed: () => Navigator.of(context).popUntil((r) => r.isFirst),
                child: const Text('Done'),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

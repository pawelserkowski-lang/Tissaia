import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:file_picker/file_picker.dart';
import 'package:desktop_drop/desktop_drop.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:animate_do/animate_do.dart';
import 'package:provider/provider.dart';

// --- CONFIG ---
const String API_URL = "http://localhost:8000";

// --- MODELS ---
class RestoreJob {
  String jobId;
  String status;
  String? resultUrl;
  String? error;

  RestoreJob({required this.jobId, required this.status, this.resultUrl, this.error});

  factory RestoreJob.fromJson(Map<String, dynamic> json) {
    return RestoreJob(
      jobId: json['job_id'] ?? "",
      status: json['status'] ?? "UNKNOWN",
      resultUrl: json['result_url'],
      error: json['error'],
    );
  }
}

// --- PROVIDER ---
class TissaiaProvider with ChangeNotifier {
  String statusMessage = "SYSTEM GOTOWY";
  bool isProcessing = false;
  RestoreJob? currentJob;
  double progress = 0.0;

  Future<void> processFile(PlatformFile file) async {
    isProcessing = true;
    progress = 0.1;
    statusMessage = "INICJOWANIE POŁĄCZENIA...";
    notifyListeners();

    try {
      // 1. UPLOAD
      statusMessage = "WYSYŁANIE PAKIETU DANYCH...";
      var request = http.MultipartRequest('POST', Uri.parse('$API_URL/process_upload'));
      request.files.add(await http.MultipartFile.fromPath('file', file.path!));

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        var data = jsonDecode(response.body);
        String jobId = data['job_id'];
        currentJob = RestoreJob(jobId: jobId, status: "QUEUED");
        statusMessage = "DANE PRZESŁANE. ID ZADANIA: $jobId";
        progress = 0.3;
        notifyListeners();

        // 2. POLL STATUS
        _pollStatus(jobId);
      } else {
        throw Exception("Błąd serwera: ${response.statusCode}");
      }
    } catch (e) {
      statusMessage = "BŁĄD KRYTYCZNY: $e";
      isProcessing = false;
      notifyListeners();
    }
  }

  void _pollStatus(String jobId) async {
    const interval = Duration(seconds: 2);
    int attempts = 0;

    Timer.periodic(interval, (timer) async {
      attempts++;
      try {
        var res = await http.get(Uri.parse('$API_URL/status/$jobId'));
        if (res.statusCode == 200) {
          var data = jsonDecode(res.body);
          var job = RestoreJob.fromJson(data);
          currentJob = job;

          if (job.status == "PROCESSING") {
            statusMessage = "PRZETWARZANIE NEURO-SIECIOWE... [PRÓBA $attempts]";
            progress = (progress < 0.8) ? progress + 0.05 : 0.8;
          } else if (job.status == "COMPLETED") {
            statusMessage = "REKONSTRUKCJA ZAKOŃCZONA POMYŚLNIE.";
            progress = 1.0;
            isProcessing = false;
            timer.cancel();
          } else if (job.status == "FAILED") {
            statusMessage = "ZADANIE NIEUDANE: ${job.error}";
            isProcessing = false;
            timer.cancel();
          }
          notifyListeners();
        }
      } catch (e) {
        print("Polling error: $e");
      }
    });
  }

  void reset() {
    statusMessage = "SYSTEM GOTOWY";
    isProcessing = false;
    currentJob = null;
    progress = 0.0;
    notifyListeners();
  }
}

// --- MAIN ---
void main() {
  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => TissaiaProvider()),
      ],
      child: const NecroApp(),
    ),
  );
}

class NecroApp extends StatelessWidget {
  const NecroApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Tissaia V14',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        scaffoldBackgroundColor: Colors.black,
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF00FF41), // Cyber Green
          secondary: Color(0xFF008F11),
          surface: Colors.black,
          background: Colors.black,
        ),
        textTheme: GoogleFonts.courierPrimeTextTheme(
          Theme.of(context).textTheme.apply(bodyColor: const Color(0xFF00FF41)),
        ),
        useMaterial3: true,
      ),
      home: const MainScreen(),
    );
  }
}

class MainScreen extends StatelessWidget {
  const MainScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final provider = Provider.of<TissaiaProvider>(context);

    return Scaffold(
      body: Stack(
        children: [
          // Grid Background Effect
          Positioned.fill(
            child: CustomPaint(painter: GridPainter()),
          ),

          Column(
            children: [
              // Header
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  border: Border(bottom: BorderSide(color: Theme.of(context).primaryColor, width: 2)),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.terminal, color: Color(0xFF00FF41), size: 32),
                    const SizedBox(width: 10),
                    Text("TISSAIA V14 :: NECRO_OS",
                      style: GoogleFonts.courierPrime(fontSize: 24, fontWeight: FontWeight.bold, letterSpacing: 2)),
                    const Spacer(),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                      decoration: BoxDecoration(border: Border.all(color: const Color(0xFF00FF41))),
                      child: Text("STAN: ONLINE", style: GoogleFonts.courierPrime(fontSize: 12)),
                    )
                  ],
                ),
              ),

              // Content
              Expanded(
                child: Row(
                  children: [
                    // Left Panel: Drop Zone
                    Expanded(
                      flex: 1,
                      child: Padding(
                        padding: const EdgeInsets.all(20.0),
                        child: DropTarget(
                          onDragDone: (details) {
                            if (details.files.isNotEmpty && !provider.isProcessing) {
                              // Convert XFile to PlatformFile for simplicity in this demo structure
                              // In real app, might need better handling
                              // provider.processFile(details.files.first);
                              // DropTarget returns XFile, FilePicker returns PlatformFile.
                              // For now, let's just use the button to avoid type mismatch issues in this snippet.
                            }
                          },
                          child: Container(
                            decoration: BoxDecoration(
                              border: Border.all(color: const Color(0xFF00FF41).withOpacity(0.5), width: 1),
                              color: const Color(0xFF001100),
                            ),
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(Icons.upload_file, size: 64, color: const Color(0xFF00FF41).withOpacity(0.7)),
                                const SizedBox(height: 20),
                                Text("PRZECIĄGNIJ OBRAZ TUTAJ", style: GoogleFonts.courierPrime(fontSize: 16)),
                                const SizedBox(height: 20),
                                ElevatedButton.icon(
                                  onPressed: provider.isProcessing ? null : () async {
                                    FilePickerResult? result = await FilePicker.platform.pickFiles(type: FileType.image);
                                    if (result != null) {
                                      provider.processFile(result.files.single);
                                    }
                                  },
                                  icon: const Icon(Icons.add, color: Colors.black),
                                  label: Text("WYBIERZ PLIK", style: GoogleFonts.courierPrime(fontWeight: FontWeight.bold)),
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: const Color(0xFF00FF41),
                                    foregroundColor: Colors.black,
                                    shape: const RoundedRectangleBorder(),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ),
                    ),

                    // Right Panel: Preview & Terminal
                    Expanded(
                      flex: 1,
                      child: Padding(
                        padding: const EdgeInsets.all(20.0),
                        child: Column(
                          children: [
                            // Result View
                            Expanded(
                              flex: 2,
                              child: Container(
                                width: double.infinity,
                                decoration: BoxDecoration(
                                  border: Border.all(color: const Color(0xFF00FF41), width: 1),
                                ),
                                child: provider.currentJob?.resultUrl != null
                                    ? Image.network("$API_URL${provider.currentJob!.resultUrl}", fit: BoxFit.contain)
                                    : Center(
                                        child: Text("[BRAK DANYCH WIZUALNYCH]",
                                          style: GoogleFonts.courierPrime(color: const Color(0xFF00FF41).withOpacity(0.5))),
                                      ),
                              ),
                            ),
                            const SizedBox(height: 10),

                            // Terminal Output
                            Expanded(
                              flex: 1,
                              child: Container(
                                width: double.infinity,
                                padding: const EdgeInsets.all(10),
                                decoration: BoxDecoration(
                                  color: const Color(0xFF050505),
                                  border: Border.all(color: const Color(0xFF00FF41), width: 1),
                                ),
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text("> SYSTEM LOG:", style: GoogleFonts.courierPrime(fontWeight: FontWeight.bold)),
                                    const SizedBox(height: 5),
                                    Text("> ${provider.statusMessage}", style: GoogleFonts.courierPrime(fontSize: 12)),
                                    if (provider.isProcessing) ...[
                                      const SizedBox(height: 10),
                                      LinearProgressIndicator(
                                        value: provider.progress,
                                        backgroundColor: const Color(0xFF003300),
                                        color: const Color(0xFF00FF41)
                                      ),
                                    ]
                                  ],
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class GridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = const Color(0xFF00FF41).withOpacity(0.05)
      ..strokeWidth = 1;

    const double step = 40;
    for (double x = 0; x < size.width; x += step) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), paint);
    }
    for (double y = 0; y < size.height; y += step) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

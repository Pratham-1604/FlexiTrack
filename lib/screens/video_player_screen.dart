import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:video_player/video_player.dart';
import 'package:chewie/chewie.dart';

class VideoPlayerScreen extends StatefulWidget {
  const VideoPlayerScreen({
    super.key,
    required this.choice,
  });
  final int choice;
  @override
  _VideoPlayerScreenState createState() => _VideoPlayerScreenState();
}

class _VideoPlayerScreenState extends State<VideoPlayerScreen> {
  final ImagePicker _picker = ImagePicker();
  late VideoPlayerController _videoPlayerController;
  late ChewieController _chewieController;
  bool _videoReceived = false;
  bool _isLoading = false;
  String displayTextPick = 'Pick a video';
  String displayTextSend = 'Upload the video';
  File? videoFile;
  bool isError = false;
  String errorMessage = 'Some error occured! Retry!';

  Future<void> pickVideo() async {
    final XFile? pickedVideo = await _picker.pickVideo(
      source: ImageSource.gallery,
    );

    if (pickedVideo != null) {
      setState(() {
        videoFile = File(pickedVideo.path);
        displayTextPick = 'Upload another video';
      });
      debugPrint("video picked");
    }
  }

  Future<void> sendVideoFunction() async {
    if (videoFile != null) {
      setState(() {
        _isLoading = true;
      });
      final responseVideo = await sendVideo(videoFile!);
      debugPrint("video sent");
      if (responseVideo != null) {
        debugPrint("video received");
        final directory = await getApplicationDocumentsDirectory();
        final File newVideo = File('${directory.path}/video.mp4');
        newVideo.writeAsBytesSync(responseVideo.bodyBytes);

        await initializePlayer(newVideo.path);
        setState(() {
          _isLoading = false;
          _videoReceived = true;
        });
      }
    } else {
      debugPrint("video not received");
      setState(() {
        _isLoading = false;
        displayTextPick = 'Pick a video';
        displayTextSend = 'Upload this video';
        videoFile = null;
        isError = true;
      });
    }
  }

  Future<http.Response?> sendVideo(File videoFile) async {
    try {
      final url = Uri.parse(
        'http:localhost:5000/upload?choice=${widget.choice}',
      );
      var request = http.MultipartRequest(
        'POST',
        url,
      );
      request.files.add(
        await http.MultipartFile.fromPath('video_file', videoFile.path),
      );
      var response = await request.send();
      var receivedResponse = await http.Response.fromStream(response);

      if (receivedResponse.statusCode == 200) {
        return receivedResponse;
      } else {
        debugPrint(
          'Server responded with status code: ${receivedResponse.statusCode}',
        );
        debugPrint('Response body: ${receivedResponse.body}');
        setState(() {
          isError = true;
        });
        return null;
      }
    } catch (e) {
      debugPrint(e.toString());
      return null;
    }
  }

  Future<void> initializePlayer(String videoPath) async {
    _videoPlayerController = VideoPlayerController.file(File(videoPath));
    await Future.wait([
      _videoPlayerController.initialize(),
    ]);
    _chewieController = ChewieController(
      videoPlayerController: _videoPlayerController,
      autoPlay: true,
      looping: true,
      allowFullScreen: true,
    );
  }

  @override
  void dispose() {
    _videoPlayerController.dispose();
    _chewieController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Center(
      child: _isLoading
          ? Column(
              mainAxisSize: MainAxisSize.max,
              mainAxisAlignment: MainAxisAlignment.center,
              children: const [
                CircularProgressIndicator(),
                SizedBox(height: 20),
                Text(
                  'Fetching results! Pls wait',
                  style: TextStyle(color: Colors.grey),
                ),
              ],
            )
          : isError
              ? Text(
                  errorMessage,
                  style: const TextStyle(color: Colors.grey),
                )
              : _videoReceived
                  ? Chewie(
                      controller: _chewieController,
                    )
                  : Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      mainAxisSize: MainAxisSize.max,
                      children: [
                        ElevatedButton(
                          onPressed: pickVideo,
                          child: Text(
                            displayTextPick,
                            style: const TextStyle(fontSize: 24),
                          ),
                        ),
                        if (videoFile != null)
                          ElevatedButton(
                            onPressed: sendVideoFunction,
                            child: Text(
                              displayTextSend,
                              style: const TextStyle(fontSize: 24),
                            ),
                          ),
                      ],
                    ),
    );
  }
}

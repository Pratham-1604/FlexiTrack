import 'package:fitnessfreakerzz/main.dart';
import 'package:fitnessfreakerzz/screens/video_player_screen.dart';
import 'package:flutter/material.dart';
class Page2 extends StatelessWidget {
  const Page2({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Pushup'),
        backgroundColor: Colors.blueGrey,
      ),
      backgroundColor: Colors.black,

      body: const Center(
        child: VideoPlayerScreen(choice: 2),
      ),
    );
  }
}
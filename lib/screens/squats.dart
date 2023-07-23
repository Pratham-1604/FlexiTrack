import 'package:flutter/material.dart';

import 'video_player_screen.dart';
class Page3 extends StatelessWidget {
  const Page3({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Squats'),
        backgroundColor: Colors.blueGrey,
      ),
      backgroundColor: Colors.black,
      body: const Center(
        child: VideoPlayerScreen(choice: 3),
      ),
    );
  }
}
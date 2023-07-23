import 'package:flutter/material.dart';

import 'video_player_screen.dart';
class Page1 extends StatelessWidget {
  const Page1({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Plank'),
        backgroundColor: Colors.blueGrey,
      ),
      backgroundColor: Colors.black,

      body: const Center(
        child: VideoPlayerScreen(choice: 1),
      ),
    );
  }
}
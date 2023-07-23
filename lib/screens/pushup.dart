import 'package:flutter/material.dart';

import 'video_player_screen.dart';
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
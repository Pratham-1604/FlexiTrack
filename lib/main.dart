import 'package:FlexiTrack/screens/select_screen.dart';
import 'package:flutter/material.dart';

import 'screens/plank.dart';
import 'screens/pushup.dart';
import 'screens/squats.dart';

void main() {
  runApp(
    const VideoPlayerApp(),
  );
}

class VideoPlayerApp extends StatelessWidget {
  const VideoPlayerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'FlexiTrack',
      home: const HomeScreen(),
      theme: ThemeData(
        backgroundColor: Colors.black,
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.grey,
            textStyle: const TextStyle(
              color: Colors.black,
            ),
          ),
        ),
      ),
      routes: {
        '/page1': (context) => const Page1(),
        '/page2': (context) => const Page2(),
        '/page3': (context) => const Page3(),
      },
    );
  }
}

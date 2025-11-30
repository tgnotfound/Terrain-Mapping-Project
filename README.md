📘 Overview

This project implements a compact LiDAR-based terrain mapping system capable of generating real-time 2-D (and partial 3-D) spatial profiles of an environment.
A LiDAR sensor mounted on a wheeled robotic platform collects continuous range data, which is processed into point clouds and terrain maps using Python-based visualization tools.
The project demonstrates the core principles of LiDAR sensing, data fusion, mapping algorithms, and real-time environmental reconstruction.

🎯 Objectives
Integrate a LiDAR sensor with a microcontroller/processor (ESP32/Raspberry Pi).
Acquire continuous 2-D scan data and convert it into meaningful terrain representations.
Implement filtering, coordinate conversion, and noise reduction.
Visualize terrain using point-cloud plotting tools.
Compare LiDAR results with ultrasonic sensor mapping.
Build a functional low-cost terrain mapping prototype.

🛠️ Hardware Components
RPLiDAR A-Series (A1M8) – 360° scanning, ~6 m range
Microcontroller / Processor
ESP32 (data transmission)
Raspberry Pi (host system, data processing)
Custom Interface PCB
5-pin Relimate connector
IMU (I²C)
Encoder interface
Power System – 12V battery pack
Motors & Motor Driver
Wheeled Chassis

💻 Software Components
Python (data acquisition, processing, visualization)
Open3D / Matplotlib for point-cloud rendering
Serial/UART Interface for LiDAR communication
Noise filtering algorithms
Polar → Cartesian coordinate transformation

🧭 System Workflow
LiDAR Initialization
Continuous Scan Data Acquisition
Data Cleaning (noise removal, thresholding)
Coordinate Transformation (distance + angle → x,y)
Point Cloud Formation
Terrain Map Visualization
Accuracy Evaluation
Comparison with Ultrasonic Mapping

📊 Results Summary
Generated 2-D terrain maps clearly represented slopes, obstacles, and uneven surfaces.
Produced dense and consistent point clouds.
LiDAR outperformed ultrasonic sensors in:
Resolution
Field of view (360°)
Accuracy
Scanning speed
Map completeness

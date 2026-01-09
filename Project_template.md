## IMU-Based Handwritten Digit Recognition System

### Introduction


Handwritten digit recognition is a fundamental problem in human–computer interaction and pattern recognition. While most existing approaches rely on vision-based methods, such systems often require high computational resources and are sensitive to lighting and camera placement.

This project explores an alternative approach by recognizing handwritten digits using IMU (Inertial Measurement Unit) data collected from the wrist while writing on paper. The system captures motion patterns generated during handwriting and uses machine learning models to classify digits based on temporal IMU signals.

The primary objective of this project is to design an efficient, low-latency edge AI pipeline capable of recognizing digits from wrist motion data, making it suitable for embedded and wearable devices.

Potential applications include:

Smart pens and wearable input devices

Assistive technologies

Low-power edge-based digit input systems

---

### Methodology

#### List of hardware required and their specifications:-

| Device               | Description                                               |
| -------------------- | --------------------------------------------------------- |
| Arduino Nicla Vision | Edge AI board used for IMU data collection and deployment |
| IMU Sensor (Onboard) | 6-axis accelerometer and gyroscope                        |
| Laptop (Local)       | Data preprocessing, model training, and evaluation        |

#### List of software used :-

| Software                     | Purpose                                |
| ---------------------------- | -------------------------------------- |
| Arduino IDE                  | IMU data collection                    |
| Python                       | Data preprocessing and model training  |
| TensorFlow / TensorFlow Lite | Model development and deployment       |
| NumPy, Pandas                | Signal processing and dataset handling |

---

### Data collection

IMU data was collected by attaching the Arduino Nicla Vision board to the back of a marker. Participants wrote digits on paper using the marker, enabling the capture of natural wrist and hand motion during handwriting.

Data was collected from five team members, with each participant writing numbers sequentially from 0 to 100. This resulted in multiple samples for each digit (0–9) and introduced natural variations in writing style and motion dynamics.

Classes: Digits 0–9

Sensors Used: Onboard accelerometer and gyroscope

Sampling Type: Time-series IMU data

Data Format: Multivariate sequences (Ax, Ay, Az, Gx, Gy, Gz)

The IMU data was recorded as a continuous stream and later segmented into individual digit samples. Each segment was labeled according to the corresponding digit written and prepared for model training


### Model development and compression

### 1. Handwritten Digit Recognition (IMU-based)

-Input: Time-series IMU data (Ax, Ay, Az, Gx, Gy, Gz)

-Model: Lightweight neural network for multivariate time-series classification

-Classes: Digits 0–9

Training:

   Optimizer: Adam

   Learning Rate: 0.001

   Loss Function: Categorical Cross-Entropy

Compression:

Compact model architecture designed for edge deployment

Reduced parameter count to fit embedded memory constraints

###2. Model deployment:

| Module                      | Hardware             | Notes                                        |
| --------------------------- | -------------------- | -------------------------------------------- |
| IMU-based Digit Recognition | Arduino Nicla Vision | On-device digit prediction from wrist motion |


The deployed system continuously acquires IMU data from the Nicla Vision board. The incoming time-series data is segmented into individual digit samples and passed to the deployed model for inference. The predicted digit is generated directly on the device, enabling real-time operation without reliance on external computation or cloud services.

---

### Prototype and Demo

- Real-time dashboard: Receives alerts via MQTT and displays status  
- Demo: A patient saying "Help" or waving triggers alert with bed ID  
- Integration: Multi-modal data from IMU and camera merged to generate alarm if any unusual activity detected.  

---

### Challenges and Workarounds

1. **Limited Dataset Size**  
   - Problem: Small dataset led to potential overfitting  
   - Solution: Data augmentation and careful regularization (dropout, batch norm)  

2. **Noisy Real-world Data**  
   - Hand gestures and audio were inconsistent in hospital-like settings  
   - Applied filtering, smoothing, and spectrogram preprocessing  

3. **Hardware Limitations**  
   - Resource constraints on edge devices (Nicla Vision, Nano BLE)  
   - Used compact CNNs, model quantization, and offloaded complex inference  

4. **MQTT Connectivity**  
   - Network instability affected MQTT transmission  
   - Used robust error handling and buffering  

---


### References

1. [TensorFlow Documentation](https://www.tensorflow.org/)  
2. [OpenMV IDE Docs](https://docs.openmv.io/)  
3. [Arduino Nano BLE 33 Docs](https://docs.arduino.cc/hardware/nano-33-ble/)  
4. [Raspberry Pi Pico](https://www.raspberrypi.com/products/raspberry-pi-pico/)  
5. [MQTT Protocol](https://mqtt.org/)  
6. [MFCC for Audio Processing](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)  
7. [Codebase (GitHub)](https://github.com/HTCSUYOGJARE/UrgentCare-AI-Patinet-Monitoring-System.git)  

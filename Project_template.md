## UrgentCare-AI: Patient Monitoring System

### Introduction


In critical care environments, timely detection of patient needs is essential. UrgentCare-AI addresses this by implementing a real-time AI-based system that enhances patient monitoring using three integrated modules: 

- **“Help” Call Notification (Keyword Spotting):** Detects when a patient verbally calls for help.  
- **Movement Detection:** Monitors hand-waving gestures.  
- **Bed Presence Detection:** Tracks whether a patient is on the bed.

The modules operate independently but contribute collectively to ensure rapid alerting and continuous surveillance. The system leverages compact AI models suitable for embedded deployment and uses MQTT messaging for inter-device communication (except KWS, which directly alerts staff).

In hospitals and elderly care environments, patients may not always be in a position to use conventional alerting systems. UrgentCare-AI aims to bridge this gap by enabling: 

- Contactless assistance requests through audio and gestures  
- Unattended patient movement detection  
- Immediate alerts in emergency scenarios  

This system reduces the delay between patient need and caregiver response, thus improving safety and care quality.

---

### Methodology

#### List of hardware required and their specifications:-

| Device           | Description                                                  |
|------------------|--------------------------------------------------------------|
| Nicla Vision     | Compact AI vision board with camera, microphone and onboard Wi-Fi |
| Arduino Nano BLE 33 | TinyML-compatible microcontroller with IMU                |
| Raspberry Pi Pico W | Offloads ML inference and publishes MQTT messages          |
| Server (Local Laptop) | Subscribes to MQTT topics, triggers alerts                |

#### List of software used :-

| Software           | Purpose                                           |
|--------------------|--------------------------------------------------|
| TensorFlow         | Model training                                   |
| OpenMV IDE         | Nicla Vision programming                         |
| Thonny             | IMU gesture classification, Server and analytics, Keyword spotting, Server, analytics & inference |
| Edge Impulse Studio | Audio data recording & Model Training            |

---

### Data collection

**1. Keyword Spotting**

- Classes: Help, Noise  
- Training Set: 365 samples (Help: 196, Noise: 169)  
- Test Set: 92 samples (Help: 49, Noise: 43)  

**2. Movement Detection**

- Classes: Idle (I), Waving_hand (W)  
- Samples: 6500 samples per class  
- Captured using IMU on Arduino Nano BLE 33  

**3. Bed Presence Detection**

- Classes: Present, Absent  
- Training Set: 172 samples (Present: 136, Absent: 36)  
- Test Set: 43 samples (Present: 37, Absent: 6)  
- Captured using onboard camera of Nicla Vision  

---

### Model development and compression

### 1. Help Call Notification (KWS)

- Input: MFCC Spectrogram (98×40)  
- Model: Conv2D → MaxPooling → BatchNorm → Dropout → Dense → Softmax  
- Training:  
  - Epochs: 20  
  - Batch Size: 128  
  - Optimizer: Adam  
  - Learning Rate: 0.001  
- Performance:  
  - Accuracy: 91%  
  - F1 Score: 0.91  
  - ROC AUC: 0.  

---

### 2. Movement Detection

- Model: Trained on Arduino Nano BLE 33  
- Classes: Idle vs. Waving  
- Inference: Offloaded to Raspberry Pi Pico  
- Data Pipeline: MQTT publish via pat/imu  

---

### 3. Bed Presence Detection

- Model: CNN with 2 Conv layers + Pooling + Dropout  
- Performance:  
  - Accuracy: 85.7%  
  - AUC: 0.75  
  - F1 Score: 0.84  
- Inference on Nicla Vision using OpenMV IDE  

---

### Model deployment:

| Module           | Hardware              | Notes                                |
|------------------|-----------------------|------------------------------------|
| KWS              | Nicla Vision          | Alerts staff directly with bed ID  |
| Activity Detection | Nano BLE 33 + RPi Pico | Publishes to MQTT topic pat/imu    |
| Person Present/Absence on bed | Nicla Vision | Publishes to MQTT topic openmv/test |

A local Python process on the server subscribes to MQTT topics and evaluates:  

- Presence + Gesture = Valid Movement  
- Absence + Gesture = Emergency/Unusual activity  

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

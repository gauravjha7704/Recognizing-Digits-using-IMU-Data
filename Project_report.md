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
| OpenMV IDE                  | IMU data collection                    |
| Python                       | Data preprocessing and model training  |
| TensorFlow  | Model development        |
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

- **Input:** Time-series IMU data (Ax, Ay, Az, Gx, Gy, Gz)  
- **Model:** 1D Convolutional Neural Network (CNN) for multivariate time-series classification  
- **Classes:** Digits 0–9  

- **Training:**  
  - Optimizer: Adam  
  - Learning Rate: 0.001  
  - Loss Function: Categorical Cross-Entropy  

- **Model Design:**  
  - Lightweight architecture suitable for edge deployment  
  - Reduced parameter count to meet embedded memory and compute constraints  


### 2. Model deployment:

| Module                      | Hardware             | Notes                                        |
| --------------------------- | -------------------- | -------------------------------------------- |
| IMU-based Digit Recognition | Arduino Nicla Vision | On-device digit prediction from wrist motion |


The deployed system continuously acquires IMU data from the Nicla Vision board. The incoming time-series data is segmented into individual digit samples and passed to the deployed model for inference. The predicted digit is generated directly on the device, enabling real-time operation without reliance on external computation or cloud services.

---

### Prototype and Demo
A working prototype of the IMU-based handwritten digit recognition system was implemented using the Arduino Nicla Vision board attached to the back of a marker.

During the demonstration:

The user wrote digits on paper using the instrumented marker

IMU data was captured in real time from the wrist motion

The trained model performed on-device inference

The predicted digit was displayed as output

The demo validated the end-to-end pipeline, from IMU data acquisition to real-time digit prediction, demonstrating the feasibility of recognizing handwritten digits using wrist-mounted IMU data on an edge device.
 

---

### Challenges and Workarounds

1. **Variation in Writing Styles**  
   - Different participants had varying handwriting speeds and stroke patterns
   - Collected data from multiple users to improve generalization  

2. **Segmentation of Continuous IMU Data**  
   - Continuous writing from 0 to 100 made it difficult to isolate individual digits
   - Applied manual or rule-based segmentation to extract digit-level samples

3. **Similarity in Motion Patterns Between Digits**  
   - Certain digits such as 1 and 7, and 0 and 6, exhibited similar wrist motion
     patterns
   - This led to confusion between classes and reduced classification accuracy for these pairs 

4. **Limited Edge Resources**  
   - Nicla Vision has constrained memory and computational capacity
   - Used a lightweight model architecture suitable for embedded deployment
  
4. **Consistency in Sensor Attachment**  
   - Slight changes in sensor placement affected IMU readings
   - Maintained a consistent attachment position on the marker across all recordings
  

---


### References

1. [TensorFlow Documentation](https://www.tensorflow.org/)  
2. [OpenMV IDE Docs](https://docs.openmv.io/)  
3. [IMU-Based Handwriting and Gesture Recognition](https://ieeexplore.ieee.org/)
4. [Convolutional Neural Networks for Time-Series Data – Background on 1D CNNs](https://arxiv.org/).

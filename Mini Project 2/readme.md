# Smart Road Condition Classification Using Motion Sensors and LSTM

## Objective
To develop an Android application that captures accelerometer and gyroscope data from a two-wheeler rider's smartphone and classifies the type of road surface in real-time using an LSTM (Long Short-Term Memory) neural network.

## Road Conditions Classified
The system is trained to recognize and classify the following road types:
- Kankar Road
- Bitumen Road
- Concrete Road
- Single Speed Breaker
- Multiple Speed Breakers

## Methodology

### Data Collection App
- Developed using MIT App Inventor.
- Collects real-time accelerometer and gyroscope data at a frequency of 50 Hz.

### Preprocessing
- Data is segmented into 3-second windows (150 samples per segment).
- Normalization and scaling are applied for consistency in input.

### Model Training
- Model built using TensorFlow and Keras.
- LSTM network trained to identify unique patterns in sensor data associated with different road types.

### Evaluation
- Achieved approximately 87% accuracy on test data consisting of unseen road conditions.
- Designed for real-time classification and mobile deployment.

## Model Architecture
- LSTM Layer with 64 units
- Dropout Layer for regularization
- Dense Layer with ReLU activation
- Output Layer with Softmax activation (5 classes)

## Tech Stack
- Android App: MIT App Inventor
- Backend & Model Training: Python, NumPy, Pandas, Scikit-learn
- Deep Learning Framework: TensorFlow / Keras

## Results
- Model Accuracy: Approximately 87%
- Supports real-time inference
- Lightweight and optimized for mobile deployment


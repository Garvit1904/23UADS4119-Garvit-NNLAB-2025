# MNIST Handwritten Digit Classification

## Objective
This project aims to develop a deep learning model using TensorFlow to classify handwritten digits from the MNIST dataset. The model is a fully connected neural network (Multi-Layer Perceptron) that learns to recognize digits (0-9) based on their pixel values.

## Description of the Model
The model consists of:
- **Input Layer:** 784 neurons (each representing a pixel in the 28x28 image)
- **Hidden Layer 1:** 128 neurons with ReLU activation
- **Hidden Layer 2:** 64 neurons with ReLU activation
- **Output Layer:** 10 neurons (one for each digit class) with softmax activation
- **Loss Function:** Softmax cross-entropy
- **Optimizer:** Adam Optimizer
- **Batch Size:** 100
- **Epochs:** 10

## Description of the Code
### 1. Importing Required Libraries
- TensorFlow for deep learning
- NumPy for numerical operations
- TensorFlow Datasets for loading MNIST data

### 2. Loading & Preprocessing Data
- Load MNIST dataset using `tfds.load()`
- Normalize images (convert pixel values from 0-255 to 0-1)
- Flatten images to 1D vectors (28x28 → 784)
- One-hot encode labels (convert integer labels into binary vectors)
- Create batches of 100 images for efficient training

### 3. Building the Neural Network
- Define weight and bias tensors
- Implement forward propagation using matrix multiplications and activation functions
- Compute the loss using softmax cross-entropy
- Optimize the weights using the Adam optimizer

### 4. Training the Model
- Run training for 10 epochs
- Process batches dynamically using TensorFlow’s dataset API
- Compute the average loss per epoch

### 5. Performance Evaluation
- Evaluate model accuracy on the test set
- Measure how well the model generalizes to unseen data

## Performance Evaluation
After training for 10 epochs, the model achieves:
- **Training Accuracy:** ~98%
- **Test Accuracy:** ~97%

The model effectively classifies handwritten digits with high accuracy, demonstrating the power of deep learning for image recognition tasks.

---




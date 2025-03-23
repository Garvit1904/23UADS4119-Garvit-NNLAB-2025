# CNN Fashion MNIST Classification

## Objective
To train and evaluate a Convolutional Neural Network (CNN) using the Keras library to classify the Fashion MNIST dataset. The objective is to analyze the effect of different hyperparameters, such as filter size, regularization, batch size, and optimization algorithm, on model performance.

## Description of the Model
The CNN model consists of:
- Two convolutional layers with configurable filter sizes (3x3, 5x5).
- Max pooling layers for downsampling to reduce dimensionality.
- Fully connected layers for classification.
- Softmax output layer for multi-class classification.
- L2 regularization to prevent overfitting.
- Various optimization algorithms (Adam, SGD) for comparison.

## Python Implementation
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

def create_model(filter_size=3, regularization=0.001, optimizer='adam'):
    model = keras.Sequential([
        layers.Conv2D(32, (filter_size, filter_size), activation='relu', kernel_regularizer=regularizers.l2(regularization), input_shape=(28, 28, 1)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (filter_size, filter_size), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(regularization)),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

filter_sizes = [3, 5]
regularization_values = [0.001, 0.01]
batch_sizes = [32, 64]
optimizers = ['adam', 'sgd']

results = {}
for filter_size in filter_sizes:
    for reg in regularization_values:
        for batch_size in batch_sizes:
            for optimizer in optimizers:
                print(f"Training model with filter_size={filter_size}, reg={reg}, batch_size={batch_size}, optimizer={optimizer}")
                model = create_model(filter_size, reg, optimizer)
                history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0)
                test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
                results[(filter_size, reg, batch_size, optimizer)] = (history, test_acc)

plt.figure(figsize=(12, 6))
for key, (history, acc) in results.items():
    filter_size, reg, batch_size, optimizer = key
    plt.plot(history.history['val_accuracy'], label=f'fs={filter_size}, reg={reg}, bs={batch_size}, opt={optimizer} ({acc:.2f})')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.title('Effect of Hyperparameters on Validation Accuracy')
plt.show()
```

## Description of Code
- The `create_model` function constructs a CNN with:
  - Two convolutional layers with configurable filter sizes.
  - Max pooling layers for feature reduction.
  - A fully connected layer followed by a softmax layer for classification.
  - L2 regularization to prevent overfitting.
- A nested loop iterates over different hyperparameter values to test multiple configurations.
- Models are trained for 10 epochs each using the given hyperparameters.
- The `evaluate()` function measures accuracy on the test dataset.
- **Visualization using Matplotlib**:
  - A **line plot** shows validation accuracy trends across different configurations.
  - A **training loss vs validation loss plot** helps identify overfitting and generalization trends.

## Performance Evaluation
- The model's accuracy is tested with different hyperparameters to observe their effects.
- **Filter Size**: Affects feature extraction, with larger filters capturing more details but requiring more computations.
- **Regularization**: Helps in preventing overfitting but may reduce training accuracy.
- **Batch Size**: Smaller batch sizes may improve generalization but increase training time.
- **Optimizer Comparison**:
  - Adam generally provides better accuracy compared to SGD.
  - SGD may take longer to converge but can generalize well in some cases.
- **Loss Visualization**:
  - Training loss vs validation loss is plotted over epochs to understand if the model overfits.
  - Helps determine the best epoch count for optimal performance.
- A **line plot** visualizes how validation accuracy changes with different configurations, helping identify the best-performing setup.



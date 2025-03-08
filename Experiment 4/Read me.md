# README.md

## Objective
WAP to evaluate the performance of an implemented three-layer neural network with variations in activation functions, size of hidden layer, learning rate, batch size, and number of epochs.

## Description of the Model
- The model is a feedforward neural network with two hidden layers:
  - **Layer 1:** 128 neurons with ReLU activation
  - **Layer 2:** 64 neurons with ReLU activation
  - **Output Layer:** 10 neurons (one per digit) with softmax activation
- The network uses the Adam optimizer with a learning rate of 0.01.
- The loss function is softmax cross-entropy.
- The dataset is preprocessed by normalizing pixel values (0-1) and flattening images into vectors of 784 elements.
- Training is performed using different batch sizes (1, 10, 100) and epochs (10, 50, 100).
- Performance metrics include accuracy, loss curves, and confusion matrices.

## Python Implementation
```python
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
from openpyxl import Workbook
tf.disable_v2_behavior()

mnist = tfds.load('mnist', split=['train', 'test'], as_supervised=True)
train_data, test_data = mnist

def preprocess(images, labels):
    images = tf.cast(images, tf.float32) / 255.0  # Normalize
    images = tf.reshape(images, [784])  # Flatten
    labels = tf.one_hot(labels, depth=10)  # One-hot encode
    return images, labels

batch_size_list = [1,10, 100]
epochs_list = [10, 50,100]

def train_and_evaluate(batch_size, epochs):
    print(f"\nTraining with batch_size={batch_size}, epochs={epochs}")

    train_dataset = train_data.map(preprocess).batch(batch_size)
    test_dataset = test_data.map(preprocess).batch(batch_size)

    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])

    weights = {
        'h1': tf.Variable(tf.random_normal([784, 128])),
        'h2': tf.Variable(tf.random_normal([128, 64])),
        'out': tf.Variable(tf.random_normal([64, 10]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([128])),
        'b2': tf.Variable(tf.random_normal([64])),
        'out': tf.Variable(tf.random_normal([10]))
    }

    def neural_network(x):
        layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
        layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
        return tf.add(tf.matmul(layer2, weights['out']), biases['out'])

    logits = neural_network(X)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    predictions = tf.nn.softmax(logits)
    correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    loss_curve, acc_curve, val_acc_curve = [], [], []
    start_time = time.time()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            avg_loss = 0
            total_batches = 0
            iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset)
            next_batch = iterator.get_next()

            while True:
                try:
                    batch_x, batch_y = sess.run(next_batch)
                    _, c = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
                    avg_loss += c
                    total_batches += 1
                except tf.errors.OutOfRangeError:
                    break  

            avg_loss /= total_batches
            train_acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
            val_acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})

            loss_curve.append(avg_loss)
            acc_curve.append(train_acc)
            val_acc_curve.append(val_acc)

            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        end_time = time.time()
        execution_time = end_time - start_time

        test_acc = []
        y_true, y_pred = [], []
        iterator = tf.compat.v1.data.make_one_shot_iterator(test_dataset)
        next_batch = iterator.get_next()

        while True:
            try:
                batch_x, batch_y = sess.run(next_batch)
                acc, preds = sess.run([accuracy, predictions], feed_dict={X: batch_x, Y: batch_y})
                test_acc.append(acc)
                y_pred.extend(np.argmax(preds, axis=1))
                y_true.extend(np.argmax(batch_y, axis=1))
            except tf.errors.OutOfRangeError:
                break

        final_test_acc = np.mean(test_acc)
        print(f"Test Accuracy: {final_test_acc:.4f}")

        conf_matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix (Batch={batch_size}, Epochs={epochs})")
        cm_filename = f"confusion_matrix_batch{batch_size}_epochs{epochs}.png"
        plt.savefig(cm_filename)
        plt.close()

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(loss_curve, label='Train Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curve")

        plt.subplot(1, 2, 2)
        plt.plot(acc_curve, label='Train Accuracy')
        plt.plot(val_acc_curve, label='Val Accuracy')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy Curve")

        curve_filename = f"curves_batch{batch_size}_epochs{epochs}.png"
        plt.savefig(curve_filename)
        plt.close()

    return batch_size, epochs, execution_time, final_test_acc, cm_filename, curve_filename

results = []
for batch_size in batch_size_list:
    for epochs in epochs_list:
        res = train_and_evaluate(batch_size, epochs)
        results.append(res)

df = pd.DataFrame(results, columns=['Batch Size', 'Epochs', 'Execution Time (s)', 'Test Accuracy', 'Confusion Matrix Image', 'Loss/Accuracy Curves'])
df.to_excel("training_results.xlsx", index=False)

print("\n All Results Saved in training_results.xlsx")

```

## Description of Code
- The MNIST dataset is loaded using `tensorflow_datasets` and split into training and test sets.
- Preprocessing steps include normalizing pixel values and one-hot encoding labels.
- A feedforward neural network is built using TensorFlow 1.x with manually defined weight and bias variables.
- The training loop iterates through epochs, computing loss and accuracy for each batch.
- Accuracy is evaluated on the test dataset after training.
- Confusion matrices and accuracy/loss curves are generated and saved as images.
- Results from different batch size and epoch combinations are stored in an Excel file.

## Performance Evaluation
### Test Accuracy
The model’s final accuracy is stored in `training_results.xlsx`.

### Confusion Matrix
A confusion matrix is computed and saved for each batch size and epoch combination.

### Loss Curve
The loss curve shows how the training loss decreases over epochs.

### Accuracy Curve
Both training and validation accuracy curves are plotted.

## My Comments
- **TensorFlow 1.x Deprecation:** The code uses TensorFlow 1.x, which is outdated. Consider updating it to TensorFlow 2.x for better performance and maintainability.
- **Weight Initialization:** Instead of using `tf.random_normal`, using `tf.keras.initializers.HeNormal()` may improve performance.
- **Data Loading Efficiency:** The current approach reinitializes the iterator every epoch. Using `tf.data`’s `repeat()` and `prefetch()` can improve efficiency.
- **Validation Set:** The code evaluates validation accuracy on the training set instead of a separate validation set.
- **Learning Rate Optimization:** A fixed learning rate of 0.01 may not be optimal; consider using an adaptive learning rate scheduler.
- **Batch Size and Training Stability:** Training with a batch size of 1 can cause instability. Using larger batch sizes with gradient accumulation may improve convergence.


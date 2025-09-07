# 🧵 Fashion MNIST CNN Classifier

A basic **machine learning project** built with **TensorFlow** and **TensorFlow Datasets** to classify clothing images from the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).  
This was a **hands-on practice project** to learn dataset handling, preprocessing, CNN architecture design, training, evaluation, and prediction visualization.

---

## 📂 Dataset
- **Name:** Fashion MNIST
- **Classes:** 10 clothing categories (e.g., T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- **Image Size:** 28×28 pixels, grayscale
- **Train/Test Split:** 60,000 training images, 10,000 test images

---

## 🏗 Model Architecture
The model is a **Convolutional Neural Network (CNN)** with the following layers:

1. **Conv2D** — 32 filters, 3×3 kernel, `same` padding, ReLU activation  
2. **MaxPooling2D** — 2×2 pool size, stride 2  
3. **Conv2D** — 64 filters, 3×3 kernel, `same` padding, ReLU activation  
4. **MaxPooling2D** — 2×2 pool size, stride 2  
5. **Flatten** — converts 2D feature maps to 1D vector  
6. **Dense** — 128 neurons, ReLU activation  
7. **Dense** — 10 neurons, Softmax activation (one per class)

**Optimizer:** Adam  
**Loss Function:** Sparse Categorical Crossentropy  
**Metric:** Accuracy

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```
### 2️⃣ Install Dependencies
```bash
pip install tensorflow tensorflow-datasets matplotlib numpy
```
### 📜 Code Walkthrough
- Data Loading & Metadata
Loads Fashion MNIST from tensorflow_datasets with as_supervised=True for (image, label) pairs and with_info=True for metadata.
- Preprocessing
Normalizes pixel values from [0, 255] to [0, 1] and caches datasets for faster training.
- Visualization
Displays sample images and their labels before training.
- Model Definition
Builds a CNN using tf.keras.Sequential.
- Training
Trains for 5 epochs with a batch size of 32.
- Evaluation
Prints test accuracy and visualizes predictions.

### 📊 Model Output

Add your training logs, accuracy, and sample prediction plots here.

Example:

Accuracy on test dataset: 0.91


Sample Predictions

### 📄 License
This project is licensed under the MIT License — see the LICENSE file for details.

### ✍️ Author
This project was created as part of my hands-on learning journey in machine learning and deep learning with TensorFlow.

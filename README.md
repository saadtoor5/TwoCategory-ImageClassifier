# 🧠 TwoCategory-ImageClassifier

A generic deep learning image classifier built with TensorFlow and Keras that can distinguish between any **two categories** provided by the user. Simply place your images in respective folders and the model will handle training, evaluation, and prediction.

---

## 📌 About the Project

This project allows users to train a **binary image classification** model that can distinguish between any two categories (e.g., Car vs Bike, Cat vs Dog, Apple vs Orange) using a **Convolutional Neural Network (CNN)**.

It includes:

* Data preprocessing using `ImageDataGenerator`
* Data augmentation
* CNN model building
* Training and validation visualization
* Single image prediction

---

## 🚀 Features

* 🖼️ Custom dataset support for two classes
* 🔄 Data augmentation techniques
* 📊 Training and validation accuracy/loss plots
* 🧪 Single image prediction capability
* ⚙️ Easy-to-use modular code structure

---

## ⚙️ Tech Stack

* Python 3.11 or older
* TensorFlow / Keras
* NumPy
* Matplotlib

---

## 📁 Project Structure

```
TwoCategory-ImageClassifier/
├── images/                        # Folder for sample input/output screenshots
├── Car-Bike-Dataset/             # Your dataset folder (rename as per your classes)
│   ├── class1/                   # First category (e.g., Car)
│   └── class2/                   # Second category (e.g., Bike)
├── two_category_classifier.py    # Main training and prediction script
├── two_category_classifier.ipynb # Jupyter Notebook
├── requirements.txt              # Required libraries and dependencies
├── README.md                     # Project overview
```

---

## 💻 How to Use

1. **Prepare Dataset**

   * Organize your dataset into two subfolders inside a main folder (e.g., `Car-Bike-Dataset/class1` and `Car-Bike-Dataset/class2`).

2. **Clone the Repository**

```bash
https://github.com/your-username/TwoCategory-ImageClassifier.git
cd TwoCategory-ImageClassifier
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the Classifier**

```bash
python two_category_classifier.py
```

5. **Predict New Image**

After training, call the `predict_image(model, 'path_to_image.jpg')` function with your desired image path.

---

## 🧪 Sample Output

```
Training and Validation Accuracy
Training and Validation Loss
The image is predicted to be a Car with confidence 0.94
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE)

---

## 👨‍💻 Author

Made by [@saadtoor5](https://github.com/saadtoor5) — feel free to fork, contribute, or star the repo!

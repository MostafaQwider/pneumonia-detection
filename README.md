🩺 Pneumonia Detection using Deep Learning

Pneumonia Detection is an AI-powered project designed to automatically classify chest X-ray images as Pneumonia or Normal using a Convolutional Neural Network (CNN). The system provides fast and reliable predictions with confidence scores to assist medical professionals, researchers, and developers in early diagnosis and decision-making.

🚀 Features

🔹 Automatic classification of chest X-ray images

🔹 Deep Learning CNN model optimized for accuracy

🔹 Provides prediction outputs with confidence scores

🔹 Modular and clean project structure

🔹 Easily integratable with medical systems or web applications


🧰 Technologies Used

Python

TensorFlow / Keras

NumPy, Pandas

Matplotlib, Seaborn

Scikit-learn

Jupyter Notebook / Python scripts

Flask (for web API)

▶️ How to Run

Install dependencies
```
pip install -r requirements.txt
```

Train the model (optional)
```
python model_training.py
```

Run the Flask API for predictions
```
python app.py
```

Send an image for prediction
Use POST /predict with an X-ray image to get prediction and confidence score.


📊 Evaluation

Accuracy, Precision, Recall, F1-score are calculated for both classes

Confusion matrix available in model_training.ipynb

🎯 Goal

Provide an open-source AI model for early detection of pneumonia, supporting medical professionals and developers in building intelligent healthcare applications.

📄 License

This project is released under the MIT License.






# Spam Detection using Machine Learning

## Overview

This project implements a spam detection system that classifies text messages as either "ham" (non-spam) or "spam" using machine learning. The model is trained using the **Linear Support Vector Classifier (SVC)** and evaluates performance using various metrics like accuracy, classification report, and precision-recall curve. The dataset used is the widely-known "SMSSpamCollection" dataset, which contains SMS messages labeled as either spam or ham.

## Features

- **Text Preprocessing**: Removes noise from messages such as numbers, URLs, and punctuation.
- **Feature Extraction**: Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorizer to convert text into numerical features.
- **Spam Classification**: Classifies a message as "ham" (non-spam) or "spam" using **LinearSVC**.
- **Model Tuning**: Uses **GridSearchCV** to fine-tune the model's hyperparameters (e.g., the regularization parameter `C`).
- **Performance Evaluation**: Reports the model's accuracy, classification report (precision, recall, f1-score), and visualizes performance with a **precision-recall curve**.

## Installation

### Prerequisites

Ensure that you have **Python 3.7 or higher** installed. The required libraries are listed in the `requirements.txt` file.

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

### Dependencies

The following libraries are required to run the project:

- `scikit-learn` - For machine learning algorithms and metrics.
- `pandas` - For data manipulation.
- `nltk` - For text preprocessing (e.g., stopword removal, lemmatization).
- `matplotlib` - For plotting the precision-recall curve.

If you don't have them installed, you can uncomment the following line in the code to install the libraries:

```bash
# !pip install scikit-learn pandas nltk matplotlib
```

## Dataset

The model is trained on the **SMSSpamCollection**, a dataset of SMS messages. The dataset contains labeled text messages:

- **Ham (non-spam)**: Legitimate messages (e.g., "Hey, how are you?").
- **Spam**: Unsolicited or irrelevant messages (e.g., "Congratulations! You've won a free iPhone!").

### Dataset Format

The dataset consists of two columns:

- **Label**: Either `ham` (non-spam) or `spam`.
- **Message**: The text message to classify.

Example:
```
ham    Hello, I hope you're doing well!
spam   You have won a lottery! Click here to claim your prize!
```

## Model Overview

The model follows the steps below:

1. **Text Preprocessing**: 
   - Remove numbers, punctuation, and URLs.
   - Tokenize and lemmatize words.
   - Remove stopwords (commonly used words like "the", "is", etc.).

2. **Feature Extraction**: 
   - Convert the text into numerical features using **TF-IDF Vectorizer**. It captures both word frequency and the importance of the words in the document.

3. **Model Training**:
   - **LinearSVC** (Support Vector Machine) is used as the classification model.
   - **GridSearchCV** is used for hyperparameter tuning (e.g., selecting the best value of the regularization parameter `C`).

4. **Model Evaluation**:
   - The model is evaluated based on accuracy, precision, recall, F1-score, and AUC (area under the curve) of the precision-recall curve.

## Code Walkthrough

### 1. **Text Preprocessing**:
The preprocessing function performs the following operations:
- Removes numbers from the text.
- Removes URLs (e.g., `http://example.com`).
- Removes punctuation.
- Tokenizes the message and removes stopwords.
- Lemmatizes each word to its base form (e.g., "running" becomes "run").

### 2. **Feature Extraction**:
We use **TF-IDF Vectorizer** to convert text messages into a matrix of numerical features. The vectorizer uses bigrams and extracts up to 2000 features from the text.

### 3. **Model Training**:
The **LinearSVC** algorithm is used to classify messages. GridSearchCV is applied to find the optimal value of the hyperparameter `C` (regularization parameter).

### 4. **Model Evaluation**:
The model is evaluated using the following metrics:
- **Accuracy**: The percentage of correctly classified messages.
- **Classification Report**: Precision, recall, F1-score for both ham and spam classes.
- **Precision-Recall Curve**: A curve that plots precision vs. recall, with the area under the curve (AUC) used to measure performance.

## How to Use

### 1. **Run the Script**:
To train the model and make predictions, simply run the Python script:

```bash
python spam_detection.py
```

This will:
- Load the dataset.
- Preprocess the text messages.
- Extract features using TF-IDF.
- Split the data into training and testing sets.
- Train a **LinearSVC** model.
- Evaluate the model's performance.

### 2. **Predict New Messages**:
To classify a new message as ham or spam, you can modify the script to accept input or use the trained model directly:

```python
# Assuming the model is saved as `best_model`
new_message = "Congratulations! You've won a free iPhone!"
processed_message = preprocess_text(new_message)
features = vectorizer.transform([processed_message])
prediction = best_model.predict(features)
print(f"Prediction: {'Spam' if prediction[0] == 1 else 'Ham'}")
```

### 3. **Retrain the Model**:
If you want to retrain the model with the latest dataset or change the hyperparameters, you can modify the script and rerun it.

## Performance Evaluation

### 1. **Accuracy**:
The model achieves an accuracy of approximately **97.85%** on the test set, which means it correctly classifies 97.85% of the messages.

### 2. **Classification Report**:
The classification report includes:
- **Precision**: How many of the predicted spam messages are actually spam.
- **Recall**: How many of the actual spam messages are correctly identified.
- **F1-score**: The harmonic mean of precision and recall, providing a balanced evaluation.

### 3. **Precision-Recall Curve**:
A precision-recall curve is plotted, and the **AUC (Area Under the Curve)** is calculated to measure the overall performance of the classifier. A higher AUC indicates better performance.

## Future Improvements

- **Improving Spam Detection**: Experiment with different algorithms (e.g., Random Forest, Naive Bayes, or deep learning models) to improve the performance of spam classification.
- **Hyperparameter Optimization**: Use advanced techniques like RandomizedSearchCV or Bayesian Optimization for better hyperparameter tuning.
- **Handle Imbalanced Dataset**: The dataset has an imbalance (more ham than spam). Future improvements could focus on handling this imbalance through techniques like oversampling or using different evaluation metrics.

## Acknowledgements

- This project utilizes the following open-source libraries:
  - [Scikit-learn](https://scikit-learn.org/) for machine learning algorithms and metrics.
  - [Pandas](https://pandas.pydata.org/) for data manipulation and processing.
  - [NLTK](https://www.nltk.org/) for text processing tasks.
  - [Matplotlib](https://matplotlib.org/) for plotting the precision-recall curve.
- The dataset used is from the **SMSSpamCollection**, a popular dataset for SMS spam classification.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

---




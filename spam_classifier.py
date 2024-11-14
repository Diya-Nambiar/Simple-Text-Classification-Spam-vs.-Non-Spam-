# Install necessary libraries (uncomment the line below if you haven't installed these packages)
# !pip install scikit-learn pandas nltk matplotlib

# Import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, auc
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
import matplotlib.pyplot as plt

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
df = pd.read_csv('SMSSpamCollection', sep='\t', names=["label", "message"], encoding='ISO-8859-1')

# Encode labels as binary values (0 for ham, 1 for spam)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
    return ' '.join(words)

# Apply preprocessing
df['message'] = df['message'].apply(preprocess_text)

# Initialize TF-IDF Vectorizer with bigrams
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['message'])  # Features
y = df['label']  # Target variable

# Split the dataset into training (80%) and test sets (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid for SVM
param_grid = {'C': [0.1, 1, 10]}

# Use GridSearchCV to find the best hyperparameters for LinearSVC
grid_search = GridSearchCV(LinearSVC(class_weight='balanced'), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, best_model.decision_function(X_test))
pr_auc = auc(recall, precision)

# Plot the precision-recall curve
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (AUC = {pr_auc:.2f})')
plt.show()

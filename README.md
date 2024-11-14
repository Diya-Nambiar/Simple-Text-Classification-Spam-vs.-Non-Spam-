# Simple-Text-Classification-Spam-vs.-Non-Spam
This project is a simple implementation of an SMS Spam Classifier using Naive Bayes. The goal is to classify SMS messages as either "ham" (non-spam) or "spam." This is achieved using text preprocessing and machine learning with the Naive Bayes algorithm.
## Project Objectives
Build a simple but effective model to classify SMS messages as spam or non-spam.
Use basic Natural Language Processing (NLP) techniques to convert text messages into machine-readable numerical features.
Implement and evaluate a Naive Bayes classifier using the preprocessed data to achieve high accuracy with minimal complexity.
## Dataset
The dataset used in this project is the SMS Spam Collection dataset from the UCI Machine Learning Repository. It contains a set of labeled SMS messages classified as either "ham" or "spam."

-Number of samples: 5,574
-Number of classes: 2 (ham and spam)
-Format: The dataset consists of two columns:
-label: Label indicating if the message is ham (non-spam) or spam.
-message: The actual SMS message text.
link for data : https://archive.ics.uci.edu/dataset/228/sms+spam+collection


## Prerequisites
The following Python packages are required to run this project:

-pandas: For data manipulation and analysis.

-scikit-learn: For building and evaluating the model.

-nltk: For natural language processing, specifically for handling stopwords.

-Install these packages with:


pip install pandas scikit-learn nltk

## Conclusion 
After running the simple Spam vs. Non-Spam Classification project, we can draw conclusions on several key aspects: model performance, the effectiveness of text preprocessing, and how machine learning can be applied to text data. Let's break down the key points based on the process and the results from the evaluation.

## 1. Model Performance
The Naive Bayes classifier is a very simple but effective model for text classification tasks, especially when dealing with text data that has a clear and relatively simple structure, such as spam messages. It is based on applying Bayes' Theorem, assuming that the features (words) are conditionally independent, which works quite well for this type of problem.

**Accuracy**: The accuracy metric provides a straightforward way to understand how well the model is doing overall. Accuracy tells us the proportion of correct predictions (both spam and non-spam) out of the total number of predictions made. For instance, if the accuracy is 90%, it means 90% of the messages in the test set were correctly classified as spam or non-spam.

**Classification Report**: The classification report goes beyond accuracy and includes important metrics like:

**Precision**: The ratio of correctly predicted positive instances (spam) out of all instances predicted as spam. Precision tells us how many of the spam messages the model identified were actually spam.

**Recall**: The ratio of correctly predicted positive instances (spam) out of all actual positive instances (all the true spam messages). Recall tells us how many of the actual spam messages the model was able to identify.

**F1-Score**: The harmonic mean of precision and recall, which balances both metrics. This score gives us a better understanding of the model's ability to handle both types of errors (false positives and false negatives).
If we have high precision and recall for the spam class, it means the model is effectively distinguishing spam from non-spam without mistakenly classifying too many legitimate messages as spam.

![Screenshot 2024-11-14 221234](https://github.com/user-attachments/assets/5ee7eb09-c562-415f-95e0-250f6c105164)


## 2. Text Preprocessing and Feature Extraction
In the project, the text data underwent a basic preprocessing step, where:

-**Tokenization**: The text was broken down into individual words (tokens).

-**Stop Words Removal**: Common but meaningless words like "the", "is", etc., were removed because they don’t contribute much to distinguishing spam from non-spam.

-**Bag of Words**: This technique turned the text into a vector of word counts, allowing the model to learn from the frequency of each word in the messages.

Why is preprocessing important?

-Noise Reduction: Removing unnecessary words (like stop words) helps reduce noise in the data and allows the model to focus on the more important words.

-Text to Numerical Representation: Machine learning models can’t directly work with text, so converting text into numerical features (like word counts) allows us to apply models like Naive Bayes.

-Using CountVectorizer helped generate this numerical representation in the form of a term-document matrix.

## 3. Effectiveness of Naive Bayes for Text Classification
Naive Bayes is particularly well-suited for text classification tasks because:

-**Scalability**: It performs well on large datasets because it makes relatively few assumptions, is computationally efficient, and is easy to implement.
Works Well for High-Dimensional Data: Text data typically has high dimensionality (lots of unique words), and Naive Bayes tends to work effectively even in such scenarios, as it assumes independence between features (words), which simplifies the learning process.
However, Naive Bayes assumes word independence, which is often not true in real-world text data. Despite this, it performs surprisingly well because many of the correlations between words don’t significantly affect the performance in simple classification tasks like spam detection.

## 4. Potential Improvements and Extensions
Although Naive Bayes performs well for this simple task, there are many ways the model could be improved for real-world applications:

-**Data Augmentation**: Adding more messages to the dataset would help improve the model's performance, especially on more varied types of text.

-**Feature Engineering**: Instead of just using word counts, other feature extraction techniques like TF-IDF (Term Frequency-Inverse Document Frequency) can be used to give more weight to words that are more informative across documents, rather than just frequent.

-**Advanced Models**: While Naive Bayes is a great starting point, more advanced models like Support Vector Machines (SVM), Logistic Regression, or deep learning-based models (such as Recurrent Neural Networks or Transformers) could improve classification performance, especially for more complex datasets.
-**Cross-validation**: Instead of just splitting the data once into training and testing sets, using k-fold cross-validation would provide a more reliable estimate of the model’s performance by testing it on different subsets of the data.

## 5. Real-World Applications
Spam detection is just one example of how this simple text classification model can be used. Some other real-world applications of text classification include:

-**Email Filtering**: Classifying emails as spam or non-spam automatically.

-**Customer Support**: Automatically categorizing support tickets into different topics (e.g., billing issues, technical problems).

-**Content Moderation**: Identifying offensive or harmful content in user-generated texts.

-**Sentiment Analysis**: Classifying reviews or comments into positive, negative, or neutral categories.

## License
This project is licensed under the MIT License. See the LICENSE file for more information.







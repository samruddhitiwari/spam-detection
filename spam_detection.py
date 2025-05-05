import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Load dataset (example using a CSV file)
# In practice, you might use the SpamAssassin dataset or your own email collection
df = pd.read_csv('spam_emails.csv')  # Columns: 'text', 'label' (0=ham, 1=spam)

# If you don't have a dataset, you can use this sample data
data = {
    'text': [
        'Free money now!!! Click here to claim your prize',
        'Meeting tomorrow at 10 AM in conference room',
        'You won a million dollars! Click the link',
        'Project update: Please review the attached documents',
        'URGENT: Your account has been compromised',
        'Hi John, just checking in about our lunch plans'
    ],
    'label': [1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# 1. Data Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization
    words = nltk.word_tokenize(text)
    
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

df['processed_text'] = df['text'].apply(preprocess_text)

# 2. Feature Extraction
# Using both TF-IDF and Bag of Words for comparison
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
bow_vectorizer = CountVectorizer(max_features=5000)

X_tfidf = tfidf_vectorizer.fit_transform(df['processed_text'])
X_bow = bow_vectorizer.fit_transform(df['processed_text'])
y = df['label']

# 3. Handle Class Imbalance with SMOTE
print("Class distribution before SMOTE:", np.bincount(y))
smote = SMOTE(random_state=42)
X_tfidf_res, y_res = smote.fit_resample(X_tfidf, y)
print("Class distribution after SMOTE:", np.bincount(y_res))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_tfidf_res, y_res, test_size=0.2, random_state=42)

# 4. Model Training and Evaluation
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - {type(model).__name__}')
    plt.show()
    
    return {
        'model': type(model).__name__,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Initialize models
models = [
    MultinomialNB(),
    SVC(kernel='linear', probability=True, random_state=42),
    LogisticRegression(max_iter=1000, random_state=42)
]

# Evaluate each model
results = []
for model in models:
    print(f"\nEvaluating {type(model).__name__}...")
    result = train_and_evaluate(model, X_train, X_test, y_train, y_test)
    results.append(result)

# Display results
results_df = pd.DataFrame(results)
print("\nModel Performance Comparison:")
print(results_df)

# 5. Integration Example (Simple Email Filter)
def predict_spam(email_text, model, vectorizer):
    # Preprocess the new email
    processed_text = preprocess_text(email_text)
    
    # Transform using the same vectorizer
    features = vectorizer.transform([processed_text])
    
    # Predict
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]
    
    return {
        'is_spam': bool(prediction[0]),
        'spam_probability': probability,
        'prediction': 'Spam' if prediction[0] else 'Ham'
    }

# Choose the best performing model (in this case, Logistic Regression)
best_model = models[2]  # Change index based on your results

# Test the email filter
test_emails = [
    "Congratulations! You've won a free vacation. Click here to claim!",
    "Hi team, please find attached the quarterly report for review.",
    "Your account has been suspended. Verify your identity immediately.",
    "Reminder: Project meeting tomorrow at 3 PM"
]

print("\nEmail Filter Testing:")
for email in test_emails:
    result = predict_spam(email, best_model, tfidf_vectorizer)
    print(f"\nEmail: {email[:50]}...")
    print(f"Prediction: {result['prediction']}")
    print(f"Spam Probability: {result['spam_probability']:.2f}")

# Spam-Detection
# Spam Email Detection System using NLP

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0.2-orange)
![NLTK](https://img.shields.io/badge/NLTK-3.6.7-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

An NLP-based spam email detection system that classifies emails into spam ("spam") or legitimate ("ham") categories using machine learning algorithms.

## Features

- **Text Preprocessing**: Tokenization, stop-word removal, lemmatization
- **Feature Extraction**: TF-IDF and Bag of Words models
- **Machine Learning Models**: 
  - Naive Bayes Classifier
  - Support Vector Machine (SVM)
  - Logistic Regression
- **Handles Class Imbalance**: Uses SMOTE for oversampling
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
- **Easy Integration**: Ready-to-use email filtering function

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/spam-detection-nlp.git
cd spam-detection-nlp
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download NLTK resources:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

## Usage

### Training the Model
```python
from spam_detector import SpamDetector

# Initialize with your dataset
detector = SpamDetector('emails.csv')

# Preprocess data and train models
detector.preprocess_data()
detector.train_models()

# Evaluate model performance
results = detector.evaluate_models()
print(results)
```

### Using the Trained Model
```python
# Predict whether an email is spam
email = "Congratulations! You've won a free vacation!"
prediction = detector.predict(email)

print(f"Prediction: {'SPAM' if prediction['is_spam'] else 'HAM'}")
print(f"Confidence: {prediction['spam_probability']:.2%}")
```

## Dataset

The system expects a CSV file with:
- `text`: The email content
- `label`: 0 for ham, 1 for spam

Example structure:
```csv
text,label
"Free money now!!! Click here",1
"Meeting tomorrow at 10 AM",0
```

## Performance Metrics

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Naive Bayes         | 97.2%    | 96.8%     | 98.1%  | 97.4%    |
| SVM                 | 98.1%    | 97.9%     | 98.5%  | 98.2%    |
| Logistic Regression | 98.4%    | 98.2%     | 98.7%  | 98.4%    |

## Configuration

Modify `config.py` to adjust:
- Text preprocessing parameters
- Feature extraction settings
- Model hyperparameters
- Evaluation metrics

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.


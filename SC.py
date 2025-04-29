import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class RacismDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(max_iter=1000)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)
    
    def train(self, dataset_path):
        # Load dataset (format: text,label where 1=racist, 0=not racist)
        data = pd.read_csv(dataset_path)
        
        # Preprocess text
        data['processed_text'] = data['text'].apply(self.preprocess_text)
        
        # Vectorize text
        X = self.vectorizer.fit_transform(data['processed_text'])
        y = data['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
    
    def predict(self, text):
        # Preprocess input text
        processed_text = self.preprocess_text(text)
        # Vectorize
        text_vector = self.vectorizer.transform([processed_text])
        # Predict
        prediction = self.model.predict(text_vector)
        probability = self.model.predict_proba(text_vector)[0]
        
        return {
            'is_racist': bool(prediction[0]),
            'probability': float(probability[1]),  # probability of being racist
            'text': text,
            'processed_text': processed_text
        }

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = RacismDetector()
    
    # Train with dataset (you need to prepare this dataset)
    # Dataset format: CSV with 'text' and 'label' columns (1=racist, 0=not racist)
    detector.train('racism_dataset.csv')
    
    # Test with sample text
    test_texts = [
        "All people should be treated equally regardless of race.",
        "People from that race are all criminals and should be avoided.",
        "Cultural differences make our society richer and more interesting."
    ]
    
    for text in test_texts:
        result = detector.predict(text)
        print(f"\nText: {text}")
        print(f"Racist: {result['is_racist']} (Probability: {result['probability']:.2f})")
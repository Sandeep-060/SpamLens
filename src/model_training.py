from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

def create_pipeline():

  return Pipeline([
    ('tfidf',TfidfVectorizer(stop_words='english', max_features=5000)),
    ('classifier',MultinomialNB())
  ])

def prepare_features_and_labels(data):

    X = data['message']
    y = data['label'].map({'ham': 0, 'spam': 1})
    return X, y

def train_and_save_model(X, y, model_path='models/spam_model.pkl'):

    model = create_pipeline()
    model.fit(X, y)
    joblib.dump(model, model_path)
    print(f"✅ Model trained and saved to {model_path}")
    return model
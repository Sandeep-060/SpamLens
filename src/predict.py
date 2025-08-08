import joblib

def load_model(model_path='models/spam_model.pkl'):

    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded from {model_path}")
        return model
    except FileNotFoundError:
        print(f"❌ Model not found at {model_path}")
        print("💡 Run `python train.py` first to train and save the model.")
        raise

def predict_message(message):

    model = load_model()

    pred = model.predict([message])[0] 
    prob = model.predict_proba([message])[0][1] 
    
    return pred, prob


if __name__ == "__main__":
    msg = "You got frd "
    pred, prob = predict_message(msg)
    
    if pred == 1:
        print(f"🚨 SPAM! (Confidence: {prob:.2f})")
    else:
        print(f"✅ NOT SPAM (Confidence: {1 - prob:.2f})")
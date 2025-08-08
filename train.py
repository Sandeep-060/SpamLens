from src.data_preprocessing import load_clean_data
from src.model_training import prepare_features_and_labels, train_and_save_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

# Step 1: Load and clean
data = load_clean_data('data/raw/spam.csv')

# Step 2: Prepare X and y
X, y = prepare_features_and_labels(data)

# Step 3: Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: Train on training data only
model = train_and_save_model(X_train, y_train, 'models/spam_model.pkl')

# Step 5: Evaluate on test data
y_pred = model.predict(X_test)
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
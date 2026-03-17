# -------------------- IMPORT LIBRARIES --------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# -------------------- LOAD DATASET --------------------
# Make sure your path is correct
df = pd.read_csv("./dataset/spam.csv", encoding='latin-1')

# -------------------- DATA CLEANING --------------------
# Keep only required columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

# Convert labels to numbers (ham = 0, spam = 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# -------------------- SPLIT DATA --------------------
X = df['message']   # Input (messages)
y = df['label']     # Output (spam/ham)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------- TEXT VECTORIZATION --------------------
vectorizer = CountVectorizer()

# Learn + transform training data
X_train_vec = vectorizer.fit_transform(X_train)

# Transform test data
X_test_vec = vectorizer.transform(X_test)

# -------------------- MODEL TRAINING --------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -------------------- PREDICTION --------------------
y_pred = model.predict(X_test_vec)

# -------------------- EVALUATION --------------------
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# -------------------- CUSTOM INPUT TEST --------------------
msg = ["Congratulations! You won a free ticket"]

msg_vec = vectorizer.transform(msg)
prediction = model.predict(msg_vec)

# -------------------- OUTPUT RESULT --------------------
if prediction[0] == 1:
    print("Prediction: Spam Message")
else:
    print("Prediction: Not Spam Message ")
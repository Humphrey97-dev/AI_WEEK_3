#Part 2: Practical Implementation
#Task 1: Classical ML with Scikit-learn (Iris Dataset)
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Preprocessing - Encode labels if needed
# (The labels are already numeric, so we don't need to do additional encoding)
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize DecisionTreeClassifier
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

# Print metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')

#Task 2: Deep Learning with TensorFlow (MNIST)
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)) / 255.0
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)) / 255.0

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)

# Print the test accuracy
print(f'Test Accuracy: {test_acc:.2f}')

# Visualize predictions
plt.figure(figsize=(10, 10))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {model.predict(X_test[i:i+1]).argmax()}")
    plt.axis('off')
plt.show()

#Task 3: NLP with spaCy (Amazon Reviews)
# Import necessary libraries
import spacy
from spacy import displacy
from collections import Counter
import pandas as pd

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample Amazon reviews (you can load your own dataset here)
reviews = [
    "I love this product! The camera is amazing and the battery life is great.",
    "Worst phone ever, it broke on the second day. Don't waste your money!",
    "Very satisfied with the purchase. Excellent quality and fast shipping."
]

# Process the reviews through spaCy
docs = [nlp(review) for review in reviews]

# Extract named entities
for doc in docs:
    print(f"Review: {doc.text}")
    print("Entities: ", [(ent.text, ent.label_) for ent in doc.ents])
    print()

# Sentiment analysis (basic rule-based approach)
positive_keywords = ['love', 'great', 'excellent', 'satisfied']
negative_keywords = ['worst', 'broke', 'waste']

for review in reviews:
    sentiment = "positive" if any(word in review.lower() for word in positive_keywords) else "negative" if any(word in review.lower() for word in negative_keywords) else "neutral"
    print(f"Review: {review}\nSentiment: {sentiment}\n")


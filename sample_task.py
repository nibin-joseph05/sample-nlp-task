import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/simple_sentiment_for_nlp_task.csv")

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Initialize vectorizers
bow = CountVectorizer()
tfidf = TfidfVectorizer()

# Fit and transform
X_train_bow = bow.fit_transform(X_train)
X_test_bow = bow.transform(X_test)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train models
model_bow = LogisticRegression()
model_bow.fit(X_train_bow, y_train)

model_tfidf = LogisticRegression()
model_tfidf.fit(X_train_tfidf, y_train)

# Predict and evaluate
y_pred_bow = model_bow.predict(X_test_bow)
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)

acc_bow = accuracy_score(y_test, y_pred_bow)
acc_tfidf = accuracy_score(y_test, y_pred_tfidf)

f1_bow = f1_score(y_test, y_pred_bow, pos_label='positive')
f1_tfidf = f1_score(y_test, y_pred_tfidf, pos_label='positive')

# Print scores
print("Accuracy - BoW:", acc_bow)
print("F1 Score - BoW:", f1_bow)
print("Accuracy - TF-IDF:", acc_tfidf)
print("F1 Score - TF-IDF:", f1_tfidf)

# Plot
labels = ['Accuracy', 'F1 Score']
bow_scores = [acc_bow, f1_bow]
tfidf_scores = [acc_tfidf, f1_tfidf]

x = range(len(labels))
plt.bar(x, bow_scores, width=0.4, label='BoW', align='center')
plt.bar([p + 0.4 for p in x], tfidf_scores, width=0.4, label='TF-IDF', align='center')
plt.xticks([p + 0.2 for p in x], labels)
plt.ylabel('Score')
plt.title('BoW vs TF-IDF Comparison')
plt.legend()
plt.tight_layout()
plt.show()

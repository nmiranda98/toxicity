import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


train = pd.read_csv(r"C:\Users\chady\Downloads\jigsaw-unintended-bias-in-toxicity-classification\train.csv")
test = pd.read_csv(r"C:\Users\chady\Downloads\jigsaw-unintended-bias-in-toxicity-classification\test.csv")
submission_binary = pd.read_csv(r"C:\Users\chady\Downloads\jigsaw-unintended-bias-in-toxicity-classification\sample_submission.csv")

# Vectorizing the comments in the training and testing sets.

Vectorize = TfidfVectorizer()
X = Vectorize.fit_transform(train["comment_text"])
y = np.where(train['target'] >= 0.5, 1, 0)
test_X = Vectorize.transform(test["comment_text"])

# Splitting the training set into 80% training and 20% validation set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

log = LogisticRegression(solver='sag', max_iter=1000, n_jobs=-1)
log.fit(X_train, y_train)
y_pred = log.predict(X_test)

print(accuracy_score(y_test, y_pred))

# Predicting toxicity scores of comments in testing set.

predictions = log.predict_proba(test_X)[:,1]
submission_binary['prediction'] = predictions
print(submission_binary)
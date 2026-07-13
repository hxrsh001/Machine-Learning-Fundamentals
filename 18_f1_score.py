# Precision | Recall | f1 Score

import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

actual = [1,0,1,1,0,1,0,0]
pred = [1,0,1,0,0,1,1,0]

# Confusion Matrix-
cm = confusion_matrix(actual, pred)
print(cm, "\n")

# Precsion-
print(f"Precision: {3/(3+1)}")

precision = precision_score(actual, pred)
print(precision)
print("\n")

# Recall-
recall = recall_score(actual, pred)
print("Recall: ", recall)
print("\n")

# F1 Score-
f1 = f1_score(actual, pred)
print("F1 Score: ", f1)

# roc and auc curve

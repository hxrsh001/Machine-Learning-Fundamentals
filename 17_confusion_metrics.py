from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

actual = [1,0,1,1,0,1,0,0]
pred = [1,0,1,0,0,1,1,0]


ConfusionMatrixDisplay.from_predictions(actual, pred)
#cm = confusion_matrix(actual, pred)
#print(cm)

plt.show()
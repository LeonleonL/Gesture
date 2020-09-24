# ----------------------------------------------------------------------------------------------------------------------
# KNN training
# ----------------------------------------------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plt
from sklearn import neighbors
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import plot_confusion_matrix
# ----------------------------------------------------------------------------------------------------------------------

TRAINING_DATASET = "./data/out.csv"
df = pd.read_csv(TRAINING_DATASET, index_col=False, low_memory=False)

# ----------------------------------------------------------------------------------------------------------------------

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

# ----------------------------------------------------------------------------------------------------------------------
# Divided into training part and testing part, initialization, fitting
# ----------------------------------------------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = neighbors.KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, y_train)
knnPickle = open("./model/knnclassifier_file", "wb")
pickle.dump(clf, knnPickle)

# ----------------------------------------------------------------------------------------------------------------------
# Use test data to evaluate the fit effect
# ----------------------------------------------------------------------------------------------------------------------

score = clf.score(X_test, y_test)
print(score)

# ----------------------------------------------------------------------------------------------------------------------
# Confusion matrix
# ----------------------------------------------------------------------------------------------------------------------

y_pred = clf.predict(X_test)
print(y_test)
print(y_pred)

class_names = ['1', '2', '3', '4', '5']

disp = plot_confusion_matrix(clf, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize='pred')
plt.savefig('./out_knn.jpg')
plt.show()
out = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
print(out)



# ----------------------------------------------------------------------------------------------------------------------
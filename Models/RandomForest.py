import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


data = pd.read_csv('./randomforest.csv')
data = data.sample(frac=0.9)


X = data.drop(['SUBJECT_ID','TARGET'], axis=1)
y = data['TARGET']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)


for i, tree in enumerate(rf.estimators_[:3]):
    dot_data = export_graphviz(tree,
                               out_file=None,
                               feature_names=X_train.columns,
                               filled=True,
                               max_depth=2,
                               impurity=False,
                               proportion=True)
    graph = graphviz.Source(dot_data)
    graph.render(f'rf_tree_{i}', format='png', cleanup=True)
  
for i in range(3):
    img = mpimg.imread(f'rf_tree_{i}.png')
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.axis('off')  # Do not show axes to keep it tidy
    plt.show()
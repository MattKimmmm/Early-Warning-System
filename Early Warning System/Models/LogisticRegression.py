import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV


def lr_results():
    #data = pd.read_csv('./data.csv')
    data = pd.read_csv('./data_balanced.csv')
    data = data.sample(frac= 1)


    # feature_cols = ['AGE', 'GENDER', 'ETHNICITY', '50971', '51221', '50983', '50912', '50902' ]
    feature_cols = ['AGE', 'GENDER', 'ETHNICITY', '50971']
    X = data[feature_cols]
    y = data['TARGET']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    logreg = LogisticRegression(random_state=16, max_iter=100)
    logreg.fit(X_train_scaled, y_train)
    y_pred = logreg.predict(X_test_scaled)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)



    #print(y.value_counts())



    class_names = [0, 1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    plt.savefig('lr_confusion_matrix.png', bbox_inches='tight')
    plt.close()



    print(metrics.classification_report(y_test, y_pred))
    param_grid = {'C': np.logspace(-4, 4, 20), 'penalty': ['l1', 'l2']}
    logreg_cv = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
    logreg_cv.fit(X_train_scaled, y_train)
    #metrics.plot_roc_curve(logreg, X_test_scaled, y_test)
    #plt.show()



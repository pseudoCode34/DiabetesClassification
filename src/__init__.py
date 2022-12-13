import pydotplus
from IPython.display import Image
from six import StringIO
from sklearn.tree import export_graphviz
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

## Load the dataset
col_names = [
    "pregnant",
    "glucose",
    "bp",
    "skin",
    "insulin",
    "bmi",
    "pedigree",
    "age",
    "label",
]
PATH = "resources/pima-indians-diabetes.csv"
pima = pd.read_csv(PATH, header=0, names=col_names)

print(pima.head())
##

## Feature selection
# split dataset in features and target variable
feature_cols = [
    "pregnant",
    "insulin",
    "bmi",
    "age",
    "glucose",
    "bp",
    "pedigree",
]
X = pima[feature_cols]  # Features
y = pima.label  # Target variable
##

## Data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)
##

## Create and train Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3).fit(
    X_train, y_train
)

# Predict the response for test dataset
y_pred = clf.predict(X_test)
##

## Model evaluation
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
##

## Decision Tree visualisation
dot_data = StringIO()
export_graphviz(
    clf,
    out_file=dot_data,
    filled=True,
    rounded=True,
    special_characters=True,
    feature_names=feature_cols,
    class_names=["0", "1"],
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("resources/diabetes.png")
Image(graph.create_png())
##

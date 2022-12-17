import pickle
import pydotplus
from IPython.display import Image
from six import StringIO
import pandas as pd
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

from features import feature_cols

# Load the dataset
PATH = "resources/pima-indians-diabetes.csv"
data_set = pd.read_csv(PATH)
print(data_set.head())
##

# Feature selection
# split dataset in features and target variable
X = data_set[feature_cols]
y = data_set.Outcome
##

# Data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)
##

# Create and train Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3).fit(
    X_train, y_train
)

# Predict the response for test dataset
y_pred = clf.predict(X_test)
##

# Model evaluation
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
##

# Decision Tree visualisation
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

# now you can save it to a file
with open("src/diagnosis.pkl", "wb") as f:
    pickle.dump(clf, f)

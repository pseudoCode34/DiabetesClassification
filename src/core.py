# Save the trained model for further usage
import pickle

# Draw the decision tree
import pydotplus
from IPython.display import Image
from six import StringIO
from sklearn.tree import export_graphviz

# IO processing
import pandas as pd

# Process decision tree model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Evaluate the model's accuracy
from sklearn import metrics

from features import feature_cols


def load_data(PATH):
    """Load the dataset using pandas' read CSV function and print 5 first rows

    Parameters
    ----------
    PATH : str
        The directory of data set. It can be relative or absolute to the current
        directory, check via ```pwd```

    Returns
    ----------
    data_set : DataFrame
        Object containing data
    """
    data_set = pd.read_csv(PATH)
    print(data_set.head())
    return data_set


def feature_selection(feature_cols):
    """Split the dataset into features and target variable

    Parameters
    ----------
    feature_cols : list
        List of labels of the arguments(dependant variables) assgined

    Returns
    ----------
    health_status_measures : DataFrame
        List of arguements

    diabetes_states : DataFrame
        List of target variables(independant ones)
    """
    health_status_measures = data_set[feature_cols]
    diabetes_states = data_set.Outcome
    return health_status_measures, diabetes_states


# FIXME: How to refactor this to a new function? Should I?
# Data split
(
    train_arguments,
    test_arguments,
    train_results,
    test_results,
) = train_test_split(
    health_status_measures,
    diabetes_states,
    test_size=0.3,
    random_state=1,
)


def build_decision_tree_model(classifier, test_arguments):
    """Once create and train Decision Tree classifer object, it's time to
        predict the response of the test dataset

    Parameters
    ----------
    classifier : DecisionTreeClassifier object
        Create decision tree via entropy function and optimise it to contain at
        most 3 depths so that overfitting can be minimised. If max_depth is set
        to None(default), nodes are expanded until all the leaves contain less
        than min_samples_split samples. The higher value of maximum depth causes
        overfitting, and a lower value causes underfitting.

        Supported criteria are “gini”(default) for the Gini index and “entropy”
        for the information gain. Both are provided as differnt attribute
        selection measures.

    test_arguments: DataFrame
        A proportion of the orginal data set containing inputs for testing
        purpose only.

    Returns
    ----------
    data_set : DataFrame
        Object containing data
    """
    # FIXME : write docs (and modify this function to fit) 2 following variables
    classifier = DecisionTreeClassifier(criterion="entropy", max_depth=3).fit(
        train_arguments, train_results
    )

    predicted_responses = classifier.predict(test_arguments)
    return predicted_responses, classifier


def evaluate(actual_results, calculated_responses):
    """Estimate how accurately the classifier or model can predict the type of 
    cultivars.

    Parameters
    ----------
    actual_results, calculated_responses : DataFrame
        Accuracy can be calculated by setting the actual test set values and
        predicted values side by side.
    """
    accuracy = metrics.accuracy_score(actual_results, calculated_responses)
    print("Accuracy:", accuracy)


# FIXME: Should return the path to image and take dot_data as arguments or initialise a local variable image_dir?
def visualise(image_dir):
    export_graphviz(
        decision_tree,
        out_file=dot_data,
        filled=True,
        rounded=True,
        special_characters=True,
        feature_names=feature_cols,
        class_names=["0", "1"],
    )
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(image_dir)
    Image(graph.create_png())
    return graph


graph = visualise(dot_data, feature_cols)

if __name__ == "__main__":
    PATH = "resources/pima-indians-diabetes.csv"
    data_set = load_data(PATH)

    health_status_measures, diabetes_states = feature_selection(feature_cols)
    predicted_responses, classifier = build_decision_tree_model(
        classifier, test_arguments
    )

    evaluate(test_results, predicted_responses)

    dot_data = StringIO()
    with open("src/diagnosis.pkl", "wb") as file:
        pickle.dump(classifier, file)

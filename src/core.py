# -*- coding: utf-8 -*-
"""
File: core.py
Author: Nguyễn Khắc Trường
Email: truong_dev@icloud.com
Description: Build Decision Classfier model
Last Modified: 2022-12-27 19:00
"""

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


def preprocessing(data):
    """There are 3 main phases involved in preprocessing the data. First, it is
    required to clean the data crawled earlier. What if there is some missing
    and noisy data? It is highly recommended that tuple be discared and absent
    values be filled in manually by mean or the most probable value. To deal
    with noisy data, binning method, regression and clustering would be
    considered. Now, it is high time to transform the data, i.e, normalise,
    feature selection, generate concept hierachy. But, in this circumstance, it
    is redundant. Ultimately, we proceed to the very last step of preprocessing,
    data reduction. Either performing data cube arrgregation, selecting a subset
    of attributes, reducing numerosity or dropping redundant dimensionality.

    Parameters
    ----------
    data: DataFrame
        A set of data crawled earlier

    Returns
    -------
    train_arguments, train_results, test_arguments, test_results : DataFrame

    """
    if data.isnull().sum().sum() != 0:
        raise ValueError("There is some missing data in this data set")

    data.drop_duplicates(in_place=True)
    health_status_measures, diabetes_states = select_features_from(feature_cols)
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
    return train_arguments, train_results, test_arguments, test_results


def load_data(URL):
    """Load the dataset using pandas' read CSV function and print 5 first rows

    Parameters
    ----------
    URL : str
        The directory of data set. It can be relative or absolute to the current
        directory, check via ```pwd```

    Returns
    -------
    data: DataFrame
        Object containing data
    """
    data = pd.read_csv(URL)
    return data


def select_features_from(label):
    """Split the dataset into features and target variable

    Parameters
    ----------
    feature_cols : list
        List of labels of the arguments(dependant variables) assgined

    Returns
    -------
    health_status_measures, diabetes_states : DataFrame
        List of arguments(features) and target variables(independant ones)
    """
    health_status_measures = data_set[label]
    diabetes_states = data_set.Outcome
    return health_status_measures, diabetes_states


# FIXME: Could I delete classifier from paramlist?
def build_decision_tree_model(test_arguments):
    """Once create and train Decision Tree classifer object, it's time to
    predict the response of the test dataset

    Create decision tree via entropy function and optimise it to contain 3
    depths at most so that overfitting can be minimised. If max_depth is set to
    None(default), nodes are expanded until all the leaves contain less than
    min_samples_split samples. The higher value of maximum depth causes
    overfitting, and a lower value causes underfitting.

    Parameters
    ----------
    test_arguments: DataFrame
        A proportion of the orginal data set containing inputs for testing
        purpose only.

    Returns
    -------
    clf : DecisionTreeClassifier object
    predicted_results : DataFrame
    """

    # Supported criteria are “gini”(default) for the Gini index and “entropy”
    # for the information gain. Both are provided as differnt attribute
    # selection measures.
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=3).fit(
        train_arguments, train_results
    )

    predicted_results = clf.predict(test_arguments)
    return clf, predicted_results


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


def visualise(decision_tree, image_dir):
    """Draw the decision tree

    Parameters
    ----------
    decision_tree : DecisionTreeClassifier object

    image_dir : str
        Directory to export the image
    """
    dot_data = StringIO()
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


if __name__ == "__main__":
    URL = "resources/pima-indians-diabetes.csva"
    IMAGE_DIR = "resources/diabetes.png"

    data_set = load_data(URL)
    (
        train_arguments,
        train_results,
        test_arguments,
        test_results,
    ) = preprocessing(data_set)
    classifier, predicted_responses = build_decision_tree_model(test_arguments)
    evaluate(test_results, predicted_responses)
    visualise(classifier, IMAGE_DIR)

    # save decision tree model for further use, no need to rebuild
    with open("src/diagnosis.pkl", "wb") as file:
        pickle.dump(classifier, file)

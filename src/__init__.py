# Load the trained model
import pickle

# Deploy into a web interface
import streamlit as st

# Deal with the type arguments of classifier object
import numpy as np

# Crawl the reduced table's names
from features import feature_cols


def split_half(list):
    half = len(list) // 2
    return list[:half], list[half:]


def print(diagnosis: np.ndarray):
    if diagnosis[0] == 0:
        st.success("Bệnh nhân này không mắc tiểu đường")
    else:
        st.error("Bệnh nhân này mắc tiểu đường")


def input() -> list:
    left_column, right_column = split_half(feature_cols)
    col1, col2 = st.columns(2)
    with col1:
        left_input = [st.text_input(col) for col in left_column]
    with col2:
        right_input = [st.text_input(col) for col in right_column]

    return left_input + right_input


if __name__ == "__main__":
    with open("src/diagnosis.pkl", "rb") as f:
        classifier = pickle.load(f)

    title = st.title("Chẩn đoán nhân mắc tiểu đường")

    patient_health_indexes = input()
    if st.button("Chẩn đoán"):
        data = np.array(patient_health_indexes).reshape(1, -1)
        diagnosis = classifier.predict(data)
        print(diagnosis)

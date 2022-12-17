import pickle
import streamlit as st
import numpy as np
from features import feature_cols


def split_list(a_list):
    half = len(a_list) // 2
    return a_list[:half], a_list[half:]


def print_diagnosis(result: np.ndarray):
    if result[0] == 0:
        st.success("Bệnh nhân này không mắc tiểu đường")
    else:
        st.error("Bệnh nhân này mắc tiểu đường")


def input() -> list:

    first_column_half, last_column_half = split_list(feature_cols)
    col1, col2 = st.columns(2)
    with col1:
        left_input = [st.text_input(col) for col in first_column_half]
    with col2:
        right_input = [st.text_input(col) for col in last_column_half]
    return left_input + right_input


if __name__ == "__main__":
    with open("src/diagnosis.pkl", "rb") as f:
        classifier = pickle.load(f)

    title = st.title("Chẩn đoán nhân mắc tiểu đường")

    test_result = input()
    if st.button("chẩn đoán"):
        data = np.array(test_result).reshape(1, -1)
        result = classifier.predict(data)
        print_diagnosis(result)

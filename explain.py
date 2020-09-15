import joblib
import pandas as pd
import shap
from anchor.anchor_tabular import AnchorTabularExplainer
from pathlib import Path
import numpy as np
import streamlit as st
import streamlit.components.v1 as componenets

categorical_columns = ['island', 'gender']
numerical_columns = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']


@st.cache
def get_penguins():
    return pd.read_csv("penguins.csv",
                       dtype={
                           "species": 'category',
                           "island": 'category',
                           "gender": 'category'
                       })


penguins = get_penguins()


@st.cache
def get_metadata():
    metadata = {}

    for cat_col in categorical_columns:
        metadata[f'{cat_col}_labels'] = np.unique(penguins[cat_col]).tolist()

    for num_col in numerical_columns:
        metadata[f"{num_col}_min"] = penguins[num_col].min()
        metadata[f"{num_col}_max"] = penguins[num_col].max()

    metadata['class_labels'] = np.unique(penguins['species']).tolist()
    metadata['all_columns'] = penguins.columns.drop('species').tolist()
    return metadata


@st.cache
def get_clf():
    return joblib.load('penguin_clf.joblib')


metadata = get_metadata()
user_input = {}

for col in categorical_columns:
    user_input[col] = [st.sidebar.radio(f"Select {col}", metadata[f"{col}_labels"])]


step_map = {
    'culmen_length_mm': 5.0,
    'culmen_depth_mm': 2.0,
    'flipper_length_mm': 10.0,
    'body_mass_g': 200.0
}
for col in numerical_columns:
    user_input[col] = [st.sidebar.number_input(f"Select {col}",
                                               min_value=metadata[f"{col}_min"],
                                               max_value=metadata[f"{col}_max"],
                                               step=step_map[col])]

clf = joblib.load("penguin_clf.joblib")
class_names = metadata['class_labels']
all_columns = metadata['all_columns']

user_df = pd.DataFrame(user_input, columns=all_columns)

prediction = clf.predict(user_df)[0]
class_prediction = class_names[prediction]
st.sidebar.write(f"## Prediction: {class_prediction}")
proba = clf.predict_proba(user_df)
proba_df = pd.DataFrame(proba, columns=class_names)
st.sidebar.write(proba_df)

X_encoded = clf[:-1].transform(user_df)
encoded_columns = categorical_columns + numerical_columns
X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_columns)
explainer = shap.TreeExplainer(clf[-1])
shap_values = explainer.shap_values(X_encoded[[0], :], check_additivity=False)

st.write(f"## Explanation for Predicting: **{class_prediction}**")
st.subheader("SHAP values")

@st.cache
def bundle_js():
    bundle_path = Path(shap.__file__).parent / 'plots' / 'resources' / "bundle.js"
    with bundle_path.open('r') as f:
        bundle_data = f.read()
        bundle_js = f"<script charset='utf-8'>{bundle_data}</script>"
    return bundle_js


shape_js = bundle_js()

values1 = shap.force_plot(explainer.expected_value[0],
                          shap_values[0],
                          X_encoded_df,
                          out_names=class_names[0])
values2 = shap.force_plot(explainer.expected_value[1],
                          shap_values[1],
                          X_encoded_df,
                          out_names=class_names[1])
values3 = shap.force_plot(explainer.expected_value[2],
                          shap_values[2],
                          X_encoded_df,
                          out_names=class_names[2])
componenets.html(
    f"{shape_js}"
    f"{values1._repr_html_()}"
    f"{values2._repr_html_()}"
    f"{values3._repr_html_()}",
    height=420)

st.header("Anchors")


@st.cache
def get_anchor_explainder():
    X_encoded_all = clf[:-1].transform(penguins[all_columns])
    anchor_explainer = AnchorTabularExplainer(class_names,
                                              encoded_columns,
                                              X_encoded_all,
                                              categorical_names={
                                                  0: metadata['island_labels'],
                                                  1: metadata['gender_labels']
                                              })
    return anchor_explainer


anchor_explainer = get_anchor_explainder()
exp = anchor_explainer.explain_instance(X_encoded[0, :],
                                        clf[-1].predict, threshold=0.95)

exp_html = exp.as_html()
componenets.html(exp_html, height=1000)

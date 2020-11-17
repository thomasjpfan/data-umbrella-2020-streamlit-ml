import joblib
import pandas as pd
import shap
from anchor.anchor_tabular import AnchorTabularExplainer
from pathlib import Path
import numpy as np
import streamlit as st
import streamlit.components.v1 as componenets

categorical_columns = ['island', 'gender']
numerical_columns = [
    'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g'
]

penguins = pd.read_csv("penguins.csv")

metadata = {}

for cat_col in categorical_columns:
    metadata[f'{cat_col}_labels'] = np.unique(penguins[cat_col]).tolist()

for num_col in numerical_columns:
    metadata[f"{num_col}_min"] = penguins[num_col].min()
    metadata[f"{num_col}_max"] = penguins[num_col].max()

metadata['class_labels'] = np.unique(penguins['species']).tolist()
metadata['all_columns'] = penguins.columns.drop('species').tolist()

user_input = {}

for col in categorical_columns:
    user_input[col] = [
        st.sidebar.radio(f"Select {col}", metadata[f"{col}_labels"])
    ]

step_map = {
    'culmen_length_mm': 5.0,
    'culmen_depth_mm': 2.0,
    'flipper_length_mm': 10.0,
    'body_mass_g': 200.0
}
for col in numerical_columns:
    user_input[col] = [
        st.sidebar.number_input(f"Select {col}",
                                min_value=metadata[f"{col}_min"],
                                max_value=metadata[f"{col}_max"],
                                step=step_map[col])
    ]

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
explainer = shap.TreeExplainer(clf[-1])
shap_values = explainer.shap_values(X_encoded[[0], :], check_additivity=False)

st.write(f"## Explanation for Predicting: **{class_prediction}**")
st.subheader("SHAP values")

bundle_path = Path(shap.__file__).parent / 'plots' / 'resources' / "bundle.js"
with bundle_path.open('r') as f:
    bundle_data = f.read()
    shape_js = f"<script charset='utf-8'>{bundle_data}</script>"

feature_names = categorical_columns + numerical_columns

shap_plots = []

for i in range(3):
    shap_plot = shap.force_plot(explainer.expected_value[i],
                                shap_values[i],
                                X_encoded,
                                feature_names=feature_names,
                                out_names=class_names[i])
    shap_plots.append(shap_plot)

shap_html_reprs = [f"{shap_plot._repr_html_()}" for shap_plot in shap_plots]
shap_html_repr = "".join(shap_html_reprs)

componenets.html(f"{shape_js}{shap_html_repr}", height=420)

st.header("Anchors")

X_encoded_all = clf[:-1].transform(penguins[all_columns])
anchor_explainer = AnchorTabularExplainer(class_names,
                                          feature_names,
                                          X_encoded_all,
                                          categorical_names={
                                              0: metadata['island_labels'],
                                              1: metadata['gender_labels']
                                          })

exp = anchor_explainer.explain_instance(X_encoded[0, :],
                                        clf[-1].predict,
                                        threshold=0.95)

exp_html = exp.as_html()
componenets.html(exp_html, height=700)

import joblib
import pandas as pd
import shap
from anchor.anchor_tabular import AnchorTabularExplainer
from pathlib import Path
import numpy as np
import streamlit as st
import streamlit.components.v1 as componenets

st.set_page_config(initial_sidebar_state='expanded')

categorical_columns = ['island', 'gender']
numerical_columns = [
    'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g'
]
feature_names = categorical_columns + numerical_columns

penguins = pd.read_csv("penguins.csv")
X = penguins[feature_names]
y = penguins['species']

metadata = {}

for cat_col in categorical_columns:
    metadata[f'{cat_col}_categories'] = np.unique(penguins[cat_col]).tolist()

for num_col in numerical_columns:
    metadata[f"{num_col}_min"] = penguins[num_col].min()
    metadata[f"{num_col}_max"] = penguins[num_col].max()

class_names = np.unique(penguins['species']).tolist()

user_input = []

for col in categorical_columns:
    col_value = st.sidebar.radio(f"Select {col}",
                                 metadata[f"{col}_categories"])
    user_input.append(col_value)

step_map = {
    'culmen_length_mm': 5.0,
    'culmen_depth_mm': 2.0,
    'flipper_length_mm': 10.0,
    'body_mass_g': 200.0
}

for col in numerical_columns:
    num_value = st.sidebar.number_input(
        f"Select {col}",
        min_value=metadata[f"{col}_min"],
        max_value=metadata[f"{col}_max"],
        step=step_map[col])
    user_input.append(num_value)

user_df = pd.DataFrame([user_input], columns=feature_names)

clf = joblib.load("penguin_clf.joblib")
prediction = clf.predict(user_df)[0]
class_prediction = class_names[prediction]

st.sidebar.write(f"## Prediction: {class_prediction}")
proba = clf.predict_proba(user_df)
proba_df = pd.DataFrame(proba, columns=class_names)
st.sidebar.write(proba_df)

st.write(f"## Explanation for Predicting: **{class_prediction}**")
st.subheader("SHAP values")

user_encoded = clf[:-1].transform(user_df)
explainer = shap.TreeExplainer(clf[-1])
shap_values = explainer.shap_values(user_encoded[[0], :],
                                    check_additivity=False)

shap_plot_reprs = []

for i in range(3):
    shap_plot = shap.force_plot(explainer.expected_value[i],
                                shap_values[i],
                                user_encoded,
                                feature_names=feature_names,
                                out_names=class_names[i])
    shap_plot_reprs.append(shap_plot._repr_html_())

shap_html_repr = "".join(shap_plot_reprs)

bundle_path = Path(shap.__file__).parent / 'plots' / 'resources' / "bundle.js"
with bundle_path.open('r') as f:
    bundle_data = f.read()
    shape_js = f"<script charset='utf-8'>{bundle_data}</script>"

componenets.html(f"{shape_js}{shap_html_repr}", height=420)

st.header("Anchors")

X_encoded = clf[:-1].transform(X)
anchor_explainer = AnchorTabularExplainer(
    class_names, feature_names, X_encoded,
    categorical_names={0: metadata['island_categories'],
                       1: metadata['gender_categories']})

exp = anchor_explainer.explain_instance(user_encoded[0, :],
                                        clf[-1].predict)
exp_html = exp.as_html()
componenets.html(exp_html, height=700)

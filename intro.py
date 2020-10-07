import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Show me some penguins!")

species = st.radio("Select a penguin", ["adelie", "gentoo", "chinstrap"])

st.image(f"{species}.jpg", use_column_width=True)

species_to_wiki = {
    'adelie': 'https://en.wikipedia.org/wiki/Adélie_penguin',
    'gentoo': 'https://en.wikipedia.org/wiki/Gentoo_penguin',
    'chinstrap': 'https://en.wikipedia.org/wiki/Chinstrap_penguin'
}

st.markdown(f"### Learn more about {species} penguins at "
            f"[wikipedia]({species_to_wiki[species]})!")

st.title("EDA for penguins dataset!")

penguins = pd.read_csv("penguins.csv")

st.header("Lets see the dataset")
penguins

st.header("How much of each species are in our dataset?")

fig, ax = plt.subplots()
penguins['species'].value_counts().plot.bar(ax=ax,
                                            color=['blue', 'red', 'green'])

fig

st.header("Can flipper length be used to distingushed between species?")

flipper_fig = sns.displot(penguins, x='flipper_length_mm', hue='species',
                          kind='kde')

flipper_fig.fig

st.header("Is the culmen useful for classifying species?")

scat = px.scatter(penguins,
                  x="culmen_length_mm",
                  y="culmen_depth_mm",
                  marginal_y="violin",
                  marginal_x='box',
                  color='species')
scat

st.header("How is the body mass for each species?")

box = px.box(penguins, x="species", y="body_mass_g", color="gender")
box

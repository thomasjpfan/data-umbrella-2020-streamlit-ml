# Streamlit Talk for Data Umbrella 2020 [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/thomasjpfan/data-umbrella-2020-streamlit-ml/explain.py)

- [Link to slides](https://thomasjpfan.github.io/data-umbrella-2020-streamlit-slides/#1)
- [Hosted version of explain.py](https://share.streamlit.io/thomasjpfan/data-umbrella-2020-streamlit-ml/explain.py)

Simple dashboard using SHAP values and Anchors for explanations for predictions:

![Demo](demo.gif)

## Usage

0. Install [anaconda](https://www.anaconda.com/products/individual).

1. Create a virtual environment:

```bash
conda create -n data-umbrella-streamlit python=3.8
conda activate data-umbrella-streamlit
```
2. Clone this repository

```bash
git clone https://github.com/thomasjpfan/data-umbrella-2020-streamlit-ml.git
cd data-umbrella-2020-streamlit-ml
```

3. Install requirements:

```bash
pip install -r requirements-all.txt
```

4. Run the intro:

```bash
streamlit run intro.py --server.runOnSave True
```

5. Run the explanation:

```py
streamlit run explain.py --server.runOnSave True
```

## License

This repo is under the [MIT License](LICENSE).

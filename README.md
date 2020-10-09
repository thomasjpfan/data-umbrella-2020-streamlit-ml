# Streamlit Talk for Data Umbrella 2020

Simple dashboard using SHAP values and Anchors for explanations for predictions:

![Demo](demo.gif)

## Usage

0. Install [anaconda](https://www.anaconda.com/products/individual).

1. Create a virtual environment:

```bash
conda create -n data-umbrella-streamlit python=3.8
```
2. Clone this repository

```bash
git clone https://github.com/thomasjpfan/data-umbrella-2020-streamlit-ml.git 
```

3. Install requirements:

```bash
conda activate data-umbrella-streamlit
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

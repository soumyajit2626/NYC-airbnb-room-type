
# NYC Airbnb Room-Type Predictor

An interactive machine learning web application that predicts
the room type of a New York City Airbnb listing based on its
price, location, availability, minimum nights, and review data.

## Features

- Predicts Airbnb room types using a trained ML model.
- Displays prediction confidence.
- Interactive 3D map of NYC listings.
- 3D visualization of price and location.
- Confidence comparison chart.
- Interactive controls for listing characteristics.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly
- PyDeck

## Project Files

- `app.py` — Streamlit application
- `Model_1.pkl` — Trained classification model
- `ohe.pkl` — One-hot encoder
- `scaler.pkl` — Feature scaler
- `meta.json` — Neighbourhood metadata and sample listings
- `columns.pkl` — Saved columns artifact
- `nyc_airbnb_room_type_classification(5).ipynb` — Jupyter notebook

## Run Locally

1. Install the dependencies:

   `pip install -r requirements.txt`

2. Start the application:

   `streamlit run app.py`

3. Open the local URL printed in your terminal.

## Disclaimer

This project is for educational and demonstration purposes.
Predictions depend on the trained model and the input data.

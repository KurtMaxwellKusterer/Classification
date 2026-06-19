import streamlit as st
import numpy as np
import joblib
from pathlib import Path

BASE = Path(__file__).parent

model     = joblib.load(BASE / 'model.pkl')
scaler    = joblib.load(BASE / 'scaler.pkl')
le_region = joblib.load(BASE / 'le_region.pkl')
le_variety = joblib.load(BASE / 'le_variety.pkl')

REGIONS   = [r for r in le_region.classes_ if r.strip()]   # drop the rogue ' Chilean' duplicate
VARIETIES = list(le_variety.classes_)

st.set_page_config(page_title='White Wine Quality Predictor', page_icon='🍷', layout='centered')

st.title('🍷 White Wine Quality Predictor')
st.markdown('Enter the details from the wine label to predict whether the wine is **Good** or **Bad** quality.')
st.divider()

col1, col2 = st.columns(2)

with col1:
    region = st.selectbox('Region', sorted(REGIONS))
    variety = st.selectbox('Grape Variety', VARIETIES)

with col2:
    year = st.number_input('Vintage Year', min_value=1983, max_value=2019, value=2015, step=1)
    price = st.number_input('Price (€)', min_value=10.0, max_value=10000.0, value=50.0, step=0.5,
                            help='Bottle price in euros')

st.divider()

if st.button('Predict Quality', use_container_width=True, type='primary'):
    region_enc  = le_region.transform([region])[0]
    variety_enc = le_variety.transform([variety])[0]

    features = np.array([[year, region_enc, variety_enc, price]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    confidence = probability[prediction] * 100

    if prediction == 1:
        st.success(f'### Good Quality Wine  ✅')
        st.markdown(f'The model rates this wine as **good quality** with **{confidence:.1f}% confidence**.')
    else:
        st.error(f'### Below Average Quality  ❌')
        st.markdown(f'The model rates this wine as **below average** with **{confidence:.1f}% confidence**.')

    st.divider()
    col_a, col_b = st.columns(2)
    col_a.metric('Good Quality Probability',  f"{probability[1]*100:.1f}%")
    col_b.metric('Below Average Probability', f"{probability[0]*100:.1f}%")

    with st.expander('What did the model use?'):
        st.markdown(f"""
| Feature | Value |
|---|---|
| Region | {region} |
| Grape Variety | {variety} |
| Vintage Year | {year} |
| Price | €{price:.2f} |
""")

st.divider()
st.caption('Model: Random Forest trained on 4,200+ white wines · Features: Price, Vintage Year, Region, Variety')

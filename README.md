# Wine Quality Classification & Prediction

This repository contains two phases of a machine learning project centred on predicting white wine quality — starting from a classification experiment and progressing to a consumer-facing prediction app.

---

## Phase 1 — Classification Experiment (`Classification.ipynb`)

The first phase uses the UCI Wine Quality dataset to explore and compare multiple classification algorithms. The goal is to understand which features drive wine quality and which model performs best before building a real application.

### Dataset
- **Red wine:** 1,599 samples | **White wine:** 4,898 samples
- 11 physicochemical features per wine (alcohol, acidity, sulphates, pH etc.)
- Target variable: quality score (0–10) converted to a binary label — **Good (1)** if score ≥ 7, otherwise **Bad (0)**

### What the notebook does

**1. Data Loading & Exploration**
The dataset is loaded and inspected — checking data types, null values, and the distribution of quality scores across the samples.

**2. Data Visualisation**
Bar plots compare each physicochemical feature against quality, giving an early indication of which features are likely to be most predictive (e.g. alcohol content shows a clear positive relationship with quality).

**3. Data Preprocessing**
- The quality score is converted into a binary label (Good / Bad)
- Features are normalised using `MinMaxScaler` fitted only on the training data to prevent data leakage

**4. Train / Test Split**
The dataset is split 80% training / 20% test, with a fixed random seed to ensure reproducibility.

**5. Model Training & Evaluation**
Three classifiers are trained and compared:

| Model | Description |
|---|---|
| **Random Forest** | An ensemble of 200 decision trees — the best performing model |
| **SGD Classifier** | A linear model trained using stochastic gradient descent |
| **Support Vector Machine (SVM)** | Finds the optimal boundary between Good and Bad wines |

Each model is evaluated using accuracy, ROC-AUC score, and a full classification report (precision, recall, F1) to account for the class imbalance between Good and Bad wines.

**6. Cross Validation**
The Random Forest is evaluated using 40-fold cross validation, reporting the mean and standard deviation across folds — a more reliable measure of generalisation than a single train/test split.

**7. Hyperparameter Tuning — Randomized Search**
`RandomizedSearchCV` is used to search across a range of SVM hyperparameter combinations, sampling 50 random configurations rather than exhaustively testing every combination (as Grid Search would). This finds strong parameters faster, particularly useful as the number of features grows.

**8. Feature Importance**
The Random Forest is used to rank all 11 features by how much they contributed to predictions. A horizontal bar chart visualises the results. Alcohol content consistently ranks as the most important feature across both red and white wine datasets.

---

## Phase 2 — App Model Pipeline (`WineQualityApp.ipynb`)

The second phase shifts focus from lab measurements to consumer-facing features — things a person can actually read from a wine bottle or find in a shop. A new dataset of 4,594 white wines is used, containing region, grape variety, vintage year, and price alongside community ratings from Vivino.

### Dataset
- 4,594 white wines with ratings, pricing, region and grape variety
- Target variable: `WineRating` converted to a binary label — **Good (1)** if rating ≥ 4.3, otherwise **Bad (0)**
- Resulting split: ~70% Bad, ~30% Good

### What the notebook does

**1. Data Cleaning**
- Vintage year is stripped of whitespace and non-vintage entries (`N.V.`) are removed
- Rows with missing region or variety are dropped
- Final clean dataset: 4,208 wines

**2. Exploratory Data Analysis**
Three charts explore the relationship between the consumer-facing features and quality:
- Price vs Quality (box plot — higher priced wines skew Good)
- Good wine rate by Region
- Good wine rate by Grape Variety

**3. Feature Engineering**
Text columns (Region and RegionalVariety) are converted to numbers using `LabelEncoder` so the model can process them. The four final features used are:

| Feature | Why it was chosen |
|---|---|
| **WinePrice** | Strongest predictor — accounts for 76% of model decisions |
| **Year** | Vintage year — older vintages from good years score higher |
| **Region** | Geographic origin is a strong quality signal |
| **RegionalVariety** | Grape variety influences style and quality potential |

**4. Model Training & Tuning**
A Random Forest is trained on the four features and tuned using `RandomizedSearchCV` optimising for ROC-AUC across 50 parameter combinations. The tuned model achieves:
- **Accuracy: 84.4%**
- **ROC-AUC: 91.6%**

**5. Saving Model Artefacts**
Four files are saved using `joblib` so the app can load them without retraining:
- `model.pkl` — the trained Random Forest
- `scaler.pkl` — the fitted MinMaxScaler
- `le_region.pkl` — the Region label encoder
- `le_variety.pkl` — the Variety label encoder

---

## Phase 3 — Streamlit App (`wine_quality_app.py`)

A lightweight web application that lets anyone predict the quality of a white wine using only information available on the bottle or in a shop.

### How to run it

Open a terminal in the project folder and run:

```
python -m streamlit run wine_quality_app.py
```

The app will open in your browser at `http://localhost:8501`.

### How it works

The user selects a **Region**, **Grape Variety**, **Vintage Year** and enters a **Price**. When they click **Predict Quality**, the app:

1. Encodes the Region and Variety using the saved label encoders
2. Scales the four inputs using the saved scaler
3. Passes the scaled inputs to the saved Random Forest model
4. Returns a **Good ✅** or **Bad ❌** prediction with a confidence percentage and probability breakdown

---

## Repository Structure

```
Classification/
│
├── Classification.ipynb         # Phase 1 — classification experiment (red & white wine)
├── WineQualityApp.ipynb         # Phase 2 — app model pipeline (white wine price/rating)
├── wine_quality_app.py          # Phase 3 — Streamlit web application
│
├── winequality-red.csv          # UCI red wine dataset
├── winequality-white.csv        # UCI white wine dataset
├── white-wine-price-rating.csv  # Vivino white wine price & rating dataset
│
├── model.pkl                    # Trained Random Forest model
├── scaler.pkl                   # Fitted MinMaxScaler
├── le_region.pkl                # Region label encoder
└── le_variety.pkl               # Variety label encoder
```

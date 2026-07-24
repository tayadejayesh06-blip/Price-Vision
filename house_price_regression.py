"""
TASK 01 - Linear Regression: House Price Prediction
=====================================================
Predicts house prices based on:
  - Square footage (GrLivArea)
  - Number of bedrooms (BedroomAbvGr)
  - Number of bathrooms (FullBath + HalfBath)

Works out of the box with a bundled synthetic dataset, OR with the real
Kaggle "House Prices - Advanced Regression Techniques" train.csv if you
place it at ./train.csv (it auto-detects and uses it instead).

Author: Generated for SkillCraft Technology Internship - Task 01
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def inr(n):
    """Format a number in Indian digit grouping with a rupee sign, e.g. ₹12,34,567."""
    n = round(float(n))
    sign = "-" if n < 0 else ""
    n = abs(n)
    s = str(n)
    if len(s) <= 3:
        grouped = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        grouped = ",".join(parts) + "," + last3
    return f"{sign}\u20b9{grouped}"

# ----------------------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------------------
RANDOM_STATE = 42
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

KAGGLE_CSV_PATH = "train.csv"   # Kaggle House Prices dataset, if you have it


# ----------------------------------------------------------------------
# 2. LOAD DATA
# ----------------------------------------------------------------------
def load_kaggle_dataset(path):
    """Load and map the real Kaggle House Prices dataset to our 3 features."""
    df = pd.read_csv(path)
    df = df.rename(columns={
        "GrLivArea": "SquareFootage",
        "BedroomAbvGr": "Bedrooms",
        "SalePrice": "Price"
    })
    df["Bathrooms"] = df.get("FullBath", 0) + 0.5 * df.get("HalfBath", 0)
    df = df[["SquareFootage", "Bedrooms", "Bathrooms", "Price"]].dropna()
    return df


def generate_synthetic_dataset(n=1500, seed=RANDOM_STATE):
    """
    Generate a realistic synthetic housing dataset so the pipeline
    runs end-to-end even without the Kaggle CSV on disk.
    Relationship mimics real housing economics with added noise.
    """
    rng = np.random.default_rng(seed)

    sqft = rng.normal(1800, 650, n).clip(450, 6000)
    bedrooms = rng.integers(1, 6, n)
    bathrooms = np.round(rng.uniform(1, 4, n) * 2) / 2  # allows .5 baths

    # Base price formula + noise (loosely realistic ₹/sqft economics for
    # a mid-tier Indian city — adjust PRICE_PER_SQFT for your city/locality)
    price = (
        15_00_000
        + sqft * 4_500
        + bedrooms * 3_50_000
        + bathrooms * 5_00_000
        + rng.normal(0, 8_00_000, n)
    ).clip(15_00_000, None)

    df = pd.DataFrame({
        "SquareFootage": sqft.round(0),
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Price": price.round(0)
    })
    return df


if os.path.exists(KAGGLE_CSV_PATH):
    print(f"Found {KAGGLE_CSV_PATH} — using real Kaggle House Prices dataset.")
    data = load_kaggle_dataset(KAGGLE_CSV_PATH)
else:
    print("No train.csv found — using a generated synthetic dataset instead.")
    print("(To use the real dataset: download Kaggle's 'House Prices - "
          "Advanced Regression Techniques' train.csv and place it in this folder.)")
    data = generate_synthetic_dataset()

print(f"\nDataset shape: {data.shape}")
print(data.head())
print("\nSummary statistics:")
print(data.describe())


# ----------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.scatterplot(data=data, x="SquareFootage", y="Price", ax=axes[0, 0], alpha=0.5)
axes[0, 0].set_title("Price vs Square Footage")

sns.boxplot(data=data, x="Bedrooms", y="Price", ax=axes[0, 1])
axes[0, 1].set_title("Price vs Bedrooms")

sns.boxplot(data=data, x="Bathrooms", y="Price", ax=axes[1, 0])
axes[1, 0].set_title("Price vs Bathrooms")

corr = data.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1, 1])
axes[1, 1].set_title("Correlation Heatmap")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_overview.png"), dpi=150)
plt.close()
print(f"\nSaved EDA plots -> {OUTPUT_DIR}/eda_overview.png")


# ----------------------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ----------------------------------------------------------------------
FEATURES = ["SquareFootage", "Bedrooms", "Bathrooms"]
TARGET = "Price"

X = data[FEATURES]
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)


# ----------------------------------------------------------------------
# 5. TRAIN THE LINEAR REGRESSION MODEL
# ----------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("\n" + "=" * 50)
print("MODEL COEFFICIENTS")
print("=" * 50)
for feature, coef in zip(FEATURES, model.coef_):
    print(f"  {feature:15s}: {coef:,.2f}  (price change per unit)")
print(f"  {'Intercept':15s}: {model.intercept_:,.2f}")


# ----------------------------------------------------------------------
# 6. EVALUATE THE MODEL
# ----------------------------------------------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE (on held-out test set)")
print("=" * 50)
print(f"  MAE  (Mean Absolute Error) : {inr(mae)}")
print(f"  MSE  (Mean Squared Error)  : {mse:,.2f}")
print(f"  RMSE (Root Mean Sq. Error) : {inr(rmse)}")
print(f"  R^2 Score                  : {r2:.4f}  ({r2*100:.1f}% variance explained)")


# ----------------------------------------------------------------------
# 7. VISUALIZE PREDICTIONS
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.5, color="teal")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
axes[0].plot(lims, lims, 'r--', label="Perfect prediction")
axes[0].set_xlabel("Actual Price")
axes[0].set_ylabel("Predicted Price")
axes[0].set_title("Actual vs Predicted Price")
axes[0].legend()

# Residuals
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.5, color="orange")
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_xlabel("Predicted Price")
axes[1].set_ylabel("Residual (Actual - Predicted)")
axes[1].set_title("Residual Plot")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_evaluation.png"), dpi=150)
plt.close()
print(f"\nSaved evaluation plots -> {OUTPUT_DIR}/model_evaluation.png")


# ----------------------------------------------------------------------
# 8. SAVE THE TRAINED MODEL
# ----------------------------------------------------------------------
model_path = os.path.join(OUTPUT_DIR, "linear_regression_house_price_model.pkl")
joblib.dump(model, model_path)
print(f"Saved trained model -> {model_path}")


# ----------------------------------------------------------------------
# 9. PREDICT ON NEW HOUSES
# ----------------------------------------------------------------------
def predict_price(square_footage, bedrooms, bathrooms, trained_model=model):
    """Predict the price of a house given its features."""
    input_df = pd.DataFrame([{
        "SquareFootage": square_footage,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms
    }])
    return trained_model.predict(input_df)[0]


print("\n" + "=" * 50)
print("SAMPLE PREDICTIONS ON NEW HOUSES")
print("=" * 50)
sample_houses = [
    (1500, 3, 2),
    (2500, 4, 3),
    (900, 2, 1),
    (3800, 5, 4),
]
for sqft, beds, baths in sample_houses:
    price = predict_price(sqft, beds, baths)
    print(f"  {sqft} sqft, {beds} bed, {baths} bath  ->  {inr(price)}")

print("\nDone. All outputs saved in the 'outputs/' folder.")

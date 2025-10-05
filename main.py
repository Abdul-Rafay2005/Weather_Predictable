import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)



df = pd.read_csv("C:\\Users\\kk\\Downloads\\sampe\\NASA prediction final\\nyc_training_data.csv")


# Basic testing
print(df.info())
print(df.head())

#No null present or duplicates
print(df.isnull().sum())
print("Number of duplicates", df.duplicated().sum())

#Converting date to proper format
df['date'] = pd.to_datetime(df['date'])

# ---EDA (Part 1)---
print(df.describe())


'''# Boxplots
sns.boxplot(x=df['tmax'])
plt.show()

sns.boxplot(x=df['precip'])
plt.show()'''


# Target: Rain amount
y_precip = df['precip']

# Features (you can add more engineered features later)
X = df[['tmax', 'tmin', 'wind', 'rh', 'doy']]

# Step 1: Create Rain/No Rain flag
df['rain_flag'] = (df['precip'] > 0).astype(int)

# ----------------------
# Split into Train/Test
# ----------------------
X_train, X_test, y_train_flag, y_test_flag = train_test_split(
    X, df['rain_flag'], test_size=0.2, shuffle=False
)

# ----------------------
# Step 2A: Classification (Rain vs No Rain)
# ----------------------
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ----------------------
# Rain / No Rain Prediction (Classification)
# ----------------------

df['rain_flag'] = (df['precip'] > 0).astype(int)

X = df[['tmax', 'tmin', 'wind', 'rh', 'doy']]
y = df['rain_flag']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Model
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Metrics
print("\n--- Rain Prediction (Yes/No) ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Rain", "Rain"])
disp.plot(cmap="Blues")
plt.title("Rain Prediction Confusion Matrix")
plt.show()

# ----------------------
# Step 2B: Regression for Rain Amount (only rainy days)
# ----------------------
# Use log-transform to stabilize rainfall outliers
rainy = df[df['precip'] > 0].copy()
rainy['precip_log'] = np.log1p(rainy['precip'])

X_rain = rainy[['tmax', 'tmin', 'wind', 'rh', 'doy']]
y_rain = rainy['precip_log']

X_train_rain, X_test_rain, y_train_rain, y_test_rain = train_test_split(
    X_rain, y_rain, test_size=0.2, shuffle=False
)

reg = RandomForestRegressor(n_estimators=200, random_state=42)
reg.fit(X_train_rain, y_train_rain)

y_pred_rain_log = reg.predict(X_test_rain)
y_pred_rain = np.expm1(y_pred_rain_log)  # invert log

print("\nRain Amount Regression Metrics (for rainy days only):")
print("R2:", r2_score(np.expm1(y_test_rain), y_pred_rain))
print("RMSE:", np.sqrt(mean_squared_error(np.expm1(y_test_rain), y_pred_rain)))


# ----------------------
# Combined Approach (Optional):
# Use classifier to decide if rain happens, then regressor for amount
# ----------------------
def combined_predict(X_row):
    rain_flag_pred = clf.predict([X_row])[0]
    if rain_flag_pred == 0:
        return 0.0
    else:
        rain_amount_log = reg.predict([X_row])[0]
        return np.expm1(rain_amount_log)

# Example test on first 5 test rows
print("\nCombined Predictions (first 5 test rows):")
for i in range(5):
    print("Actual:", y_precip.iloc[len(X_train)+i], 
          "Predicted:", combined_predict(X_test.iloc[i].values))
    
# ----------------------
# Step 3: Predict All Weather Variables
# ----------------------

targets = ['tmax', 'tmin', 'wind', 'rh', 'precip']
features_all = ['doy', 'tmax', 'tmin', 'wind', 'rh']  # you can tune this

results = {}

print("\n--- Overall Weather Prediction Models ---")

for target in targets:
    # Create feature set excluding the target itself
    X = df[[f for f in features_all if f != target]]
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Model
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    results[target] = {'R²': r2, 'RMSE': rmse, 'MAE': mae}

    print(f"\nTarget: {target}")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # Optional scatter plot for visualization
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.4)
    plt.xlabel(f"Actual {target}")
    plt.ylabel(f"Predicted {target}")
    plt.title(f"{target.upper()} Prediction: Actual vs Predicted")
    plt.show()

# Display final results summary
print("\nSummary of Weather Prediction Performance:")
for k, v in results.items():
    print(f"{k.upper()} → R²: {v['R²']:.3f}, RMSE: {v['RMSE']:.3f}, MAE: {v['MAE']:.3f}")

# %% [markdown]
# Import the neccessary libraries

# %%
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import kneighbors_graph
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.exceptions import ConvergenceWarning
import warnings

# %%
# Supress warnings for cleaner output (especially convergence warnings)
warnings.filterwarnings("ignore",category=ConvergenceWarning)
warnings.filterwarnings("ignore",category=FutureWarning)

# %% [markdown]
# Data Acquistion and Loading

# %%
print("Fetching Blackstone (BX) stock data...........")
# Fetch 5 years of historical data for Blackstone
data = yf.download("BX",start="2019-01-01",end="2024-01-01")

if data.empty:
    print("Error: No data fetched. Check ticker symbol or network connection.")
    

print(f"Data fetched sucessfuly. Shape: {data.shape}")

# %%
data

# %% [markdown]
# Feature Engineering

# %%
# We need to create features (X) and a target (y)
# Target (y): "Will the price go up tomorrow"
# 1 = Price went up, 0 = Price went dowm or stayed the same

# Calculate the change in price tommorrow
data["Price_Change"] = data["Close"].shift(-1) - data["Close"]

# Calculate the target variable: 1 if Price_Change > 0, else 0
data['Target'] = (data["Price_Change"] > 0).astype(int)

# Create features (X)
# We use past data (lags) and technical indicators

# 5-day moving average
data["SMA_5"] = data["Close"].rolling(window=5).mean()

# 20-day moving average
data["SMA_20"] = data["Close"].rolling(window=20).mean()

# 5-day moving average of volume
data["Volume_SMA_5"] = data["Volume"].rolling(window=5).mean()

# Daily percentage change
data["Daily_Change"] = (data["Close"] - data["Open"]) / data["Open"]

# Price change over the last 5 days
data["Lag_5_Change"] = data["Close"].diff(5)

# %% [markdown]
# Data Cleaning

# %%
# We must drop rows with NaN values, which are created by:
# 1. rolling() windows (at the start of the dataset)
# 2. shift(-1) (the very last row, as it has no "next day" to compare)
original_shape = data.shape
data = data.dropna()
print(f"Data cleaned. Dropped NaN rows. Shape changed from {original_shape} to {data.shape}")

# %% [markdown]
# Exploratory Data Visualization (Before Training)

# %%
print("Generating pre-training visualizations...........")
# Plot 1: Close price and Moving Averages
plt.Figure(figsize=(14,7))
plt.plot(data["Close"],label="Close Price")
plt.plot(data["SMA_5"],label="5-Day SMA",linestyle="--")
plt.plot(data["SMA_20"],label="20-Day SMA",linestyle="--")
plt.title("BX Close Price and Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.savefig("bx_price_sma.png")
plt.show()

# %%
# Plot 2: Target Variable Dstribution (Class Balance)
plt.Figure(figsize=(7,5))
sns.countplot(x="Target",data=data,palette="viridis")
plt.title("Target Variable Distribution (0=Down/Same, 1=Up)")
plt.xlabel("Target Class")
plt.ylabel("Count")
plt.savefig("bx_target_distribution.png")
plt.show()

# %% [markdown]
# Feature Engineering

# %%
# Define our features (X) and target (y)
# We exclude columns used to create the target or non-numeric data
features = ["SMA_5","SMA_20","Volume_SMA_5","Daily_Change","Lag_5_Change","Open","High","Low","Close","Volume"]

# Check if all features set
missing_features = [f for f in features if f not in data.columns]
if missing_features:
    print(f"Error: Missing features: {missing_features}")
    

# %%
X = data[features]
y = data["Target"]

# %%
# Plot 3: Correlation Heatmap
plt.Figure(figsize=(12,8))
sns.heatmap(X.corr(),annot=True,fmt=".2f",cmap="coolwarm",linewidth=0.5)
plt.title("Feature Correlation Heatmap")
plt.xticks(rotation=45,ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("bx_feature_heatmap.png")
plt.show()

# %% [markdown]
# Data Splitting

# %%
# Split data into training and testing sets
# We use shuffle=False because this is time-series data
# We want to train an older data and test on more recent data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=False)
print(f'Training set size: {X_train.shape[0]} samples')
print(f"Testing set size: {X_test.shape[0]} samples")

# %% [markdown]
# Data Scaling

# %%
# Scale the features
# This is cruical for models like SVM and KNN
scaler = StandardScaler()

# Fit the scaler ONLY on the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the scaler fitted on training data
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# Model Training and Comparison

# %%
print("----- Comparing Classfication Models -----")

# A dictionary of models we want to test set
models = {
    "Logistic Regression":LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors":KNeighborsClassifier(),
    "Support Vector Machine":SVC(),
    "Decision Tree":DecisionTreeClassifier(random_state=42),
    "Random Forest":RandomForestClassifier(random_state=42),
    "Gaussian Naive Bayes":GaussianNB()
}

# Store results to find the best model
results = {}

# %%
for name,model in models.items():
    print(f"Training {name}.....")
    # Train the model
    model.fit(X_train_scaled,y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test,y_pred)

    # Store and print the results
    results[name] = accuracy
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test,y_pred,zero_division=0))
    print("-"*30)


# Find the best model based on accuracy
best_model_name = max(results,key=results.get)
print(f"Best Initial model: {best_model_name} with accuracy: {results[best_model_name]:.4f}")

# %% [markdown]
# Hyperparameter Tuning (Using GridSearchCV)



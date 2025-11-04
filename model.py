# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("✅ Libraries imported successfully!")

# Download Blackstone Group stock data (ticker: BX)
def load_stock_data():
    """
    Load Blackstone stock data from Yahoo Finance
    Returns: DataFrame with stock data
    """
    print("📊 Downloading Blackstone stock data...")
    ticker = "BX"
    stock = yf.download(ticker, start="2015-01-01", end="2024-01-01")
    return stock

# Load the data
data = load_stock_data()
print(f"✅ Data loaded successfully! Shape: {data.shape}")
print(data.head())

# Create features and target variable for classification
def create_features(df):
    """
    Create technical indicators and prepare data for classification
    The target variable will be: 1 if next day's close > current close, else 0
    """
    df = df.copy()
    
    # Calculate technical indicators
    # Moving averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # Price rate of change
    df['ROC_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    
    # Price vs Moving averages
    df['Price_MA5_Ratio'] = df['Close'] / df['MA_5']
    df['Price_MA20_Ratio'] = df['Close'] / df['MA_20']
    
    # Create target variable (Binary Classification)
    # 1 if price goes up next day, 0 if price goes down
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Drop NaN values created by technical indicators
    df = df.dropna()
    
    return df

# Create features
feature_data = create_features(data)
print(f"✅ Features created successfully! New shape: {feature_data.shape}")

# VISUALIZATION 1: Before Training - Data Exploration
def exploratory_visualization(df):
    """
    Create comprehensive visualizations of the stock data before training
    """
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    fig.suptitle('Blackstone Stock Analysis - Before Training', fontsize=16, fontweight='bold')
    
    # 1. Stock price over time
    axes[0, 0].plot(df.index, df['Close'], color='blue', linewidth=1)
    axes[0, 0].set_title('Blackstone Stock Price Over Time')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Volume traded
    axes[0, 1].plot(df.index, df['Volume'], color='green', alpha=0.7)
    axes[0, 1].set_title('Trading Volume Over Time')
    axes[0, 1].set_ylabel('Volume')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Technical indicators
    axes[1, 0].plot(df.index, df['MA_20'], label='20-Day MA', alpha=0.7)
    axes[1, 0].plot(df.index, df['Close'], label='Close Price', alpha=0.5)
    axes[1, 0].set_title('Price vs Moving Average')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. RSI
    axes[1, 1].plot(df.index, df['RSI'], color='purple')
    axes[1, 1].axhline(70, linestyle='--', alpha=0.5, color='red', label='Overbought')
    axes[1, 1].axhline(30, linestyle='--', alpha=0.5, color='green', label='Oversold')
    axes[1, 1].set_title('Relative Strength Index (RSI)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. MACD
    axes[2, 0].plot(df.index, df['MACD'], label='MACD', color='blue')
    axes[2, 0].plot(df.index, df['MACD_Signal'], label='Signal', color='red')
    axes[2, 0].set_title('MACD Indicator')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. Target distribution
    target_counts = df['Target'].value_counts()
    axes[2, 1].bar(['Down (0)', 'Up (1)'], target_counts.values, color=['red', 'green'], alpha=0.7)
    axes[2, 1].set_title('Target Variable Distribution')
    axes[2, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()

# Generate exploratory visualizations
exploratory_visualization(feature_data)

# Prepare data for machine learning
def prepare_ml_data(df):
    """
    Prepare features and target for machine learning models
    """
    # Select features for modeling
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                      'MA_5', 'MA_20', 'MA_50', 'ROC_5', 'Volatility', 
                      'RSI', 'MACD', 'MACD_Signal', 'Price_MA5_Ratio', 'Price_MA20_Ratio']
    
    X = df[feature_columns]
    y = df['Target']
    
    # Split data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns

# Prepare ML data
X_train, X_test, y_train, y_test, scaler, feature_columns = prepare_ml_data(feature_data)
print(f"✅ Data prepared for ML! Training shape: {X_train.shape}, Test shape: {X_test.shape}")

# Initialize all classification models
def initialize_models():
    """
    Initialize multiple classification models for comparison
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Gaussian Naive Bayes': GaussianNB(),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
        'Neural Network': MLPClassifier(random_state=42, max_iter=1000)
    }
    return models

# Train and compare all models
def compare_models(models, X_train, X_test, y_train, y_test):
    """
    Train multiple models and compare their performance
    """
    results = {}
    
    for name, model in models.items():
        print(f"🏋️ Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_mean,
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
        
        print(f"✅ {name} - Accuracy: {accuracy:.4f}, CV Score: {cv_mean:.4f} (±{cv_scores.std():.4f})")
    
    return results

# Initialize and train all models
models = initialize_models()
results = compare_models(models, X_train, X_test, y_train, y_test)

# VISUALIZATION 2: Model Comparison
def model_comparison_visualization(results):
    """
    Create visualizations comparing model performance
    """
    # Extract model names and metrics
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    cv_means = [results[name]['cv_mean'] for name in model_names]
    cv_stds = [results[name]['cv_std'] for name in model_names]
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Accuracy comparison
    bars1 = ax1.bar(model_names, accuracies, color='skyblue', alpha=0.7, edgecolor='black')
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy Score')
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Cross-validation scores with error bars
    x_pos = np.arange(len(model_names))
    bars2 = ax2.bar(x_pos, cv_means, yerr=cv_stds, capsize=5, color='lightgreen', 
                   alpha=0.7, edgecolor='black')
    ax2.set_title('Cross-Validation Scores (5-fold)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('CV Accuracy Score')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Find best model
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"\n🏆 BEST MODEL: {best_model_name} with accuracy: {best_accuracy:.4f}")
    
    return best_model_name

# Generate model comparison visualization
best_model_name = model_comparison_visualization(results)

# HYPERPARAMETER TUNING for the best model
def hyperparameter_tuning(best_model_name, results, X_train, y_train):
    """
    Perform hyperparameter tuning for the best performing model
    """
    print(f"\n🎯 Performing Hyperparameter Tuning for {best_model_name}...")
    
    model = results[best_model_name]['model']
    
    # Define parameter grids for different models
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'subsample': [0.8, 0.9, 1.0]
        },
        'Logistic Regression': {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    }
    
    # Get parameter grid for the best model
    if best_model_name in param_grids:
        param_grid = param_grids[best_model_name]
        
        # Perform Grid Search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"✅ Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    else:
        print(f"⚠️ No parameter grid defined for {best_model_name}. Using default model.")
        return model

# Perform hyperparameter tuning
tuned_model = hyperparameter_tuning(best_model_name, results, X_train, y_train)

# Update results with tuned model
results[best_model_name + ' (Tuned)'] = {
    'model': tuned_model,
    'accuracy': accuracy_score(y_test, tuned_model.predict(X_test)),
    'predictions': tuned_model.predict(X_test)
}

# VISUALIZATION 3: Confusion Matrix for Best Model
def confusion_matrix_visualization(best_model_name, results, y_test):
    """
    Create confusion matrix visualization for the best model
    """
    best_predictions = results[best_model_name + ' (Tuned)']['predictions']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Predicted Down', 'Predicted Up'],
                yticklabels=['Actual Down', 'Actual Up'])
    ax1.set_title(f'Confusion Matrix - {best_model_name}', fontweight='bold')
    
    # Classification report as text
    report = classification_report(y_test, best_predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Create a table for classification report
    table = ax2.table(cellText=report_df.values,
                     rowLabels=report_df.index,
                     colLabels=report_df.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax2.axis('off')
    ax2.set_title('Classification Report', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# Generate confusion matrix visualization
confusion_matrix_visualization(best_model_name, results, y_test)

# FEATURE IMPORTANCE Visualization (for tree-based models)
def feature_importance_visualization(model, feature_columns, model_name):
    """
    Visualize feature importance for tree-based models
    """
    if hasattr(model, 'feature_importances_'):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        ax.barh(range(len(indices)), importances[indices], color='steelblue')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_columns[i] for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Feature Importance - {model_name}', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()
    else:
        print(f"⚠️ Feature importance not available for {model_name}")

# Visualize feature importance for tree-based models
if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Decision Tree']:
    feature_importance_visualization(tuned_model, feature_columns, best_model_name)

# PREDICTION FUNCTION for new data
def predict_new_data(model, scaler, feature_columns):
    """
    Create a function to make predictions on new data
    """
    def predict_stock_direction(new_data):
        """
        Predict stock direction for new data
        
        Parameters:
        new_data: Dictionary containing the required features
        """
        # Create DataFrame from input
        input_df = pd.DataFrame([new_data])
        
        # Ensure all required columns are present
        missing_cols = set(feature_columns) - set(input_df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Reorder columns to match training data
        input_df = input_df[feature_columns]
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        return {
            'prediction': 'UP' if prediction == 1 else 'DOWN',
            'confidence': max(probability),
            'probability_up': probability[1],
            'probability_down': probability[0]
        }
    
    return predict_stock_direction

# Create prediction function
predictor = predict_new_data(tuned_model, scaler, feature_columns)

# Example of using the prediction function
print("\n🔮 EXAMPLE PREDICTION:")
# Get the latest data point for example
latest_data = feature_data[feature_columns].iloc[-1:].to_dict('records')[0]

prediction_result = predictor(latest_data)
print(f"Prediction: {prediction_result['prediction']}")
print(f"Confidence: {prediction_result['confidence']:.2%}")
print(f"Probability UP: {prediction_result['probability_up']:.2%}")
print(f"Probability DOWN: {prediction_result['probability_down']:.2%}")

# COMPREHENSIVE MODEL SUMMARY
def create_model_summary(results):
    """
    Create a comprehensive summary of all models' performance
    """
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE MODEL COMPARISON SUMMARY")
    print("="*80)
    
    summary_data = []
    for name, result in results.items():
        if '(Tuned)' not in name:  # Avoid duplicate entries for tuned models
            summary_data.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'CV Score': result['cv_mean'],
                'CV Std': result['cv_std']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Accuracy', ascending=False)
    
    print(summary_df.to_string(index=False, float_format='%.4f'))
    
    # Additional insights
    print(f"\n💡 INSIGHTS:")
    print(f"- Best Model: {best_model_name}")
    print(f"- Total Models Compared: {len(models)}")
    print(f"- Dataset Size: {len(feature_data)} samples")
    print(f"- Feature Count: {len(feature_columns)}")
    print(f"- Target Distribution: {feature_data['Target'].value_counts().to_dict()}")

create_model_summary(results)

print("\n🎉 MACHINE LEARNING CLASSIFICATION COMPARISON COMPLETED!")
print("✅ All models trained and compared")
print("✅ Hyperparameter tuning performed")
print("✅ Visualizations created")
print("✅ Prediction function ready for new data")
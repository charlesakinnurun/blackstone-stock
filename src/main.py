import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import data_handler
import features
import models
import visuals

# Setup
warnings.filterwarnings("ignore")

def run_pipeline():
    # 1. Acquisition
    raw_data = data_handler.fetch_stock_data("BX", "2019-01-01", "2024-01-01")
    
    # 2. Engineering & Cleaning
    df = features.create_target_and_features(raw_data)
    df = data_handler.clean_data(df)
    
    # 3. Splitting
    feature_cols = features.get_feature_lists()
    X = df[feature_cols]
    y = df["Target"]
    
    # Shuffle=False for Time Series
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 4. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Visuals (Pre-training)
    visuals.plot_price_trends(df)
    
    # 6. Model Comparison
    model_list = models.get_model_pipeline()
    best_name, _ = models.train_and_compare(model_list, X_train_scaled, y_train, X_test_scaled, y_test)
    print(f"\nWinner: {best_name}")
    
    # 7. Tuning
    final_model = models.tune_random_forest(X_train_scaled, y_train)
    
    # 8. Final Evaluation
    y_pred = final_model.predict(X_test_scaled)
    print("\n--- Final Tuned Model Results ---")
    print(classification_report(y_test, y_pred))
    
    # 9. Final Visuals
    visuals.plot_confusion_matrix(y_test, y_pred)
    visuals.plot_feature_importance(final_model, feature_cols)
    
    # 10. Single Prediction
    last_row_scaled = scaler.transform(X.iloc[[-1]])
    pred = final_model.predict(last_row_scaled)[0]
    print(f"Prediction for tomorrow: {'UP' if pred == 1 else 'DOWN/STAY'}")

if __name__ == "__main__":
    run_pipeline()
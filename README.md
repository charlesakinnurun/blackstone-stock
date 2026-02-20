<!--# BlackStone Stock-->
![blackstone-stock-image](/assets/blackstone.jpg)
# 🎯 Objectives

-   Predict whether a stock price will move up or down

-   Engineer meaningful technical indicators from raw market data

-   Compare multiple ML classification models

-   Evaluate performance using robust metrics

-   Visualize model performance and results

## Procedures
- Import libraries
    - pandas
    - numpy
    - seaborn
    - matplotlib
    - scikit-learn
    - yfinance
  



- Data Loading

<a href="/data/blackstone_stock_data.csv">Check out dataset</a>

| Date       | Price      | Close      | High       | Low        | Open       | Volume   |
|------------|------------|------------|------------|------------|------------|----------|
| 2019-01-02 | 23.392975  | 23.564125  | 22.179372  | 22.770615  | 22.770615  | 3,733,300 |
| 2019-01-03 | 22.739500  | 23.431877  | 22.638367  | 23.237390  | 23.237390  | 5,371,900 |
| 2019-01-04 | 23.517448  | 23.766392  | 23.190709  | 23.268504  | 23.268504  | 5,867,600 |
| 2019-01-07 | 24.217609  | 24.443216  | 23.237393  | 23.688604  | 23.688604  | 6,071,700 |
| 2019-01-08 | 24.427656  | 24.793294  | 24.100917  | 24.559908  | 24.559908  | 4,062,700 |
| ...        | ...        | ...        | ...        | ...        | ...        | ...      |
| 2023-12-22 | 123.609901 | 124.924598 | 122.238458 | 122.711370 | 122.711370 | 3,342,700 |
| 2023-12-26 | 124.253036 | 124.858364 | 123.278838 | 123.590962 | 123.590962 | 2,485,000 |
| 2023-12-27 | 125.425865 | 126.201434 | 123.505841 | 123.969301 | 123.969301 | 3,561,200 |
| 2023-12-28 | 125.917717 | 126.296041 | 124.886757 | 125.321842 | 125.321842 | 2,087,900 |
| 2023-12-29 | 123.827423 | 126.021744 | 123.222096 | 125.671779 | 125.671779 | 2,049,000 |







- Data Cleaning
    - Drop rows with NaN values





- Exploratory Data Visualization
    - <a href="/assets/bx_price_sma.png">Close Price and Moving Average</a>




- Feature Engineering
    - Features: SMA_5, SMA_20, Volume_SMA_5, Daily_Change, Lag_5_Change
    - Target: Close

    - <a href="/assets/bx_feature_importance.png">Feature Importance</a>
    - <a href="/assets/bx_feature_heatmap.png">Feature Correlation Heatmap</a>
    - <a href="/assets/bx_target_distribution.png">Target Distribution</a>



- Data Spltting
    - Split the data into training and testing sets
    - Use the shuffle=False because this is time-series data




- Data Scaling
    - Scale the features using Standard Scaler
    - This is cruical for models like SVM and KNN





- Models Implemented
    - Logistic Regression
    - K-Nearest Neighbors (KNN)
    - Support Vector Machine (SVM)
    - Decison Tree 
    - Random Forest
    - Gaussian Naive bayes

- Hyperparameter Tuning (GridSearchCV)
    - n_estimators
    - max_depth
    - min_samples_split


- Model Evaluation
    - Recall
    - F1 Score
    - Precision
    - AUC and ROC
    - Confusion Matrix
    - Classification Report


- Post-Training Visualization
    - <a href="/assets/bx_confusion_matrix.png">Confusion Matrix</a>


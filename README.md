# BlackStone Stock
![blackstone-stock](/blackstone_header_image.webp)


## Procedures
- Import the libraries
    - pandas
    - numpy
    - seaborn
    - matplotlib
    - scikit-learn
    - yfinance
- Data Acquisition
    - Data Acquired from Yahoo Finance API
- Data Loading

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


- Feature Engineering
    - Features: SMA_5, SMA_20, Volume_SMA_5, Daily_Change, Lag_5_Change, Open, High, Low, Close, Volume
- Data Cleaning
    - Drop rows with NaN values
- Exploratory Data Visualization

![Close-Price-and-Moving-Avearges](/bx_price_sma.png)

![Target-Variable-Distribution](/bx_target_distribution.png)
- Feature Engineering

![Feature-Correlation-Heatmap](/bx_feature_heatmap.png)
- Data Spltting
    - Split the data into training and testing sets
    - We use the shuffle=False because this is time-series data
- Data Scaling
    - Scale the features
    - This is cruical for models like SVM and KNN
- Model Comparison
    - Logistic Regression
    - K-Nearest Neighbors
    - Support Vector Machine
    - Decison Tree 
    - Random Forest
    - Gaussian Naive bayes
- Model Training


#### Classification Model Performance Summary

---

 **Logistic Regression**
**Accuracy:** 0.4919  

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.38      | 0.28   | 0.32     | 107     |
| 1     | 0.54      | 0.65   | 0.59     | 141     |

**Overall Metrics**

| Metric        | Score |
|---------------|--------|
| Macro Avg F1  | 0.46   |
| Weighted Avg F1 | 0.48 |
| Total Support | 248    |

---

**K-Nearest Neighbors**
**Accuracy:** 0.5081  

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.43      | 0.44   | 0.44     | 107     |
| 1     | 0.57      | 0.56   | 0.56     | 141     |

**Overall Metrics**

| Metric        | Score |
|---------------|--------|
| Macro Avg F1  | 0.50   |
| Weighted Avg F1 | 0.51 |
| Total Support | 248    |

---

**Support Vector Machine**

| Metric        | Score |
|---------------|--------|
| Weighted Avg Precision | 0.58 |
| Weighted Avg Recall    | 0.45 |
| Weighted Avg F1-Score  | 0.33 |
| Support                | 248 |

---

### *Best Initial Model*
Support Vector Machine
**Accuracy:** 0.5121



- Hyperparameter Tuning
    - Using GridSearchCV
- Final Model Evaluation (After Tuning)


**Classification Report**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.39      | 0.38   | 0.38     | 107     |
| 1     | 0.54      | 0.54   | 0.54     | 141     |

**Overall Metrics**

| Metric            | Score |
|-------------------|--------|
| Accuracy          | 0.4718 |
| Macro Avg F1      | 0.46   |
| Weighted Avg F1   | 0.47   |
| Total Support     | 248    |



- Post-Training Visualization

![confusion-matrix-for-best-model](/bx_confusion_matrix.png)

![feature-importance-for-the-best-model](/bx_feature_importance.png)
- Input for New Prediction


## Tech Stack and Tools
- Programming language
    - Python 
- libraries
    - scikit-learn
    - pandas
    - numpy
    - seaborn
    - matplotlib
    - yfinance
- Environment
    - Jupyter Notebook
    - Anaconda
    - Google Colab
- IDE
    - VSCode

You can install all dependencies via:
```
pip install -r requirements.txt
```


## Usage Instructions
To run this project locally:
1. Clone the repository:
```
git clone https://github.com/charlesakinnurun/blackstone-stock.git
cd blackstone-stock
```
2. Install required packages
```
pip install -r requirements.txt
```
3. Open the notebook:
```
jupyter notebook model.ipynb

```


## Project Structure
```
blackstone-stock/
│
├── model.ipynb  
|── model.py    
|── blackstone_stock_data.csv  
├── requirements.txt 
├── blackstone_header_image.webp    
├── bx_confusion_matrix.png
├── bx_feature_heatmap.png   
├── bx_price_sma.png
├── bx_target_distributing.png     
├── CONTRIBUTING.md    
├── CODE_OF_CONDUCT.md 
├── SECURITY.md
├── LICENSE
└── README.md          

```
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

def get_model_pipeline():
    """Returns a dictionary of classification models."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gaussian Naive Bayes": GaussianNB()
    }

def train_and_compare(models, X_train, y_train, X_test, y_test):
    """Trains multiple models and identifies the best performer."""
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = (acc, model)
        print(f"{name} Accuracy: {acc:.4f}")
    
    best_name = max(results, key=lambda x: results[x][0])
    return best_name, results[best_name][1]

def tune_random_forest(X_train, y_train):
    """Performs Hyperparameter tuning on Random Forest."""
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=3, n_jobs=-1, scoring="accuracy"
    )
    print("Running GridSearchCV for Random Forest...")
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
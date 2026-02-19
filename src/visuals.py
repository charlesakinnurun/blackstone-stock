import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

def plot_price_trends(data, filename="bx_price_sma.png"):
    plt.figure(figsize=(12, 6))
    plt.plot(data["Close"], label="Close Price")
    plt.plot(data["SMA_5"], label="5-Day SMA", linestyle="--")
    plt.plot(data["SMA_20"], label="20-Day SMA", linestyle="--")
    plt.title("Price and Moving Averages")
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_confusion_matrix(y_test, y_pred, filename="bx_confusion_matrix.png"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()

def plot_feature_importance(model, feature_names, filename="bx_feature_importance.png"):
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances, y=importances.index)
        plt.title("Feature Importance")
        plt.savefig(filename)
        plt.close()
from models.linear_regression import LinearClassifier
from models.random_forest import RandomForestClassifier
from models.gradient_boosting import GradientBoostingClassifier
from models.baseline import BaselineClassifier
from models.collaborative_filtering import CollaborativeFilteringClassifier
from models.neural_network import NeuralNetworkClassifier
from models.xgboost_model import XGBoostClassifier
from models.ensemble import EnsembleClassifier
from typing import Protocol
import pandas as pd
import numpy as np
import random
import time
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    auc
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns



class Classifier(Protocol):
    def setup(self, X: pd.DataFrame, y: pd.Series): ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...  # Optional, for ROC curves



def bin_prices(prices: pd.Series, bin_edges: np.ndarray = None, num_bins: int = 5) -> tuple:

    if bin_edges is None:
        # Calculate percentile edges (0th, 20th, 40th, 60th, 80th, 100th percentile for 5 bins)
        percentiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(prices, percentiles)
        
        bin_edges[0] = prices.min()
        bin_edges[-1] = prices.max() + 1  
    
    labels = [f"${int(bin_edges[i]):,}-${int(bin_edges[i+1]):,}" for i in range(len(bin_edges) - 1)]
    
    binned = pd.cut(prices, bins=bin_edges, labels=labels, include_lowest=True, right=False)
    
    return binned, bin_edges, labels



def load_data(filepath: str = "vehicles_cleaned.csv") -> pd.DataFrame:
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} rows")
    return df



def split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    print(f"\nSplitting data: {1-test_size:.0%} train, {test_size:.0%} test...")
    
    df = df[df['price'].notna() & (df['price'] > 0)].copy()
    
    # Split data into test and train
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True
    )
    
    print(f"Train set: {len(train_df):,} rows")
    print(f"Test set: {len(test_df):,} rows")
    
    return train_df, test_df



def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 1):
    if k == 0:
        return 0.0
    return precision_score(y_true, y_pred, average='weighted', zero_division=0)



def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 1):
    if k == 0:
        return 0.0
    return recall_score(y_true, y_pred, average='weighted', zero_division=0)



def average_precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 1):
    return precision_score(y_true, y_pred, average='macro', zero_division=0)



def evaluate(y_true: np.ndarray, y_pred: np.ndarray, k: int, name: str):
    p = precision_at_k(y_true, y_pred, k)
    r = recall_at_k(y_true, y_pred, k)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    map_k = average_precision_at_k(y_true, y_pred, k)
    
    print(f"\tEvaluation Results (k={k}):")
    print(f"\t  Precision@k: {p:.4f}")
    print(f"\t  Recall@k:    {r:.4f}")
    print(f"\t  F1@k:        {f1:.4f}")
    print(f"\t  MAP@k:       {map_k:.4f}\n")
    
    return name, p, r, f1, map_k


def plot_roc_curves(classifiers: list, X_test: pd.DataFrame, y_test: pd.Series, class_labels: list):
  
    classes = np.unique(y_test.values)
    
    y_test_bin = label_binarize(y_test.values, classes=classes)
    
    plt.figure(figsize=(10, 8))
    
    for classifier in classifiers:
        name = classifier.__class__.__name__
        y_score = classifier.predict_proba(X_test)
        
        if hasattr(classifier, 'classes_'):
            model_classes = classifier.classes_
        elif hasattr(classifier, 'label_encoder') and classifier.label_encoder is not None:
            model_classes = classifier.label_encoder.classes_
        elif hasattr(classifier, 'model') and hasattr(classifier.model, 'classes_'):
            model_classes = classifier.model.classes_
        else:
            model_classes = classes
        
        if len(model_classes) == len(classes) and not np.array_equal(model_classes, classes):
            class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            reordered_scores = np.zeros_like(y_score)
            for model_idx, model_cls in enumerate(model_classes):
                if model_cls in class_to_idx:
                    test_idx = class_to_idx[model_cls]
                    reordered_scores[:, test_idx] = y_score[:, model_idx]
            y_score = reordered_scores
        
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
            
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random', alpha=0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/ROC_Curves.png', dpi=150)
    print("Saved ROC curves to figures/ROC_Curves.png")
    plt.show()


def plot_confusion_matrix(classifier, X_test: pd.DataFrame, y_test: pd.Series, class_labels: list, model_name: str):
    y_pred = classifier.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True Price Bin')
    plt.xlabel('Predicted Price Bin')
    plt.tight_layout()
    
    filename = f'figures/ConfusionMatrix_{model_name}.png'
    plt.savefig(filename, dpi=150)
    print(f"Saved confusion matrix to {filename}")
    plt.close()


def eval():
    classifiers: list[Classifier] = [
        BaselineClassifier(),  
        LinearClassifier(),
        RandomForestClassifier(n_estimators=100, max_depth=20),
        NeuralNetworkClassifier(hidden_layer_sizes=(200, 100), max_iter=300), 
        CollaborativeFilteringClassifier(n_components=0.95, n_neighbors=50),  
        #GradientBoostingClassifier(n_estimators=20, max_depth=5, learning_rate=0.1, verbose=1),
        XGBoostClassifier(n_estimators=500, max_depth=8, learning_rate=0.05),
        EnsembleClassifier(),
    ]
    
    # Load data
    df = load_data("vehicles_cleaned.csv")

    # Create region feature
    north_america = [
        "buick", "cadillac", "chevrolet", "chrysler", "dodge",
        "ford", "gmc", "harley-davidson", "jeep", "lincoln",
        "mercury", "pontiac", "ram", "saturn", "tesla"
    ]

    europe = [
        "alfa-romeo", "aston-martin", "audi", "bmw", "ferrari",
        "fiat", "jaguar", "land rover", "mini", "mercedes-benz",
        "morgan", "porsche", "rover", "volkswagen"
    ]

    asia = [
        "acura", "datsun", "honda", "hyundai", "infiniti",
        "kia", "lexus", "mazda", "mitsubishi", "nissan",
        "subaru", "toyota"
    ]

    df["region"] = df["manufacturer"].apply(lambda x: "north_america" if x in north_america else "europe" if x in europe else "asia")

    # Create miles per year feature
    df['miles_per_year'] = df['odometer'] / df['age'].clip(lower=1)
    df['miles_per_year'] = df['miles_per_year'].replace([np.inf, -np.inf], np.nan)  

    
    # Split data
    train, test = split(df, test_size=0.2, random_state=42)
    
    # Bin prices into 5 bins, each ~20% of data    
    # Calculate bin edges from TRAINING data only
    train['price_bin'], train_bin_edges, train_labels = bin_prices(train['price'], num_bins=5)
    
    # Use the same bin edges for test set (no leakage - test set never used for bin calculation)
    test['price_bin'] = pd.cut(test['price'], bins=train_bin_edges, labels=train_labels, include_lowest=True, right=False)
    
    # Check bin distribution
    print(f"\nPrice bin distribution (train set):")
    bin_counts = train['price_bin'].value_counts().sort_index()
    print(f"  Number of bins: {len(bin_counts)}")
    print(f"  Bin distribution (each bin should have ~{len(train)/5:.0f} samples = 20%):")
    for bin_label, count in bin_counts.items():
        pct = (count / len(train)) * 100
        print(f"    {bin_label}: {count:,} samples ({pct:.1f}%)")
    
    # Prepare features and target
    X_train = train.drop(columns=['price', 'price_bin'])
    y_train = train['price_bin']
    X_test = test.drop(columns=['price', 'price_bin'])
    y_test = test['price_bin']
    
    
    eval_results = []
    k = 1  # For classification, k=1 means using the top prediction
    
    for classifier in classifiers:
        name = classifier.__class__.__name__
        print(f"{name} - Running... ")
        before = time.time()

        classifier.setup(X_train, y_train)
        elapsed = time.time() - before
        print(f"  {name} setup completed in {elapsed:.2f}s")
        
        # Make predictions
        y_pred = classifier.predict(X_test)
        
        seconds_since = time.time() - before
        result = evaluate(y_test.values, y_pred, k, name) + (seconds_since,)
        eval_results.append(result)
        print(f"Completed in {seconds_since:.2f}s")
    
    # Print summary table
    headers = ["Name", "Precision@k", "Recall@k", "F1@k", "MAP@k", "Time"]
    print()
    print(tabulate(
        pd.DataFrame(eval_results, columns=headers).sort_values(by="F1@k", ascending=False),
        headers=headers,
        showindex=False,
        tablefmt='psql',
        floatfmt=".4f"
    ))
    print()
    
    plot_roc_curves(classifiers, X_test, y_test, train_labels)
    
    ensemble_classifier = next((c for c in classifiers if isinstance(c, EnsembleClassifier)), None)
    if ensemble_classifier:
        plot_confusion_matrix(ensemble_classifier, X_test, y_test, train_labels, "EnsembleClassifier")



if __name__ == "__main__":
    eval()

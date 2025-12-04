import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.xgboost_model import XGBoostClassifier
from eval import load_data, split, bin_prices
from feature_engineering import create_engineered_features

def analyze_feature_importance():
    df = load_data("vehicles_cleaned.csv")
    
    north_america = ["buick", "cadillac", "chevrolet", "chrysler", "dodge", "ford", "gmc", "jeep", "lincoln", "ram", "tesla"]
    europe = ["audi", "bmw", "ferrari", "fiat", "jaguar", "land rover", "mercedes-benz", "porsche", "volkswagen", "volvo"]
    asia = ["acura", "honda", "hyundai", "infiniti", "kia", "lexus", "mazda", "mitsubishi", "nissan", "subaru", "toyota"]
    df["region"] = df["manufacturer"].apply(lambda x: "north_america" if x in north_america else "europe" if x in europe else "asia")
    
    df['price_bin'], bin_edges, bin_labels = bin_prices(df['price'], num_bins=5)
    train, test = split(df, test_size=0.2, random_state=42)
    
    categories = ['economy', 'luxury', 'exotic']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, category in enumerate(categories):
        print(f"Analyzing {category} market...")
        
        cat_data = train[train['brand_category'] == category].copy()
        
        if len(cat_data) < 100:
            print(f"Skipping {category} - not enough data ({len(cat_data)} samples)")
            continue
        
        cat_data_eng, _ = create_engineered_features(cat_data, is_training=True)
        
        drop_cols = ['price', 'price_bin', 'id', 'url', 'region_url', 'image_url', 'description', 
                     'brand_category', 'is_luxury', 'is_economy', 'is_exotic', 'county', 'lat', 'long']
        
        X_train = cat_data_eng.drop(columns=[c for c in drop_cols if c in cat_data_eng.columns])
        y_train = cat_data_eng['price_bin']
        
        print(f"Training XGBoost on {len(X_train)} samples with {len(X_train.columns)} features...")
        model = XGBoostClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)
        model.setup(X_train, y_train)
        
        # Get feature importances
        importances = model.model.feature_importances_
        
        feature_names_raw = model.preprocessor.get_feature_names_out()
        feature_names = [name.split('__')[-1] for name in feature_names_raw]
        
        print(f"\nFeature Importances for {category.upper()}:")
        print("-" * 60)
        for name, val in zip(feature_names, importances):
            print(f"{name:40s} {val:.6f}")
        
        aggregated = {}
        for name, val in zip(feature_names, importances):
            if '_' in name and name.split('_')[0] in ['transmission', 'fuel', 'drive', 'type', 'paint', 
                                                       'condition', 'cylinders', 'manufacturer', 'region', 
                                                       'state']:
                base_name = name.split('_')[0]
                aggregated[base_name] = aggregated.get(base_name, 0) + val
            elif name.startswith('model_simple_'):
                aggregated['model_simple'] = aggregated.get('model_simple', 0) + val
            else:
                aggregated[name] = aggregated.get(name, 0) + val
        
        sorted_features = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
        top_n = 5
        top_features = [f[0] for f in sorted_features[:top_n]]
        top_scores = [f[1] for f in sorted_features[:top_n]]
        
        print(f"\nTop {top_n} Aggregated Features:")
        print("-" * 60)
        for name, val in zip(top_features, top_scores):
            print(f"{name:40s} {val:.6f}")
        
        sns.barplot(x=top_scores, y=top_features, ax=axes[idx], palette='viridis')
        axes[idx].set_title(f'{category.capitalize()} Cars\nKey Price Drivers')
        axes[idx].set_xlabel('Relative Importance')
    
    plt.tight_layout()
    plt.savefig('figures/FeatureImportance_ByCategory.png', dpi=150)
    print("Saved plot to figures/FeatureImportance_ByCategory.png")

if __name__ == "__main__":
    analyze_feature_importance()

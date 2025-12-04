import pandas as pd
import numpy as np


def create_engineered_features(df: pd.DataFrame, is_training: bool = True, train_stats: dict = None) -> pd.DataFrame:
    df = df.copy()
    
    if train_stats is None:
        train_stats = {}
    
    # 1. RATIO FEATURES
    # Miles per year
    if 'age' in df.columns and 'odometer' in df.columns:
        df['miles_per_year'] = df['odometer'] / df['age'].clip(lower=1)
        df['miles_per_year'] = df['miles_per_year'].replace([np.inf, -np.inf], np.nan)
    
    # 2. BINARY FLAGS
    # Vehicle age flags
    if 'age' in df.columns:
        df['is_new'] = (df['age'] <= 2).astype(int)
        df['is_classic'] = (df['age'] >= 25).astype(int)
    
    # Mileage flags
    if 'odometer' in df.columns:
        df['is_low_mileage'] = (df['odometer'] < 30000).astype(int)
        df['is_high_mileage'] = (df['odometer'] > 100000).astype(int)
    
    # Fuel type flags
    if 'fuel' in df.columns:
        df['is_electric'] = df['fuel'].str.contains('electric', case=False, na=False).astype(int)
        df['is_hybrid'] = df['fuel'].str.contains('hybrid', case=False, na=False).astype(int)
        df['is_gas'] = df['fuel'].str.contains('gas', case=False, na=False).astype(int)
        df['is_diesel'] = df['fuel'].str.contains('diesel', case=False, na=False).astype(int)
    
    # 3. NUMERIC ENCODINGS
    # Condition to numeric
    if 'condition' in df.columns:
        condition_map = {
            'excellent': 5,
            'good': 4,
            'fair': 3,
            'like new': 5,
            'new': 5,
            'salvage': 1
        }
        df['condition_numeric'] = df['condition'].map(condition_map).fillna(3)  # Default to 'fair'
    
    # Cylinders to numeric
    if 'cylinders' in df.columns:
        # Extract number from strings like "8 cylinders" or "V8"
        df['cylinders_numeric'] = df['cylinders'].astype(str).str.extract(r'(\d+)').astype(float)
        df['cylinders_numeric'] = df['cylinders_numeric'].fillna(df['cylinders_numeric'].median())
    
    # 4. DRIVE TYPE FLAGS
    if 'drive' in df.columns:
        df['is_4wd'] = df['drive'].str.contains('4wd|4x4', case=False, na=False).astype(int)
        df['is_awd'] = df['drive'].str.contains('awd|all wheel', case=False, na=False).astype(int)
    
    # 5. TRANSMISSION FLAGS
    if 'transmission' in df.columns:
        df['is_automatic'] = df['transmission'].str.contains('automatic', case=False, na=False).astype(int)
        df['is_manual'] = df['transmission'].str.contains('manual', case=False, na=False).astype(int)
    
    # 6. VEHICLE TYPE FLAGS
    if 'type' in df.columns:
        df['is_truck'] = df['type'].str.contains('truck|pickup', case=False, na=False).astype(int)
        df['is_suv'] = df['type'].str.contains('suv', case=False, na=False).astype(int)
        df['is_sedan'] = df['type'].str.contains('sedan', case=False, na=False).astype(int)
        df['is_coupe'] = df['type'].str.contains('coupe', case=False, na=False).astype(int)
    
    # 7. BRAND CATEGORY FLAGS
    if 'brand_category' in df.columns:
        df['is_luxury'] = (df['brand_category'] == 'luxury').astype(int)
        df['is_exotic'] = (df['brand_category'] == 'exotic').astype(int)
        df['is_economy'] = (df['brand_category'] == 'economy').astype(int)
    
    return df, train_stats


def get_new_feature_names() -> list:
    return [
        'miles_per_year',
        'is_new', 'is_classic',
        'is_low_mileage', 'is_high_mileage',
        'is_electric', 'is_hybrid', 'is_gas', 'is_diesel',
        'condition_numeric', 'cylinders_numeric',
        'is_4wd', 'is_awd',
        'is_automatic', 'is_manual',
        'is_truck', 'is_suv', 'is_sedan', 'is_coupe',
        'is_luxury', 'is_exotic', 'is_economy'
    ]

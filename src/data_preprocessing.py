import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path, index_col='building_id')

def clean_data(df):
    """Basic data cleaning: drop missing values."""
    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print("Missing values found and dropped:")
        print(missing_counts[missing_counts > 0])
        df.dropna(inplace=True)
    return df

def add_physical_features(df):
    """Adds engineered physical features based on structural data."""
    # Volume and Slenderness
    df['volume'] = df['plinth_area_sq_ft'] * df['height_ft_pre_eq']
    df['slenderness'] = df['height_ft_pre_eq'] / (np.sqrt(df['plinth_area_sq_ft']))
    
    # Interaction features
    df['age_x_mud'] = df['age_building'] * df['has_superstructure_mud_mortar_stone']
    df['height_per_floor'] = df['height_ft_pre_eq'] / df['count_floors_pre_eq']
    
    # Weakness Score (Sum of vulnerable materials)
    weak_materials = [
        'has_superstructure_mud_mortar_stone',
        'has_superstructure_adobe_mud',
        'has_superstructure_stone_flag',
        'has_superstructure_mud_mortar_brick'
    ]
    df['weakness_score'] = df[weak_materials].sum(axis=1)
    df['height_x_weakness'] = df['height_ft_pre_eq'] * df['weakness_score']
    
    # Strength Score (Sum of stable materials)
    strong_materials = [
        'has_superstructure_cement_mortar_brick',
        'has_superstructure_rc_engineered',
        'has_superstructure_rc_non_engineered'
    ]
    df['strength_score'] = df[strong_materials].sum(axis=1)
    
    return df

def smooth_mean_encoding(df, by, on, m=5):
    """Compute smooth mean encoding for categorical features."""
    mean = df[on].mean()
    agg = df.groupby(by)[on].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    smooth = (counts * means + m * mean) / (counts + m)
    return df[by].map(smooth).fillna(mean)

def merge_labels(val):
    """Merges 5 damage levels into 3 categories."""
    if val <= 0: return 0      # Low/Safe (Grade 1)
    elif val <= 2: return 1    # Mid/Repair (Grade 2 & 3)
    else: return 2             # High/Rebuild (Grade 4 & 5)

def preprocess_features(df):
    """Main preprocessing function: encode features and clean data."""
    df = clean_data(df)
    
    le = LabelEncoder()
    df['damage_grade_encoded'] = le.fit_transform(df['damage_grade'])
    
    # Add physical features
    df = add_physical_features(df)
    
    categorical_cols = [
        'vdcmun_id', 'ward_id',
        'land_surface_condition', 'foundation_type', 'roof_type', 
        'ground_floor_type', 'other_floor_type', 
        'position', 'plan_configuration'
    ]
    
    for col in categorical_cols:
        if col in df.columns:
            df[f'{col}_encoded'] = smooth_mean_encoding(df, by=col, on='damage_grade_encoded')
            
    # Apply label merging
    df['target'] = df['damage_grade_encoded'].apply(merge_labels)
            
    return df, le

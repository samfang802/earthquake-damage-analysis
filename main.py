import os
import pandas as pd
from src.data_preprocessing import load_data, preprocess_features
from src.model_training import split_data, train_rf_model, evaluate_model
from src.visualization import plot_damage_distribution, plot_feature_importance, plot_correlation_heatmap

def main():
    # 1. Setup paths
    data_path = 'data/csv_building_structure.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the CSV file is placed in the 'data/' directory.")
        return

    # 2. Data Loading & Preprocessing
    print("--- Phase 1: Data Loading & Preprocessing ---")
    df = load_data(data_path)
    print(f"Loaded {df.shape[0]} rows.")
    
    df, le = preprocess_features(df)
    print("Preprocessing complete. Engineered physical features and encoded categorical columns.")

    # 3. Exploratory Data Analysis (EDA) / Visualization
    print("\n--- Phase 2: Exploratory Data Analysis ---")
    # plot_damage_distribution(df)
    
    # 4. Model Training
    print("\n--- Phase 3: Model Training ---")
    features = [
        # Encoded features
        'vdcmun_id_encoded', 'ward_id_encoded',
        'land_surface_condition_encoded', 'foundation_type_encoded',
        'roof_type_encoded', 'ground_floor_type_encoded',
        'other_floor_type_encoded', 'position_encoded',
        'plan_configuration_encoded',
        
        # Original numerical features
        'count_floors_pre_eq', 'age_building',
        'plinth_area_sq_ft', 'height_ft_pre_eq',
        
        # Engineered physical features
        'volume', 'slenderness', 'age_x_mud', 
        'height_per_floor', 'weakness_score', 'height_x_weakness', 'strength_score',

        # Specific structure flags used in notebook
        'has_superstructure_mud_mortar_stone',
        'has_superstructure_rc_engineered',
        'has_superstructure_mud_mortar_brick'
    ]
    
    # Filter features that actually exist in df
    available_features = [f for f in features if f in df.columns]
    
    X_train, X_test, y_train, y_test = split_data(df, available_features, 'target')
    print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples.")
    
    model = train_rf_model(X_train, y_train)
    print("Random Forest model trained.")

    # 5. Evaluation
    print("\n--- Phase 4: Model Evaluation ---")
    accuracy, report = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    # 6. Post-training Visualization
    print("\n--- Phase 5: Post-training Visualization ---")
    plot_feature_importance(model, available_features)

    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()

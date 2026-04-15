# Earthquake Damage Analysis

This project aims to predict building damage grades following an earthquake using structural and geographical data. It has been modularized from a Jupyter Notebook into a professional Python package structure.

## Project Structure

```text
earthquake-damage-analysis/
│
├── data/                       # Dataset directory (place csv_building_structure.csv here)
│   └── csv_building_structure.csv
│
├── notebooks/                  # Original Jupyter Notebooks
│   └── 第六組_數據分析.ipynb
│
├── src/                        # Modularized source code
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data loading, cleaning, and feature engineering
│   ├── model_training.py       # Model splitting, training, and evaluation
│   └── visualization.py        # plotting functions
│
├── main.py                     # Main entry point to run the analysis pipeline
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore settings
└── README.md                   # Project documentation
```

## Setup Instructions

1.  **Clone the repository** (or navigate to the project directory).
2.  **Ensure you have the dataset**: Place the `csv_building_structure.csv` file inside the `data/` folder.
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the analysis**:
    ```bash
    python main.py
    ```

## Key Features

*   **Custom Encoding**: Uses Smooth Mean Encoding for high-cardinality geographical features (`vdcmun_id`, `ward_id`).
*   **Physical Feature Engineering**: Computes derived features like `volume`, `slenderness`, and `weakness_score` (sum of vulnerable materials).
*   **Label Merging**: Simplifies the 5-grade damage scale into a 3-category target (Safe, Repair, Rebuild) for better model performance.
*   **Random Forest Classifier**: Trained with balanced weights to handle class imbalance.

## Author
Sixth Group (第六組)

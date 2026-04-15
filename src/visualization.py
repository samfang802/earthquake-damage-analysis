import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_damage_distribution(df):
    """Plot the distribution of damage grades."""
    plt.figure(figsize=(12, 5))
    order = ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5']
    sns.countplot(x='damage_grade', data=df, order=[o for o in order if o in df['damage_grade'].unique()])
    plt.title("Distribution of Damage Grade")
    plt.xlabel("Damage Grading")
    plt.show()

def plot_feature_importance(model, feature_names):
    """Plot feature importance from a trained model."""
    importances = model.feature_importances_
    df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df_importance = df_importance.sort_values(by='Importance', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df_importance)
    plt.title("Top 10 Feature Importances")
    plt.show()
    
def plot_correlation_heatmap(df, cols):
    """Plot correlation heatmap for selected columns."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.show()

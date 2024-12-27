import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Helper functions to use it in the app
def count_features_plot(data: pd.DataFrame, feature_name: pd.Series):
    """Plot the count of the feature in descending.
    Args:
    -----
    
    data (pd.DataFrame): Takes A Pandas DataFrame etc.(laptop_df)
    
    feature_name (pd.Series): Takes a Pandas Series like features to plot it's count in the dataFrame
    """
    fig, ax = plt.subplots(figsize=(8,10))
    feature_count = data[feature_name].value_counts().sort_values(ascending=True)
    feature_count.plot(kind="barh", ax=ax)
    plt.xlabel("Count")
    plt.title(f"The count of {feature_name} in the dataset");
    return fig
    
def dis_char(data: pd.DataFrame, feature_name: pd.Series, target_feat: pd.Series):
    """Char displays the distribution between the feature and target like (OpSys vs Price).
    Args:
    -----
    data (pd.DataFrame): Takes a Pandas DataFrame
    
    feature_name (pd.Series): Takes the dataset feature.
    
    target_feat (pd.Series): Takes the target feature in my case is the Price_EGP col.
    """
    if target_feat == "Product":
        target_feat = "Price_EGP"
        
    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(x = data[feature_name], y = data[target_feat])
    plt.xticks(rotation='vertical')
    plt.title(f'{feature_name} & {target_feat}')
    return fig

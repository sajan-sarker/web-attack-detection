import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load dataset from the given path and return dataframe

    Args:
        file_path (_string_): file directory path for the dataset

    Returns:
        df (_DataFrame_): return pandas DataFrame
    """
    try: 
        file = pd.read_csv(file_path)
        return pd.DataFrame(file)

    except Exception as e:
        print(f"An error occurred: {e}")

def create_status(df):
    """create a 'status' column based on the 'Label' column.
    - 'Benign' = 'safe'
    - 'Anything else' = 'malicious'

    Args:
        df (pd.DataFrame): The input DataFrame with a 'Label' column

    Returns: 
        pd.Dataframe: The modified DataFrame with a new 'status' column
    """
    df['status'] = df['Label'].apply(lambda x: 'safe' if x == 'Benign' else 'malicious')
    return df

def count_plot(df, feature, x=6, y=5, rotation='horizontal'):
    """Plot the count plot figure of any given columns in the DataFrame

    Args:
        df (pd.DataFrame): The input DataFrame
        feature (string): column name in the DataFrame
        x, y (int, int): x and y axis for the plot figure
        rotation (string): rotation of the x-axis labels
    """
    plt.figure(figsize=(x, y))
    sns.countplot(x=feature, data=df, palette='viridis')
    plt.xticks(rotation=rotation)
    plt.show()

def find_constant_columns(df):
    """Find columns with constant values in the DataFrame

    Args:
        df (pd.DataFrame): The input DataFrame

    Returns:
        list: List of columns with constant values
    """
    constant_columns = [column for column in df.columns if df[column].unique()==1]
    return constant_columns

def find_low_variance_columns(df, threshold=0.05):
    """Find columns with low variance in the DataFrame

    Args:
        df (pd.DataFrame): The input DataFrame
        threshold (float): The threshold for variance, default is 0.05

    Returns:
        list: List of columns with low variance
    """
    temp = df.select_dtypes(include=[np.number])    # select only numeric features
    variances = temp.var()  # calculate variance of each column
    low_variance_columns = variances[variances < threshold].index.tolist()

    return low_variance_columns
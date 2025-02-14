import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#######################################
######### Data Load Functions #########
def load_data(file_path):
    """Load dataset from the given path and return dataframe"""
    try: 
        file = pd.read_csv(file_path)
        return pd.DataFrame(file)

    except Exception as e:
        print(f"An error occurred: {e}")


#######################################
###### Data Processing Functions ######
######### Checking Functions ##########

def create_status(df):
    """create a 'status' column based on the 'Label' column in the DataFrame"""
    df['status'] = df['Label'].apply(lambda x: 'safe' if x == 'Benign' else 'malicious')
    return df

def find_duplicate_rows(df):
    """Find duplicate rows in the DataFrame"""
    duplicate_rows = df.duplicated().sum()
    return duplicate_rows

def find_duplicate_columns(df):
    """Find duplicate columns in the DataFrame"""
    identical_columns = []  # list of tuples (original, duplicate)
    original_columns = []
    duplicate_columns = []
    _, n = df.shape
    for i in range(n):
        for j in range(i+1, n):
            if df.iloc[:, i].equals(df.iloc[:, j]):
                identical_columns.append((df.columns[i], df.columns[j]))
                original_columns.append(df.columns[i])
                duplicate_columns.append(df.columns[j])
    return identical_columns, original_columns, duplicate_columns

def find_missing_values(df):
    """Find columns with missing values in the DataFrame"""
    missing_values = df.isnull().sum()
    missing_columns = missing_values[missing_values > 0].index.tolist()
    return missing_columns

def find_infinite_values(df):
    """Find columns with infinite values in the DataFrame"""
    infinite_counts = df.isin([np.inf, -np.inf]).sum()
    infinite_columns = infinite_counts[infinite_counts > 0].index.tolist()
    return infinite_columns

def find_constant_columns(df):
    """Find columns with constant values in the DataFrame"""
    constant_columns = [column for column in df.columns if df[column].nunique()==1]
    return constant_columns

def find_low_variance_columns(df, threshold=0.05):
    """Find columns with low variance in the DataFrame based on the threshold"""
    temp = df.select_dtypes(include=[np.number])    # select only numeric features
    variances = temp.var()  # calculate variance of each column
    low_variance_columns = variances[variances < threshold].index.tolist()
    return low_variance_columns

#######################################
###### Data Processing Functions ######
######### Replacing Functions #########

def drop_columns(df, columns):
    """Drop columns in the DataFrame"""
    return df.drop(columns=columns, axis=1, inplace=True)

def drop_rows(df):
    """Drop rows with the duplicated values in the DataFrame"""
    return df.drop_duplicated(inplace=True)

def replace_infinite_values(df):
    """Replace infinite values with NaN in the DataFrame"""
    return df.replace([np.inf, -np.inf], np.nan, inplace=True)

def replace_null_values(df):
    """Replace missing values with the mean of the columns in the DataFrame"""
    return df.fillna(df.mean(), implace=True)



########################################
##### Data Visualization Functions #####
#### Data Imbalance Check Functions ####

def count_plot(df, feature, x=6, y=5, rotation='horizontal'):
    """Plot the count plot figure of any given columns in the DataFrame"""
    plt.figure(figsize=(x, y))
    sns.countplot(x=feature, data=df, palette='viridis')
    plt.xticks(rotation=rotation)
    plt.show()

def print_label_distribution(df, feature, a=5, b=5):
    """Print the distribution of the target column in the DataFrame"""
    label_count = df[feature].value_counts()
    x = label_count.index
    y = label_count.values

    fig, ax = plt.subplots(figsize=(a, b))

    ax.bar(x, y, color='#9D8DF1', edgecolor='black' ) # plot the distribution
    for i, value in enumerate(y):
        ax.text(
            i,
            value,
            f" {value:,}",
            ha='center',
            va='bottom',
            fontsize=12,
            color='black'
        )
    ax.set_title('Class Distribution', fontsize=12)
    ax.set_xlabel(feature, fontsize=10)
    ax.set_ylabel('Count', fontsize=10)

    # adjust x-tick limits and format labels
    ax.set_xticklabels(x, rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
    plt.show()
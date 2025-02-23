import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

########################################
##### Data Load and Save Functions #####
def load_data(file_path):
    """Load dataset from the given path and return dataframe"""
    try: 
        file = pd.read_csv(file_path)
        return pd.DataFrame(file)

    except Exception as e:
        print(f"An error occurred: {e}")

def save_data(df, file_path):
    """Save the dataframe to the given path"""
    try:
        df.to_csv(file_path, index=False)
        print("Data has been saved successfully")
    except Exception as e:
        print(f"An error occurred: {e}")

def save_split_data(data, columns, file_path, file_name):
    """Reduce the dataset size and Save the dataframe to the given path"""
    try:
        path = os.path.join(file_path, file_name)   # join the file path and file name
        df = pd.DataFrame(data, columns=columns)    # create a DataFrame from the data
        
        for column in df.columns:   # iterate through the columns and reduce data types
            if pd.api.types.is_integer_dtype(df[column]):
                # downcast the integer columns to the smallest possible types
                df[column] = pd.to_numeric(df[column], downcast='integer')  
            else:
                # downcast the float columns to the smallest possible types
                df[column] = pd.to_numeric(df[column], downcast='float')
        
        df.to_csv(path, index=False)
    except Exception as e:
        print(f"An error occurred: {e}")


#######################################
###### Data Processing Functions ######
######### Checking Functions ##########

def create_status(df):
    """create a 'status' column based on the 'Label' column in the DataFrame"""
    """map 'Benign' to 0 and 'Malicious' to 1"""
    df['status'] = df['Label'].apply(lambda x: 0 if x == 'Benign' else 1)
    return df

def find_duplicate_rows(df):
    """Find total number of duplicate values in the DataFrame"""
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

def remove_duplicate_values(df):
    """Drop rows with the duplicated values in the DataFrame"""
    df = df.drop_duplicates()
    return df

def replace_infinite_values(df):
    """Replace infinite values with NaN in the DataFrame"""
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def replace_null_values(df):
    """Replace missing values with the mean of the columns in the DataFrame"""
    df = df.fillna(df.mean())
    return df



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
    ax.set_xticks(x)
    ax.set_xticklabels(x, rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
    plt.show()

##########################################
##### Correlation Analysis Functions #####
def correlation_analysis(df, target_column, threshold=0.9):
    """Analyse the correlation between the numerical features in the dataframe"""
    feature_columns = [col for col in df.columns if col not in target_column]   # get the feature columns only
    
    # calculate the correlation matrix
    corr_matrix = df[feature_columns].corr()
    
    # plot the correlation matrix
    plt.figure(figsize=(20,20))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, linewidths=0.5)
    plt.title("Correlation Matrix of the Features Heatmap")
    plt.show()
    
    # Find the highly correlated features
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # check if the absolute correlation is above the threshold
                # append the column names and the correlation value
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    return high_corr_pairs

def calculate_outliers_percentage(df):
    """calculate the percentage of outliers in each column using the IQR method"""
    outlier_percentage = {}
    
    for column in df.columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        # calculate the percentage of outliers in the column
        outlier = df[(df[column] < lower) | (df[column] > upper)]
        percentage = (outlier.shape[0] / df.shape[0]) * 100
        outlier_percentage[column] = percentage
    
    return outlier_percentage
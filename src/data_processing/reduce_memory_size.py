import pandas as pd

def reduce_memory_size(df):
    """Reduce memory size of the DataFrame

    Args:
        df (pd.DataFrame): The input DataFrame

    Returns:
        pd.DataFrame: The modified DataFrame with reduced memory size
    """
    initial_size = df.memory_usage(deep=True).sum() / 1024**2   # calculate initial memory size
    print(f"Initial Memory Size: {initial_size:.2f} MB")

    for column in df.columns:
        if pd.api.types.is_integer_dtype(df[column]):
            # downcast the integer columns to the smallest possible types
            df[column] = pd.to_numeric(df[column], downcast='integer')  
        elif pd.api.types.is_float_dtype(df[column]):
            # downcast the float columns to the smallest possible types
            df[column] = pd.to_numeric(df[column], downcast='float')
        else:
            pass  

    updated_size = df.memory_usage(deep=True).sum() / 1024**2   # calculate updated memory size
    print(f"Updated Memory Size: {updated_size:.2f} MB")
    print(f"Memory Usage Reduced by: {initial_size - updated_size:.2f} MB")     # calculate the memory reduced
    print(f"this is: {100*(updated_size/initial_size):.2f}% of the initial size")   # calculate the percentage of memory reduced

    return df
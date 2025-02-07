import kagglehub
import pandas as pd
import os

# Download latest version
path = kagglehub.dataset_download("dhoogla/cicids2017")

#print("Path to the dataset files:",path)

dir_list = os.listdir(path) # get the filename of the directory
dfps = []
for file_name in dir_list:
    dfps.append(os.path.join(path, file_name))  # merged directory and file name is appended.
    print(file_name)

# store the filtered list of file paths in the variable 'df_temp'
df_temp = [dfp for dfp in dfps if not 'Benign' in dfp]

# read filtered files and concatenate all files into a single dataframe.
df = pd.concat([pd.read_parquet(dfp) for dfp in df_temp], ignore_index=True)

directory = "D:/Programing/web-attack-detection/data/raw"   # directory path
file_name = "cic-ids2017.csv"

if not os.path.exists(directory):
    os.makedirs(directory, exist_ok=True)   # make directory if not exists

file_path=os.path.join(directory, file_name)

try:
    df.to_csv(file_path, index=False)
    print(f"CSV file saved at: {file_path}")
except Exception as e:
    print(f"Error saving CSV file: {e}")



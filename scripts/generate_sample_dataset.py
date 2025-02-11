import pandas as pd

df = pd.read_csv("D:/Programing/web-attack-detection/data/raw/cic-ids2017.csv")
sample_size = 2000  # specify the sample size

# generate a sample dataset
sample_df = df.sample(n=sample_size, random_state=42)

# save the sample dataset
sample_df.to_csv("D:/Programing/web-attack-detection/data/sample/cic-ids2017_sample.csv", index=False)
print(f"Shape of the Sample Dataset: {sample_df.shape}")
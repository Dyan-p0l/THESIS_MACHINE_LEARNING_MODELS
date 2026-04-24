import pandas as pd

df_nr = pd.read_csv("data/raw/raw_split/readings_nr.csv")
df_rc = pd.read_csv("data/raw/raw_split/readings_rc.csv")
df_rf = pd.read_csv("data/raw/raw_split/readings_rf.csv")
df_secondbatch = pd.read_csv("data/raw/raw_split/readings_second_batch.csv")

merged_df = pd.concat([df_nr, df_rc, df_rf, df_secondbatch], ignore_index=True)

merged_df = merged_df.drop(columns=['group'])

merged_df.to_csv("data/raw/readings.csv", index=False)

print('readings merged')
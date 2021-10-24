import os
import sys
import preprocessing

files = ["../data/sources/converted" + p for p in os.listdir("../data/sources/converted") if p.endswith(".json")]

df_list = list()
for f in files:
    fname = f[:-5]
    df = preprocessing.createDataFrame(f, clean=False, verbose=False)
    df.to_csv(f"{fname}.csv", index=False)
    df_list.append(df)
    print(f"{f} CONVERTED")

final_df = preprocessing.createMergedDataFrame(df_list)
final_df.to_csv("../data/merged.csv", index=False)

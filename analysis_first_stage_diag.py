# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 14:08:16 2025

@author: celinep
"""

import pandas as pd



file = r"C:\Veda\Veda_models\IFE-NO-2024.08.27_simplified\IFE-NO-2024.08.27_simplified\Exported_files\032025_131939245.csv"


df = pd.read_csv(file, sep=";", )
scenarios = df["Scenario"].unique()

# print(df.head(10))
df.drop(df[(~df.Attribute.str.contains('VAR')) | (df.Attribute.str.endswith('M')) | (df.Process.str.contains('TRANS')) | (df.Process.str.contains('RESS-'))| (df.Process.str.contains('RESM-')) | (df.Process.str.contains('COM-'))].index, inplace = True) # | (~df.Attribute.str.contains('ObjZ'))

df["Pv"] = pd.to_numeric(df["Pv"].str.replace(",", "."))

df = df.groupby(["Scenario", "Attribute", "Commodity", "Process"], as_index=False, sort=False).agg({'Pv':'sum', })
df["count"] = 1
df["scenarios"] = df["Scenario"].apply(lambda x: x[-2:])
df = df.groupby(["Attribute", "Commodity", "Process", "Pv"], as_index=False, sort=False).agg({'count':'sum', "Scenario":"first", "scenarios":' '.join})
df.drop(df[(df["count"] == 64)].index, inplace=True)



# df.drop_duplicates(subset=["Attribute", "Commodity", "Process", "Period", "Pv"], inplace=True)



# df = pd.pivot_table(df, index=["Scenario", "Attribute", "Commodity", "Process", "count"], columns=["Period"], values="Pv").fillna(0).reset_index()

df["tot_relative"] = 0
for idx, row in df.iterrows():
    mini_df = df[(df["Attribute"] == row["Attribute"]) & (df["Commodity"] == row["Commodity"]) & (df["Process"] == row["Process"])].reset_index()
    tot = 0
    count = mini_df["count"].sum()
    for idx2, row2 in mini_df.iterrows():
        tot += row2["count"] * row2["Pv"]
    average = tot / count
    if average != 0:
        df.at[idx,"tot_relative"] = (row["Pv"] - average) / average


df = df[df["tot_relative"] > 0.1].reset_index()
pd.set_option('display.max_columns', 10)
print(df[["Scenario", "Attribute", "Commodity", "Process", "count", "scenarios", "tot_relative"]].sort_values(by=['tot_relative'], ascending=False).head(50))
# scenarios_diag~0010	Cap_New	-	TSEA1-ELC	2025	NO3	2025	-	INSTCAP	59,72289317
# "Attribute", "Commodity", "Process", "Period", "Region", "Vintage", "Timeslice", "Userconstraint"


#%%
import pandas as pd
import numpy as np


pd.set_option('display.max_columns', 10)
file = r"C:\Veda\Veda_models\IFE-NO-2024.08.27_simplified\IFE-NO-2024.08.27_simplified\Exported_files\040125_112447802.csv"
df = pd.read_csv(file, sep=";", dtype={'Vintage': str, 'Period': str})
scenarios = df["Scenario"].unique()
df["Pv"] = pd.to_numeric(df["Pv"].str.replace(",", "."))

df = df[["Scenario", "Attribute", "Commodity", "Process", "Region", "Vintage", "Timeslice", "Period", "Userconstraint", "Pv"]]

s1 = df[df["Scenario"] == scenarios[0]][["Attribute", "Commodity", "Process", "Region", "Vintage", "Timeslice", "Period", "Userconstraint", "Pv"]]
s2 = df[df["Scenario"] == scenarios[1]][["Attribute", "Commodity", "Process", "Region", "Vintage", "Timeslice", "Period", "Userconstraint", "Pv"]]

df = df.pivot(index=["Attribute", "Commodity", "Process", "Region", "Vintage", "Timeslice", "Period", "Userconstraint"], columns=["Scenario"], values="Pv").fillna(0).reset_index()
df["diff"] = df[scenarios[0]] - df[scenarios[1]]
df["diff_rel"] = df["diff"] / df[scenarios[1]]
df["diff_rel"] = df["diff_rel"].replace(np.inf, 1)

df_similar = df[(abs(df["diff_rel"]) < 0.01)]
df_diff = df[(abs(df["diff_rel"]) >= 0.01)]
print(df_diff)
df_diff.to_csv('out.csv', index=False) 


#%%
from functools import reduce
import numpy as np

file = r"C:\Veda\Veda_models\IFE-NO-2024.08.27_simplified\IFE-NO-2024.08.27_simplified\Exported_files\032025_131939245.csv"
df = pd.read_csv(file, sep=";", )

scenarios = df["Scenario"].unique()




df.drop(df[(~df.Attribute.str.contains('VAR')) | (df.Attribute.str.endswith('M'))].index, inplace = True)
df["Pv"] = pd.to_numeric(df["Pv"].str.replace(",", "."))

print(df["Period"])
df = df[df["Period"].isin(["2018", "2020", "2025", "2030", "2035", "-"])]


# compile the list of dataframes you want to merge
data_frames = [df[df["Scenario"] == s][["Attribute", "Commodity", "Process", "Region", "Timeslice", "Period", "Pv"]].rename(columns={"Pv":f"Pv_{str(i).zfill(2)}"}) for i,s in enumerate(scenarios)]
# data_frames = data_frames[:2]
# data_frames = zip(data_frames, range(len(data_frames)))

# df_merged = reduce(lambda  left,right: pd.merge(left,right,on=["Attribute", "Commodity", "Process", "Region", "Timeslice", "Period"],
#                                             how='outer'), data_frames)

matrix_corr = np.empty(shape=(len(scenarios), len(scenarios)), dtype='float')
list_corr = []

for i in range(len(data_frames)):
    matrix_corr[i,i] = 1
    for j in range(i+1, len(data_frames)):
        df_ij = pd.merge(data_frames[i], data_frames[j], on=["Attribute", "Commodity", "Process", "Region", "Timeslice", "Period"], how='outer')[[f"Pv_{str(i).zfill(2)}", f"Pv_{str(j).zfill(2)}"]]
        corr = df_ij.corr()
        matrix_corr[i,j] = corr.at[f"Pv_{str(i).zfill(2)}", f"Pv_{str(j).zfill(2)}"]
        matrix_corr[j,i] = corr.at[f"Pv_{str(i).zfill(2)}", f"Pv_{str(j).zfill(2)}"]
        list_corr.append(matrix_corr[i,j]*10000)

print(matrix_corr)
indices_1d = np.argpartition(matrix_corr, 10, axis=None)[:10]
indices_2d = np.unravel_index(indices_1d, matrix_corr.shape)
least_three = matrix_corr[indices_2d]

print('least three values : ', least_three)
print('indices : ', *zip(*indices_2d) )





import math
from matplotlib import pyplot as plt

bins = np.linspace(math.ceil(min(list_corr)), 
                   math.floor(max(list_corr)),
                   100) # fixed number of bins

# plt.xlim([min(list_corr)-5, max(list_corr)+5])

plt.hist(list_corr, bins=bins, alpha=0.5)
plt.xlabel('variable X (20 evenly spaced bins)')
plt.ylabel('count')

plt.show()





cluster = []
used = []
for i in range(len(data_frames)):
    if i not in used:
        cluster.append([i])
        used.append(i)
        for j in range(i+1, len(data_frames)):
            if j not in used and matrix_corr[i,j] > 0.9995:
                cluster[-1].append(j)
                used.append(j)

print(cluster)

# for i in range(1, 2):
#     df_i = df[df["Scenario"] == f"scenarios_diag~000{i}"][["Attribute", "Commodity", "Process", "Region", "Timeslice", "Period", "Pv"]]
#     for j in range(1,3):
#         df_j = df[df["Scenario"] == f"scenarios_diag~000{j}"][["Attribute", "Commodity", "Process", "Region", "Timeslice", "Period", "Pv"]]
#         n1, n2 = f"{str(i).zfill(2)}", f"{str(j).zfill(2)}"
#         df_full = pd.merge(df_i, df_j, on=["Attribute", "Commodity", "Process", "Region", "Timeslice", "Period"], how="outer", suffixes=[n1, n2])
#         print(df_full)

        # df_comp = df_i.compare(df_j)
        # print(df_comp)
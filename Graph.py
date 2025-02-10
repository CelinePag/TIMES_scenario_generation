# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:24:17 2025

@author: celinep
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

PATH_OUTPUT = "output.xlsx"

# df_matrix_medoid = pd.read_excel(PATH_OUTPUT, sheet_name="output_medoid").fillna(0)
# df_matrix = pd.read_excel(PATH_OUTPUT, sheet_name="output_bis").fillna(0)

df_comparison = pd.read_excel(PATH_OUTPUT, sheet_name="comparison").fillna(0)


styles = ["-", "--", ".-", "-.", "-x", "-o", "-.", "-^", "-o", "-v"]
styles_dict = {"NOR1":"-", "NOR2":"--", "NOR3":".-", "NOR4":"-.", "NOR5":"-x"}
color_it = ["", "green", "orange", "violet", "red", "blue", "purple", "grey", "lila", ""]
# fig, axs = plt.subplots(2, 2, layout='constrained', figsize=(12,6), sharey=True)
# axe = axs.ravel()
# for j,k in enumerate([1,2,3,5]):
#     ax = axe[j]
#     db = df_matrix[df_matrix["diag"]]
#     for gr in range(1, k+1):
#         db_bis = db[db[f"g{k}"]==gr]
#         db_bis_r = db_bis[db_bis[f"p{k}"] > 0.0005]
#         db_bis_rest = db_bis[db_bis[f"p{k}"]==0]
#         db_bis_rest.plot.scatter(x="scenario", y="obj", ax=ax, color=color_it[gr])
#         db_bis_r.plot.scatter(x="scenario", y="obj", ax=ax, color=color_it[gr], s=150,edgecolors='black')
        

#     ax.set_title(f"K={k}")
#     # ax.set_xlim(2028, 2050)
#     # ax.set_xticks(ticks=[2030+i*5 for i in range(5)], labels=[str(2030+i*5) for i in range(5)], rotation=45)
#     # ax.set_xticklabels([2030, "", 2040, "", 2050] )
#     # ax.set_xticks(ticks=[i for i in range(26)], minor=True)
#     ax.grid(visible=True, which="major", axis="y")
#     ax.grid(visible=True, which="both", axis="x")
#     ax.set_ylabel("Objective value")
#     ax.set_xlabel(None)
# plt.show()


# fig, axs = plt.subplots(3, 2, layout='constrained', figsize=(12,8), sharey=True, sharex=True)
# axe = axs.ravel()
# for j,k in enumerate([2,3,5]):
#     for j2, df in enumerate([df_matrix, df_matrix_medoid]):
#         ax = axe[j*2+j2]
#         db = df[df["diag"]]
#         for gr in range(1, k+1):
#             db_bis = db[db[f"g{k}"]==gr]
#             db_bis_r = db_bis[db_bis[f"p{k}"] > 0.0005]
#             db_bis_rest = db_bis[db_bis[f"p{k}"]==0]
#             db_bis_rest.plot.scatter(x="scenario", y="obj", ax=ax, color=color_it[gr])
#             db_bis_r.plot.scatter(x="scenario", y="obj", ax=ax, color=color_it[gr], s=150,edgecolors='black')
        
#         ax.set_title(f"K={k}")
#         ax.set_xlim(0, 25)
#         # ax.set_xticks(ticks=[2030+i*5 for i in range(5)], labels=[str(2030+i*5) for i in range(5)], rotation=45)
#         # ax.set_xticklabels([2030, "", 2040, "", 2050] )
#         # ax.set_xticks(ticks=[i for i in range(26)], minor=True)
#         ax.grid(visible=True, which="major", axis="y")
#         ax.grid(visible=True, which="both", axis="x")
#         ax.set_ylabel("Objective value")
#         if j == 2:
#             if j2 ==0:
#                 ax.set_xlabel("CSSC")
#             else:
#                 ax.set_xlabel("K-medoids")
#         else:
#             ax.set_xlabel(None)

# # fig.suptitle("sup")
# # axs.set_title("ax")
# plt.show()
# plt.show()


# fig, axs = plt.subplots(2, 2, layout='constrained', figsize=(12,6), sharey=True)
# axe = axs.ravel()
# for j,k in enumerate([1,2,3,5]):
#     ax = axe[j]
#     db = df_matrix
#     db.plot.scatter(x="scenario", y="obj", ax=ax, color="grey")
#     for gr in range(1, k+1):
#         db_bis = db[db[f"g{k}"]==gr]
#         db_bis_r = db_bis[db_bis[f"p{k}"] > 0.0005]
#         db_bis_rest = db_bis[db_bis[f"p{k}"]==0]
#         db_bis_rest.plot.scatter(x="scenario", y="obj", ax=ax, color=color_it[gr])
#         db_bis_r.plot.scatter(x="scenario", y="obj", ax=ax, color=color_it[gr], s=150,edgecolors='black')
#         num_r = db_bis_r["scenario"].iloc[0]
#         print(num_r)
#         db[db["fix_var"]==num_r].plot.scatter(x="scenario", y="obj", ax=ax, color="grey",edgecolors=color_it[gr])
       

#     ax.set_title(f"K={k}")
#     # ax.set_xlim(2028, 2050)
#     # ax.set_xticks(ticks=[2030+i*5 for i in range(5)], labels=[str(2030+i*5) for i in range(5)], rotation=45)
#     # ax.set_xticklabels([2030, "", 2040, "", 2050] )
#     # ax.set_xticks(ticks=[i for i in range(26)], minor=True)
#     ax.grid(visible=True, which="major", axis="y")
#     ax.grid(visible=True, which="both", axis="x")
#     ax.set_ylabel("Objective value")
#     ax.set_xlabel(None)
# plt.show()









#%% df_comparison

size = (8, 3) 
fig, ax = plt.subplots(layout='constrained', figsize=size)
width = 0.085
coeff_pos = 1

list_K = [1,2,3,5,10,15]
dict_list_K = {1:0,2:1,3:2,5:3,10:4,15:5, 20:6}


delta_K = 2
delta_style = (0.5)

color = ["red", "green", "orange"]
methods = ["CSSC", "random"]


posi_xticks = [delta_K*i+j*delta_style for j in [0,1] for i in range(len(list_K))]
print(posi_xticks)

for i,style in enumerate(methods):
    boxprops = dict(linestyle='-', linewidth=1, color='black')
    medianprops = dict(linestyle='-', linewidth=1, color="black")
    whiskerprops = dict(linestyle='-', linewidth=1, color='black')
    db = df_comparison[df_comparison['type'] == style][['K', 'gap']]
    # db.plot.box(column)

    if style == "random":
        bp_dict = db.boxplot(column="gap", by="K", ax=ax, sym=".", rot=45,
                         medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops, patch_artist=True, return_type='both',
                         positions=[delta_K*j for j in range(len(db["K"].unique()))])
    else:
        print(db["K"])
        c = 1 if i ==0 else -1
        db["K"] = db["K"].apply(lambda x: delta_K*dict_list_K[x]+(c)*delta_style)
        print(db["K"])
        bp_dict = db.plot.scatter(y="gap", x="K", ax=ax, rot=45, color=color[i])
    
    

    # db.columns = db.columns.droplevel(0)
    # y_offset = 1
    # list_bar = {round(float(0*coeff_pos+it*width), 3):None, round(float(1*coeff_pos+it*width), 3):None, round(float(2*coeff_pos+it*width), 3):None, round(float(3*coeff_pos+it*width), 3):None}
    # for bar in ax.patches:
    #     bx = round(bar.get_x(), 3)
    #     if bar.get_y() >= 0 and bar.get_height() >= 0 and bx in list_bar.keys():
    #         if (list_bar[bx] == None or list_bar[bx].get_y() < bar.get_y()):
    #             list_bar[bx] = bar
    
    # for bx, bar in list_bar.items():
    #     ax.text(
    #           bar.get_x() + bar.get_width() / 2,
    #           bar.get_height() + bar.get_y() + y_offset,
    #           it*coeff_pos+1,
    #           ha='center',
    #           color='black',
    #           size=8)

plt.legend(["CSSC", "MC"], loc="upper right", ncols=1)

# ax.set_xticks(ticks=[i*delta_K+delta_style for i in range(len(list_K))], labels=[i for i in range(1, len(list_K+1))], rotation=45)
print(6*["medoid", "MC", "CSSC"])
ax.set_xticks(ticks=posi_xticks, labels =6*["MC"]+6*["CSSC"], minor=False)
# ax.set_xticks(ticks=posi_xticks, minor=True)
ax.grid(visible=True, which="major", axis="x")
ax.grid(visible=True, which="minor", axis="x")
nspace= 22
ax.set_xlabel('K=1' + nspace*' ' +  'K=2' + nspace*' ' + 'K=3' + nspace*' ' + 'K=5' + nspace*' ' + 'K=10' + nspace*' ' + 'K=15') #f"Grouped by {by}")

ax.set_axisbelow(True)
ax.set_xlim(-0.5, 11)
ax.set_ylabel("Gap [%]")

fig.suptitle(None)
ax.set_title(None)

plt.show()







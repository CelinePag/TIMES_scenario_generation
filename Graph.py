# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:24:17 2025

@author: celinep
"""

import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.ticker import FuncFormatter, FormatStrFormatter
import matplotlib.ticker as ticker
import pandas as pd



PATH_OUTPUT = "output.xlsx"

# df_matrix_medoid = pd.read_excel(PATH_OUTPUT, sheet_name="output_medoid").fillna(0)
# df_matrix = pd.read_excel(PATH_OUTPUT, sheet_name="output_bis").fillna(0)


colors = {"random":"blue", "medoid":"orange",
           "CSSC":"red", "CSSC_sparse_998":"orange", "CSSC_sparse_995":"purple",
             "medoid_distance":"grey", "spect50":"green", "CSSC-2S":"cyan"}

styles = ["-", "--", ".-", "-.", "-x", "-o", "-.", "-^", "-o", "-v"]
styles_dict = {"NOR1":"-", "NOR2":"--", "NOR3":".-", "NOR4":"-.", "NOR5":"-x"}
color_it = ["", "green", "orange", "violet", "red", "blue", "purple", "grey", "lila", ""]


def gap_methods(list_K=[1,2,3,4,5,10,15,20,25,35,45,55], methods=["random", "medoid", "spect50", "CSSC", "CSSC_sparse_998", "CSSC-2S"], v=1, save=True, show=False):
    size = (12, 4) 
    width = 0.085
    coeff_pos = 1
    # delta_K = 4
    # delta_style = 0.5
    delta_K = 2
    delta_style = 0.3
    dict_list_K = {1:0,2:1,3:2,4:3,5:4,10:5,15:6,20:7,25:8,35:9,45:10,55:11}
    dict_list_K = {j:i for i,j in enumerate(list_K)}
    methods_name = {"random":"SAA", "medoid":"Med.", "spect50":"spect50", "CSSC":"CSSC", "CSSC_sparse_998":"CSSC-98", "CSSC-2S":"CSSC-2S"}

    col_name = "gap_abs"
    df_comparison = pd.read_excel(PATH_OUTPUT, sheet_name="comparison-64").fillna(0)
    df_comparison_SAA = pd.read_excel(PATH_OUTPUT, sheet_name="comparison-64-SAA").fillna(0)
    # col_name = "gap"
    size_marker = 70
    fig, ax = plt.subplots(layout='constrained', figsize=size)
    for i,style in enumerate(methods):
        if style == "random":
            # df_comparison_SAA
            df_comparison_SAA = df_comparison_SAA[(df_comparison_SAA['K'].isin(list_K))]
            average_gap = pd.DataFrame(columns=["K", "av_gap"])
            for k in list_K:
                # average_gap = average_gap.append(, ignore_index=True)
                average_gap.loc[len(average_gap)] = {"K":k, "av_gap":df_comparison_SAA[(df_comparison_SAA['K'] == k)]["GAP2"].mean()}
            if v==1:
                df_comparison_SAA["K"] = df_comparison_SAA["K"].apply(lambda x: delta_K*dict_list_K[x]+(i)*delta_style)
                average_gap["K"] = average_gap["K"].apply(lambda x: delta_K*dict_list_K[x]+(i)*delta_style)

            elif v==2:
                df_comparison_SAA["K"] = df_comparison_SAA["K"].apply(lambda x: delta_style*dict_list_K[x]+(i)*delta_K)
                average_gap["K"] = average_gap["K"].apply(lambda x: delta_style*dict_list_K[x]+(i)*delta_K)
            #UB
            # df_comparison_SAA.plot.scatter(y="UB_GAP", x="K", ax=ax, rot=90, s=40, color="grey", marker="s", linewidth=0.7, edgecolor='black', yerr="UB_error")
            # # LB
            # df_comparison_SAA.plot.scatter(y="LB_GAP", x="K", ax=ax, rot=90, s=40, color="blue", marker="s", linewidth=0.7, edgecolor='black', yerr="LB_error")
            
            # GAP
            print(average_gap)
            df_comparison_SAA.plot.scatter(y="GAP2", x="K", ax=ax, rot=90, s=size_marker, color="blue", marker="s", linewidth=0.7, edgecolor='black', yerr="GAP2_error")
            average_gap.plot.scatter(y="av_gap", x="K", ax=ax, rot=90, s=600, color="black", marker="_",zorder=3)


            

            # bp_dict = db.boxplot(column=col_name, by="K", ax=ax, sym=".", rot=90,
            #                  medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops, patch_artist=True, return_type='both',
            #                  positions=[delta_K*j for j in range(len(db["K"].unique()))])
        else:
            db = df_comparison[(df_comparison['type'] == style) & (df_comparison['K'].isin(list_K))][['K', 'gap', "gap_abs"]]
            c = 1 if i ==0 else -1
            if v==1:
                db["K"] = db["K"].apply(lambda x: delta_K*dict_list_K[x]+(i)*delta_style)
            elif v==2:
                db["K"] = db["K"].apply(lambda x: delta_style*dict_list_K[x]+(i)*delta_K)
            bp_dict = db.plot.scatter(y=col_name, x="K", ax=ax, rot=90, s=size_marker, color=colors[style], marker="s", linewidth=0.7, edgecolor='black')

    # plt.legend(methods_name, loc="upper right", ncols=1)
    if v==1:
        lab = []
        for m in methods:
            lab += len(list_K)*[methods_name[m]]
        posi_xticks = [delta_K*i+j*delta_style for j in range(len(methods)) for i in range(len(list_K))]
    elif v==2:
        lab = []
        for m in methods:
            lab += list_K
        posi_xticks = [delta_style*i+j*delta_K   for j in range(len(methods)) for i in range(len(list_K))]
    
    ax.set_xticks(ticks=posi_xticks, labels = lab, minor=False)
    ax.grid(visible=True, which="major", axis="x")
    ax.grid(visible=True, which="minor", axis="x")
    ax.grid(visible=True, which="major", axis="y")

    if v==1:
        nspace= 32
        label = [f"K={i}" + nspace*' ' for i in list_K[:-1]] + [f"K={list_K[-1]}"]
        ax.set_xlabel("".join(label)) #f"Grouped by {by}")
        ax.set_axisbelow(True)
        ax.set_xlim(-0.5, delta_K*(len(list_K)-1)+(len(methods)-1)*delta_style+0.5)    
    elif v==2:
        nspace= 40
        label = [f"{methods_name[m]}" + nspace*' ' for m in methods[:-1]] + [f"{methods_name[methods[-1]]}"]
        ax.set_xlabel("".join(label)) #f"Grouped by {by}")
        ax.set_axisbelow(True)
        ax.set_xlim(-0.5, delta_K*(len(methods)-1)+(len(list_K)-1)*delta_style+0.5)
    if col_name == "gap_abs":
        ax.set_ylim(0, 8)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    else:
        ax.set_ylim(-7, 7) 
    ax.set_ylabel("Implementation error [%]")
    fig.suptitle(None)
    ax.set_title(None)

    if show:
        plt.show()
    if save:
        plt.savefig(f'figures/gap{v}.png', bbox_inches="tight")


solve_sp_time = {1:32, 2:74, 3:95, 4:240, 5:346, 10:2172, 15:4335, 20:16198, 64:95049}
solve_sp_times_full_fixed_first_stage = 1500
solve_precompute_time = {"CSSC":88627, "CSSC-98":17866, "CSSC-95":3912, "SAA":0}
M = 10

def get_graph_solve_time(chosen_k=[5, 10, 20], type_matrix = ["CSSC", "CSSC-98", "SAA"], save=True, show=False):
    
    delta_K = 2.2
    delta_style = 0.55
    linewidth = 0.5
    width = 0.5

    lower_bars_precompute = [solve_precompute_time[m] for m in type_matrix for k in chosen_k ] + [0]
    upper_bars_solve = []
    for m in type_matrix:
        for k in chosen_k:
            if m == "SAA":
                upper_bars_solve.append(M*(solve_sp_time[k]+solve_sp_times_full_fixed_first_stage))
            else:
                upper_bars_solve.append(solve_sp_time[k])
    upper_bars_solve += [solve_sp_time[64]]


    fig, ax = plt.subplots(figsize=(12,5))
    bottom = np.zeros(len(chosen_k)*len(type_matrix)+1)
    labels = [m for m in type_matrix for k in chosen_k] + ["Full SP-64"]
    posi_xticks = [delta_K*i+j*delta_style for j in range(len(type_matrix)) for i in range(len(chosen_k))] + [delta_K*len(chosen_k)+(len(type_matrix)-1)*delta_style]

    ax.bar(posi_xticks, lower_bars_precompute, width=width, label="Precomputing", bottom=bottom, linewidth=linewidth, edgecolor='black', zorder=3, color="#014693")
    ax.bar(posi_xticks, upper_bars_solve, width=width, label="Solve SP", bottom=lower_bars_precompute, linewidth=linewidth, edgecolor='black', zorder=3, color="#f58c36")

    ax.set_title("Title")
    ax.legend(loc="upper right")
    ax.set_xticks(ticks=posi_xticks, labels = labels, minor=False, rotation=45)
    ax.grid(visible=True, which="major", axis="y", zorder=0)
    ax.set_ylabel("Time [s]")
    nspace = 42
    label = [f"K={i}" + nspace*' ' for i in chosen_k] + [int(nspace/2)*' ']
    ax.set_xlabel("".join(label)) #f"Grouped by {by}")
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: '{:,}'.format(int(x)).replace(",", " ")))
    fig.suptitle(None)
    ax.set_title(None)
    if show:
        plt.show()
    if save:
        plt.savefig('figures/solve_time.png', bbox_inches="tight")



def graph_solve_time_VS_gap(methods=["random", "CSSC", "CSSC_sparse_998"], save=True, show=False):
    PATH_OUTPUT = "output.xlsx"

    dict_translate_names = {"CSSC_sparse_998":"CSSC-98", "CSSC":"CSSC", "random":"SAA"}

    df_comparison = pd.read_excel(PATH_OUTPUT, sheet_name="comparison-64").fillna(0)

    size = (6, 5.5)
    size = (12, 4) 
    size_pt = 50
    fig, ax = plt.subplots(layout='constrained', figsize=size)
    list_K = [1,2,3,4,5,10,15,20]
    size_marker = 70

    for m in methods:
        if "random" == m:
            df_comparison_SAA = pd.read_excel(PATH_OUTPUT, sheet_name="comparison-64-SAA").fillna(0)
            # df_comparison_SAA["GAP2"] = df_comparison_SAA["GAP2"] + df_comparison_SAA["GAP2_error"]
            average_gap = pd.DataFrame(columns=["K", "av_gap", "yerror"])
            for k in list_K:
                average_gap.loc[len(average_gap)] = {"K":k, "av_gap":df_comparison_SAA[(df_comparison_SAA['K'] == k)]["GAP2"].mean(), "yerror":df_comparison_SAA[(df_comparison_SAA['K'] == k)]["GAP2_error"].max()}
            df_comparison_SAA["solve_time"] = df_comparison_SAA["K"].apply(lambda x: M*(solve_sp_time[x]+solve_sp_times_full_fixed_first_stage))
            average_gap["solve_time"] = average_gap["K"].apply(lambda x: M*(solve_sp_time[x]+solve_sp_times_full_fixed_first_stage))
            print(df_comparison_SAA.columns)
            # idx = df_comparison_SAA.groupby(["K", "solve_time"])['GAP2'].transform("min") == df_comparison_SAA['GAP2']
            # print(idx)
            # df_comparison_SAA = df_comparison_SAA[idx]
            print(df_comparison_SAA[["K", "solve_time", "GAP2", "GAP2_error"]])
            # df_comparison_SAA.plot.scatter(x="GAP2", y="solve_time", ax=ax, rot=90, s=40, color=colors[m], marker="s", linewidth=0.7, edgecolor='black', zorder=3, label=dict_translate_names[m], xerr="yerror")
            average_gap.plot.scatter(x="av_gap", y="solve_time", ax=ax, rot=90, s=size_marker, color=colors[m], marker="s", linewidth=0.7, edgecolor='black', zorder=3, label=dict_translate_names[m])#, xerr="yerror")
        else:
            db = df_comparison[(df_comparison['type'] == m) & (df_comparison['K'] <= 20)][['K', "gap_abs"]]
            db["solve_time"] = db["K"].apply(lambda x: solve_sp_time[x]+solve_precompute_time[dict_translate_names[m]])
            db["GAP2"] = db["gap_abs"]
            db.plot.scatter(x="GAP2", y="solve_time", ax=ax, rot=90, s=size_marker, color=colors[m], marker="s", linewidth=0.7, edgecolor='black', zorder=3, label=dict_translate_names[m])

    ax.scatter(x=[0], y=[solve_sp_time[64]], s=size_pt, color="black", marker="s", linewidth=0.7, edgecolor='black', zorder=3, label='full SP')

    ax.grid(visible=True, which="major", axis="both", zorder=0)
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: '{:,}'.format(int(x)).replace(",", " ")))
    ax.set_ylabel("Time [s]")
    ax.set_xlabel("Implementation error [%]")
    plt.legend()
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 180000)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%g'))

    if show:
        plt.show()
    if save:
        plt.savefig('figures/solve_time_VS_gap.png', bbox_inches="tight")



# gap_methods(list_K=[1,2,3,4,5,10,20], methods=["random", "CSSC", "CSSC_sparse_998"], v=1)
# gap_methods(list_K=[1,2,3,4,5,10,20], methods=["random", "CSSC", "CSSC_sparse_998"], v=2)

# get_graph_solve_time()
graph_solve_time_VS_gap()

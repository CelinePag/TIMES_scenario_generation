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


solve_sp_time = {1:32, 2:74, 3:95, 4:240, 5:346, 10:2172, 15:4335, 20:16198, 64:95049}
solve_sp_times_full_fixed_first_stage = 1500
solve_precompute_time = {"CSSC":88627, "CSSC-98":17866, "CSSC-95":3912, "SAA":0}
M = 10


class Graphs():
    def __init__(self, path_data, sheetname="comparison-64", sheetname_SAA="comparison-64-SAA"):
        self.df_comparison = pd.read_excel(path_data, sheet_name=sheetname).fillna(0)
        self.df_comparison_SAA = pd.read_excel(path_data, sheet_name=sheetname_SAA).fillna(0)
        self.methods_name = {"random":"SAA", "medoid":"Med.", "spect50":"spect50", "CSSC":"CSSC", "CSSC_sparse_998":"CSSC-98", "CSSC-2S":"CSSC-2S"}
    

    def get_gap(self, list_methods, list_K, v=1, save=True, show=False):
        """ Create a graph with implementation error """

        dict_list_K = {j:i for i,j in enumerate(list_K)}
        # Caracteristics figure
        size = (12, 4)
        delta_K = 2
        delta_style = 0.3
        size_marker = 70
        fig, ax = plt.subplots(layout='constrained', figsize=size)

        # get and plot data
        col_name = "gap_abs"
        for i, method in enumerate(list_methods):
            if method == "random":
                df_comparison_SAA = self.df_comparison_SAA[(self.df_comparison_SAA['K'].isin(list_K))]
                average_gap = pd.DataFrame(columns=["K", "av_gap"])
                for k in list_K:
                    average_gap.loc[len(average_gap)] = {"K":k, "av_gap":df_comparison_SAA[(df_comparison_SAA['K'] == k)]["GAP2"].mean()}
                if v==1:
                    df_comparison_SAA["K"] = df_comparison_SAA["K"].apply(lambda x: delta_K*dict_list_K[x]+(i)*delta_style)
                    average_gap["K"] = average_gap["K"].apply(lambda x: delta_K*dict_list_K[x]+(i)*delta_style)
                elif v==2:
                    df_comparison_SAA["K"] = df_comparison_SAA["K"].apply(lambda x: delta_style*dict_list_K[x]+(i)*delta_K)
                    average_gap["K"] = average_gap["K"].apply(lambda x: delta_style*dict_list_K[x]+(i)*delta_K)

                df_comparison_SAA.plot.scatter(y="GAP2", x="K", ax=ax, rot=90, s=size_marker, color=colors[method], marker="s", linewidth=0.7, edgecolor='black', yerr="GAP2_error")
                average_gap.plot.scatter(y="av_gap", x="K", ax=ax, rot=90, s=600, color="black", marker="_",zorder=3)
            else:
                db = self.df_comparison[(self.df_comparison['type'] == method) & (self.df_comparison['K'].isin(list_K))][['K', 'gap', "gap_abs"]]
                c = 1 if i ==0 else -1
                if v==1:
                    db["K"] = db["K"].apply(lambda x: delta_K*dict_list_K[x]+(i)*delta_style)
                elif v==2:
                    db["K"] = db["K"].apply(lambda x: delta_style*dict_list_K[x]+(i)*delta_K)
                db.plot.scatter(y=col_name, x="K", ax=ax, rot=90, s=size_marker, color=colors[method], marker="s", linewidth=0.7, edgecolor='black')

        # arrange graph
        if v==1:
            lab = []
            for m in list_methods:
                lab += len(list_K)*[self.methods_name[m]]
            posi_xticks = [delta_K*i+j*delta_style for j in range(len(list_methods)) for i in range(len(list_K))]
            nspace= 32
            label = [f"K={i}" + nspace*' ' for i in list_K[:-1]] + [f"K={list_K[-1]}"]
            ax.set_xlabel("".join(label)) #f"Grouped by {by}")
            ax.set_axisbelow(True)
            ax.set_xlim(-0.5, delta_K*(len(list_K)-1)+(len(list_methods)-1)*delta_style+0.5)
        elif v==2:
            lab = []
            for m in list_methods:
                lab += list_K
            posi_xticks = [delta_style*i+j*delta_K for j in range(len(list_methods)) for i in range(len(list_K))]
            nspace= 40
            label = [f"{self.methods_name[m]}" + nspace*' ' for m in list_methods[:-1]] + [f"{self.methods_name[list_methods[-1]]}"]
            ax.set_xlabel("".join(label)) #f"Grouped by {by}")
            ax.set_axisbelow(True)
            ax.set_xlim(-0.5, delta_K*(len(list_methods)-1)+(len(list_K)-1)*delta_style+0.5)
        
        ax.set_xticks(ticks=posi_xticks, labels = lab, minor=False)
        ax.grid(visible=True, which="major", axis="x")
        ax.grid(visible=True, which="minor", axis="x")
        ax.grid(visible=True, which="major", axis="y")

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



    def get_solve_time(self, list_methods=["CSSC", "CSSC-98", "SAA"], list_K=[5, 10, 20], save=True, show=False):
        """ Create a bar graph with solve time decomposed in precomputing and solving SP"""

        # Caracteristics figure
        delta_K = 2.2
        delta_style = 0.55
        linewidth = 0.5
        width = 0.5
        fig, ax = plt.subplots(figsize=(12,5))

        # get data
        lower_bars_precompute = [solve_precompute_time[m] for m in list_methods for k in list_K ] + [0]
        upper_bars_solve = []
        for m in list_methods:
            for k in list_K:
                if m == "SAA":
                    upper_bars_solve.append(M*(solve_sp_time[k]+solve_sp_times_full_fixed_first_stage))
                else:
                    upper_bars_solve.append(solve_sp_time[k])
        upper_bars_solve += [solve_sp_time[64]]
    

        bottom = np.zeros(len(list_K)*len(list_methods)+1)
        posi_xticks = [delta_K*i+j*delta_style for j in range(len(list_methods)) for i in range(len(list_K))] + [delta_K*len(list_K)+(len(list_methods)-1)*delta_style]
        ax.bar(posi_xticks, lower_bars_precompute, width=width, label="Precomputing", bottom=bottom, linewidth=linewidth, edgecolor='black', zorder=3, color="#014693")
        ax.bar(posi_xticks, upper_bars_solve, width=width, label="Solve SP", bottom=lower_bars_precompute, linewidth=linewidth, edgecolor='black', zorder=3, color="#f58c36")

        # Arrange figure
        labels = [m for m in list_methods for k in list_K] + ["Full SP-64"]
        ax.set_title("Title")
        ax.legend(loc="upper right")
        ax.set_xticks(ticks=posi_xticks, labels = labels, minor=False, rotation=45)
        ax.grid(visible=True, which="major", axis="y", zorder=0)
        ax.set_ylabel("Time [s]")
        nspace = 42
        label = [f"K={i}" + nspace*' ' for i in list_K] + [int(nspace/2)*' ']
        ax.set_xlabel("".join(label)) #f"Grouped by {by}")
        ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: '{:,}'.format(int(x)).replace(",", " ")))
        fig.suptitle(None)
        ax.set_title(None)
        if show:
            plt.show()
        if save:
            plt.savefig('figures/solve_time.png', bbox_inches="tight")

    
   

    def get_gap_vs_solve_time(self, list_K, list_methods, save=True, show=False):
        """ Create a graph with implementation error in xaxis and time to solve in yaxis"""

        # Caracteristics figure
        size = (12, 4) 
        size_pt = 50
        size_marker = 70
        fig, ax = plt.subplots(layout='constrained', figsize=size)

        for m in list_methods:
            if "random" == m:
                average_gap = pd.DataFrame(columns=["K", "av_gap", "yerror"])
                df_comparison_SAA = self.df_comparison_SAA[(self.df_comparison_SAA['K'].isin(list_K))]
                for k in list_K:
                    average_gap.loc[len(average_gap)] = {"K":k, "av_gap":df_comparison_SAA[(df_comparison_SAA['K'] == k)]["GAP2"].mean(), "yerror":df_comparison_SAA[(df_comparison_SAA['K'] == k)]["GAP2_error"].max()}
                df_comparison_SAA["solve_time"] = df_comparison_SAA["K"].apply(lambda x: M*(solve_sp_time[x]+solve_sp_times_full_fixed_first_stage))
                average_gap["solve_time"] = average_gap["K"].apply(lambda x: M*(solve_sp_time[x]+solve_sp_times_full_fixed_first_stage))
                average_gap.plot.scatter(x="av_gap", y="solve_time", ax=ax, rot=90, s=size_marker, color=colors[m], marker="s", linewidth=0.7, edgecolor='black', zorder=3, label=self.methods_name[m])#, xerr="yerror")
            else:
                db = self.df_comparison[(self.df_comparison['type'] == m) & (self.df_comparison['K'].isin(list_K))][['K', "gap_abs"]]
                db["solve_time"] = db["K"].apply(lambda x: solve_sp_time[x]+solve_precompute_time[self.methods_name[m]])
                db["GAP2"] = db["gap_abs"]
                db.plot.scatter(x="GAP2", y="solve_time", ax=ax, rot=90, s=size_marker, color=colors[m], marker="s", linewidth=0.7, edgecolor='black', zorder=3, label=self.methods_name[m])
        ax.scatter(x=[0], y=[solve_sp_time[64]], s=size_pt, color="black", marker="s", linewidth=0.7, edgecolor='black', zorder=3, label='full SP')

        # Arrange figure
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



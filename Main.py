# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 11:04:38 2025

@author: celinep
"""


import pandas as pd
import WriteFile as wf
import Clustering as cl
import numpy as np
import copy
from pathlib import Path
import itertools
import shutil
import json
import copy



# FOLDER_SUBXLS = r"C:\Veda\Veda_models\IFE-NO-2024.08.27_original\IFE-NO-2024.08.27_original\SuppXLS"
# FOLDER_SUBXLS = r"C:\Users\celinep\Documents\GitHub\TIMES_scenario_generation\SuppXLS"


NAME_STUDY = "Stocha"
PATH_TIMES = r"C:\Veda\Veda_models\IFE-NO-2024.08.27_simplified\IFE-NO-2024.08.27_simplified\\"

FOLDER_SUBXLS = f"{PATH_TIMES}SuppXLS"
PATH_UNCERTAINTIES = f"{PATH_TIMES}uncertainties.xlsx"
PATH_GDX = f"C:\Veda\GAMS_WrkTIMES\{NAME_STUDY}"
PATH_CASES = f"{PATH_TIMES}AppData\Cases.json"
PATH_GROUPS = f"{PATH_TIMES}AppData\Groups.json"
PATH_SETTINGS = f"{PATH_TIMES}SysSettings.xlsx"
PATH_SCENARIOS = f"{PATH_TIMES}SuppXLS\\" + "Scen_stocha_uncertainties.xlsx"
PATH_RESULTS = f"{PATH_TIMES}Exported_files\matrix"



class Scenarios():
    def __init__(self, uncert:dict, year_second_stage:int,rewrite=True):
        self.N = 1
        for el in uncert:
            self.N = self.N * len(el[1])
        self.list_scenarios = []
        self.uncert = uncert
        self.year_second_stage = year_second_stage
        self.id_scenariogroupe = {}
        if rewrite:
            self.update_SP_full_file()
            self.update_PAR_full_file()

    def update_SP_full_file(self):
        combinations = get_combination(self.uncert)
        sows = {j+1:1/self.N for j in range(self.N)}
        stages = {2:self.year_second_stage}
        
        sett = wf.ExcelSettings(PATH_SCENARIOS, ws_name="stochastic", data_only=False)
        sett.SOW(stages, sows)
        sett.close()
        ws_s = wf.ExcelScenarios(PATH_SCENARIOS, "source_scenarios", data_only=False, delete_old=False)
        ws_s.write_scenarios(combinations, new_scenarios=False)
        ws_s.close()
    
    def update_PAR_full_file(self):
        pass # TODO
        



    def move_gdx(self, name_case_par="scenarios_diag"):
        for n in range(1, self.N+1):
            name_s = f"{name_case_par}~{str(n).zfill(4)}"
            src_file = f"{PATH_GDX}\{name_case_par}\{name_s}\GAMSSAVE\{name_s}.gdx"
            dst_file = f"{PATH_TIMES}\AppData\GAMSSAVE\{name_s}.gdx"
            shutil.copyfile(src_file, dst_file)
            
    def create_cases(self, name_case_par="scenarios_diag"):
        with open(PATH_CASES, mode="r", encoding="utf-8") as read_file:
            cases = json.load(read_file)
        
        model_case = None
        new_cases = []
        for c in cases:
            if c["Name"] == name_case_par:
                model_case = copy.deepcopy(c)
                new_cases.append(model_case)
                break                

        nstart_id = 1000
        for n in range(1, self.N+1):
            new_case = copy.deepcopy(model_case)
            new_case["CaseId"] = nstart_id + n
            new_case["Name"] = f"{name_case_par}_{n}"
            new_case["Description"] = f"scenarios with fixed variables from {n}"
            new_case["FixResultFileName"] = "True"
            new_case["FixResultInfo"]["WorkTimesFolderPath"] = f"{PATH_TIMES}AppData\\GAMSSAVE"
            new_case["FixResultInfo"]["IsApplyFixResult"] = True
            new_case["FixResultInfo"]["GdxElasticDermands"]["IsApplied"] = True
            new_case["FixResultInfo"]["GdxIre"]["IsApplied"] = True
            new_case["FixResultInfo"]["GdxUseSolution"]["FixYearsUpto"] = f"{self.year_second_stage}"
            new_case["FixResultInfo"]["GdxUseSolution"]["GdxSelectedFile"]["FileName"] = f"{name_case_par}~{str(n).zfill(4)}"
            new_case["FixResultInfo"]["GdxUseSolution"]["GdxSelectedFile"]["FilePath"] = f"{PATH_TIMES}AppData\\GAMSSAVE\\{name_case_par}~{str(n).zfill(4)}.gdx"
            new_case["FixResultInfo"]["GdxUseSolution"]["GdxSelectedFile"]["IsSelected"] = True
            new_case["FixResultInfo"]["GdxUseSolution"]["IsApplied"] = True
            new_cases.append(copy.deepcopy(new_case))
            if n > 1:
                print(n-1, new_cases[-2]["FixResultInfo"]["GdxUseSolution"]["GdxSelectedFile"]["FileName"])
            print(n, new_case["FixResultInfo"]["GdxUseSolution"]["GdxSelectedFile"]["FileName"])
            
        with open(PATH_CASES, mode="w", encoding="utf-8") as write_file:
            json.dump(new_cases, write_file)
    

    def get_costMatrix(self, file, coeff=1): 
        self.df_obj = pd.read_csv(f"{PATH_TIMES}Exported_files\\{file}.csv", sep=";", )
        self.cost_matrix = np.empty(shape=(self.N, self.N), dtype='object')
        for idx, row in self.df_obj.iterrows():
            if "_" in row["Scenario"] and row["Scenario"].split("_")[0] != "base":
                i = int(row["Scenario"].split("_")[0])-1
                j = int(row["Scenario"].split("_")[1])-1
                if i < self.N and j < self.N:
                    self.cost_matrix[i,j] = float(row["Pv"].replace(",", "."))*coeff


    def get_clusters(self, K, list_methods=["CSSC"]):
        self.cluster = {}
        self.K = K
        for method in list_methods:
            if method == "CSSC_new":
                self.cluster[method] = cl.ClusterMIP(K).get_scenarios(self.cost_matrix, new=True)
            elif method == "CSSC_old":
                self.cluster[method] = cl.ClusterMIP(K).get_scenarios(self.cost_matrix, new=False)
            elif method == "medoid_distance":
                self.cluster[method] = cl.ClusterMIP(K).get_scenarios(self.cost_matrix)
            elif method == "medoid":
                self.cluster[method] = cl.ClusterMIP(K).get_scenarios(get_combination(uncert))
            elif "random" in method:
                self.cluster[method] = cl.ClusterRandom(K).get_scenarios(self.N)
            elif method == "spectral":
                cl.ClusterNormalizedSpectral(K).get_scenarios(self.cost_matrix, e=1)

            
    def write_SP_files(self, list_methods=["CSSC_new"]):
        for method in list_methods:
            self.write_scenarios_SP(method=method)

            
    def write_scenarios_SP(self, method="CSSC_new"):
        sows = {j+1:len(item)/self.N for j, (key,item) in enumerate(self.cluster[method].items())}
        stages = {2:f"{self.year_second_stage}"}
        
        path = f"{FOLDER_SUBXLS}\Scen_stocha_uncertainties_{method}_{self.K}.xlsx"
        shutil.copyfile(PATH_SCENARIOS, path)
        
        sett = wf.ExcelSettings(path, ws_name="stochastic", data_only=False)
        sett.SOW(stages, sows)
        sett.close()
        ws_s = wf.ExcelScenarios(path, "source_scenarios", data_only=False, delete_old=False)
        ws_s.write_scenarios(self.cluster[method].keys(), new_scenarios=True)
        ws_s.close()
        



def get_combination(uncert_list):
    all_uncert = [[f"{u[0]}_{ur}" for ur in u[1]] for u in uncert_list]
    combinations = all_uncert[0]
    if len(all_uncert) > 1:
        for i in range(1, len(all_uncert)):
            combinations = list(itertools.product(combinations, all_uncert[i]))
            if i > 1:
                for i in range(len(combinations)):
                    combinations[i] = [el for el in combinations[i][0]] + [combinations[i][-1]]
    return combinations


def get_num(uncert, combi):
    for u in uncert:
        for l in u[1]:
            print(u[0],l)
            num = []
            for i,c in enumerate(combi):
                if u[0] + "_" + l in c:
                    num.append(i+1)
            print(num)
    

def main(uncert):
    N = 1
    for el in uncert:
        N = N * len(el[1])
    print(f"{N} scenarios")
    
    rewrite = False
    S = Scenarios(uncert, year_second_stage=2030, rewrite=rewrite)
    input("Run the Parametric use case of TIMES")
    S.move_gdx()
    S.create_cases()
    input("reload run manager and run the N Parametric use cases of TIMES")
    # nfile = input("save export, name file:")
    nfile = "021125_134329569"
    S.get_costMatrix(nfile, coeff=10**-5)
    
    K = 3
    list_methods = ["CSSC_new", "CSSC_old", "medoid_distance", "medoid"]
    list_methods = ["spectral"]
    # S.get_clusters(K=K, list_methods=list_methods)
    # S.write_SP_files(list_methods=list_methods)
    
    

if __name__ == "__main__":
    # Possible uncertainties: WIND, CO2TAX, HYDROGEN, DMD, ELEC, BIOMASS, LEARN, EMISSIONCAP, SOCIETE_CHANGE
    # For each uncertainty, possible levels: high, low, med
    uncert = []
    for u in ["WIND", "CO2TAX", "HYDROGEN", "DMD", "ELEC", "BIOMASS"]:#, "LEARN", "EMISSIONCAP", "SOCIETE_CHANGE"]:
        uncert.append([u, []])
        for lvl in ["high", "low"]:
            uncert[-1][1].append(lvl)
    # for u in ["WIND", "CO2TAX", "HYDROGEN"]:#, "DMD"]:#, "ELEC", "BIOMASS"]:#, "LEARN", "EMISSIONCAP", "SOCIETE_CHANGE"]:
    #     uncert.append([u, []])
    #     for lvl in ["high", "low"]:
    #         uncert[-1][1].append(lvl)
    # uncert.append(["DMD", []])
    # for lvl in ["high", "low", "med"]:
    #     uncert[-1][1].append(lvl)
    main(uncert)

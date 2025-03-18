# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:42:52 2024

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




# FOLDER_SUBXLS = r"C:\Veda\Veda_models\IFE-NO-2024.08.27_original\IFE-NO-2024.08.27_original\SuppXLS"
# FOLDER_SUBXLS = r"C:\Users\celinep\Documents\GitHub\TIMES_scenario_generation\SuppXLS"

PATH_TIMES = r"C:\Veda\Veda_models\IFE-NO-2024.08.27_simplifiedTS2\IFE-NO-2024.08.27_simplifiedTS2\\"

FOLDER_SUBXLS = f"{PATH_TIMES}SuppXLS"
PATH_UNCERTAINTIES = f"{PATH_TIMES}uncertainties.xlsx"
PATH_GDX = r"C:\Veda\GAMS_WrkTIMES\Stocha"
PATH_CASES = f"{PATH_TIMES}AppData\Cases.json"
PATH_GROUPS = f"{PATH_TIMES}AppData\Groups.json"
PATH_SETTINGS = f"{PATH_TIMES}SysSettings.xlsx"
PATH_SCENARIOS = f"{PATH_TIMES}SuppXLS\\" + "Scen_stocha_uncertainties.xlsx"


class Uncertainty():
    """ Characterize 1 uncertainty based on its attributes.
    Allows to:
        - first create 1 appropriate type of table for the uncertainty
        - Fill it with the values provided as input (data["Values"])
        - if those values are just coeff, retrieve the old values and multiply with the coeffs"""

    def __init__(self, data:dict, old_values):
        """ data of the form:
            {"name":str, Attribute":str, "CommTechName":str,
             "Regions":list, "Periods":list,
             "Values":dict, "ReplaceValue":Bool}"""
        self.data = data
        self.name = data["name"]

        if self.name == "Demand":
            self.typeTable = "~TFM_INS-TS"
            columns = ["Attribute", "CommName", "Region"] + data["Periods"]
            self.df = pd.DataFrame(columns=columns)
            for r in data["Regions"]:
                self.add_values(r)
        elif self.name == "Tech_Char":
            self.typeTable = "~TFM_UPD"
            columns = ["Attribute", "LimType", "Year", "Region", "Pset_PN", "Value"]
            self.df = pd.DataFrame(columns=columns)
            for r in data["Regions"]:
                for y in self.data["Periods"]:
                    self.add_values(r,y)
        
        if not data["ReplaceValue"]:
            for idx, row in self.df.iterrows():
                df_mini_old_values = old_values.loc[(old_values['Attribute'] == row["Attribute"]) & (old_values['CommName'] == row["CommName"]) & (old_values['Region'] == row["Region"])].squeeze(axis=0)
                if self.name == "Demand":
                    for y in data["Periods"]:
                        self.df.at[idx, y] = self.df.at[idx, y] * df_mini_old_values.at[y]
                elif self.name == "Tech_Char":
                    self.df.at[idx, "Value"] = self.df.at[idx, "Value"] * old_values[row["Region"], row["Year"]]

    def add_values(self, r, y=None):
        """ update the Dataframe with a new row """
        row = {k:None for k in self.df.columns}
        row["Attribute"] = self.data["Attribute"]
        if self.name == "Demand":
            row["CommName"] = self.data["CommTechName"]
        elif self.name == "Tech_Char":
            row["Pset_PN"] = self.data["CommTechName"]
        row["Region"] = r
        if y is None:
            for p in self.data["Periods"]:
                row[p] = self.data["Values"][(r,p)]
        else:
            row["Year"] = y
            row["Value"] = self.data["Values"][(r,y)]
        for key in row.keys():
            row[key] = [row[key]]
        self.df = pd.concat([self.df, pd.DataFrame.from_dict(row)], axis=0, ignore_index=True).fillna("")


class Scenario():
    def __init__(self, sheet_names:list, year_second_stage:int):
        self.dfs = []
        for sh in sheet_names:
            data = pd.read_excel(PATH_UNCERTAINTIES, sheet_name=sh, header=None)
            # col_tbd = []
            # row_tbd = []
            # c_year = []
            # for i, el in enumerate(data.iloc[1]):
            #     if type(el) in [float, np.float64, int] and el < year_second_stage:
            #         col_tbd.append(i)
            #     elif el in ["Year", "Period", "Vintage"]:
            #         c_year.append(i)
            
            # for idx, row in data.iterrows():
            #     for col in c_year:
            #         if idx > 1 and data.at[idx, col] < year_second_stage:
            #             row_tbd.append(idx)
            # data.drop(col_tbd,axis=1,inplace=True)
            # data.drop(row_tbd,axis=0,inplace=True)
            self.dfs.append({"sheet_name":sh, "df":data})
        #save it to the 'new_tab' in destfile
        # data.to_excel(destfile, sheet_name='new_tab')

        # self.dfs = []
        # for u in uncertainties:
        #     uncert = Uncertainty(u, old_values)
        #     self.dfs.append({"typeTable":uncert.typeTable, "df":uncert.df})
        #ex: self.dfs = [{"typeTable":"~TFM_INS-TS", "df":pd.DataFrame}, {"typeTable":"~TFM_UPD", "df":pd.DataFrame}]

class Scenarios():
    def __init__(self, uncert:dict, year_second_stage:int):
        self.N = 1
        for el in uncert:
            self.N = self.N * len(el[1])
        self.list_scenarios = []
        self.uncert = uncert
        self.year_second_stage = year_second_stage
        self.id_scenariogroupe = {}
        self.update_SP_file()

    def update_SP_file(self):
        combinations = get_combination(self.uncert)
        sows = {j+1:1/self.N for j in range(self.N)}
        stages = {2:2030}
        
        sett = wf.ExcelSettings(PATH_SCENARIOS, ws_name="stochastic", data_only=False)
        sett.SOW(stages, sows)
        sett.close()
        
        ws_s = wf.ExcelScenarios(PATH_SCENARIOS, "source_scenarios", data_only=False, delete_old=False)
        ws_s.write_scenarios(combinations, new_scenarios=False)
        ws_s.close()
        

    def create_subXLS_scenarios(self):
        # self.get_old_values_uncertainties()
        n = 1
        combinations = get_combination(self.uncert)
        for s in combinations:
            # uncertainties = self.create_uncertainties(self.routine)
            S = Scenario(s, self.year_second_stage)
            self.list_scenarios.append(S)
            self.write_subXLS_scenario(n, S)
            n += 1

    # def create_uncertainties(self, routine): #TODO
    #     if routine == "test":
    #         uncertainties = []
    #         u = {"name":"Demand", "Attribute":"Demand", "CommTechName":"DTCAR",
    #                "Regions":["REG1", "REG2"], "Periods":[y for y in range(2035, 2051)],
    #                "Values":{}, "ReplaceValue":False}
    #         c = np.random.normal(1, 0.05)
    #         for r in u["Regions"]:
    #             for y in u["Periods"]:
    #                 u["Values"][(r,y)] = c
    #         uncertainties.append(u)
            
    #         u = copy.deepcopy(u)
    #         u["CommTechName"] = "DTPUB"
    #         c = np.random.normal(1, 0.05)
    #         for r in u["Regions"]:
    #             for y in u["Periods"]:
    #                 u["Values"][(r,y)] = c
    #         uncertainties.append(u)
            
    #         # u = {"name":"Tech_Char", "Attribute":"INVCOST", "CommTechName":"TCANELC1",
    #         #        "Regions":["REG1", "REG2"], "Periods":[2030],
    #         #        "Values":{}, "ReplaceValue":False}
    #         # c = np.random.normal(1, 0.05)
    #         # for r in u["Regions"]:
    #         #     for y in u["Periods"]:
    #         #         u["Values"][(r,y)] = c
    #         # uncertainties.append(u)

    #         return uncertainties
    
    # def get_old_values_uncertainties(self, name_sheet="uncertainty"): #TODO
    #     self.df_old_values = pd.DataFrame()
    #     files = Path(PATH_TIMES).glob('*.xlsx')
    #     for file in files:
    #         try:
    #             df = pd.read_excel(file, sheet_name="uncertainty").fillna(0)
    #             self.df_old_values = pd.concat([self.df_old_values, df])
    #         except ValueError:
    #             pass


    def write_subXLS_scenario(self, n, scenario):
        wb_name = f"{FOLDER_SUBXLS}\Scen_{n}_uncertainty.xlsx"
        with pd.ExcelWriter(wb_name) as writer:
            for i, u in enumerate(scenario.dfs):
                u["df"].to_excel(writer, sheet_name=u["sheet_name"], index=False,header=False)
            
    def create_scenarios_diag(self):
        with open(PATH_GROUPS, mode="r", encoding="utf-8") as read_file:
            scenarios = json.load(read_file)
        nstart_id = 1000
        new_scenarios = []
        example = None
        for sce in scenarios:
            if sce["GroupName"] == "Base":
                example = sce
                new_scenarios.append(sce)
            elif sce["GroupType"] != "Scenario" or sce["GroupName"] == "Base_uncertainty":
                new_scenarios.append(sce)
        for n in range(1, self.N+1):
            scen_n = self.add_scenario_n(n, n+nstart_id, example)
            new_scenarios.append(scen_n)
        with open(PATH_GROUPS, mode="w", encoding="utf-8") as write_file:
            json.dump(new_scenarios, write_file)

    def add_scenario_n(self, n, n_group_id, example):
        new_scenario = {idx:el for idx, el in example.items()}
        new_scenario["GroupName"] = f"{n}_{n}"
        new_scenario["SavedGroupId"] = f"{n_group_id}"
        new_scenario["Settings"] = new_scenario["Settings"].replace(f"\"Name\": \"{n}_uncertainty\", \"Checked\": false",
                                                                    f"\"Name\": \"{n}_uncertainty\", \"Checked\": true")
        return new_scenario
            
    def move_gdx(self):
        for n in range(1, self.N+1):
            src_file = f"{PATH_GDX}\{n}_{n}\GAMSSAVE\{n}_{n}.gdx"
            dst_file = f"{PATH_TIMES}\AppData\GAMSSAVE\{n}_{n}.gdx"
            shutil.copyfile(src_file, dst_file)

    
    def reload_case(self, diag=False):
        # PATH_CASES
        with open(PATH_CASES, mode="r", encoding="utf-8") as read_file:
            cases = json.load(read_file)
        
        with open(PATH_GROUPS, mode="r", encoding="utf-8") as read_file:
            scenarios = json.load(read_file)
        
        for sce in scenarios:
            if sce["GroupType"] == "Scenario":
                for n in range(1, self.N+1):
                    if sce["GroupName"] == f"{n}_{n}":
                        self.id_scenariogroupe[n] = sce["SavedGroupId"]
                        break
        nstart_id = 1000        
        new_cases = []
        for i, case in enumerate(cases):
            for n in range(1, self.N+1):
                if case["Name"] == f"{n}_{n}" and not diag:
                    new_cases.append(case)
                    break
            if case["Name"] in ("base_stochastic", "Base"):
                new_cases.append(case)
            if case["Name"] == "Base":
                id_region = case["RegionGroupId"]
                id_properties = case["PropertiesGroupId"]
                name_region = case["RegionGroup"]
                name_properties = case["PropertiesGroup"]
                name_periods = case["PeriodsDefinition"]
        n_id = nstart_id
        for n1 in range(1, self.N+1):
            for n2 in range(1, self.N+1):
                if (n1 != n2 and not diag) or (diag and n1 == n2):
                    new_cases.append(self.create_case(n_id, n1, n2, id_region, id_properties, name_region, name_properties, name_periods))
                    # print(case)
                    n_id += 1
        
        with open(PATH_CASES, mode="w", encoding="utf-8") as write_file:
            json.dump(new_cases, write_file)
                    
    def create_case(self, n_id, n1, n2, id_region, id_properties, name_region, name_properties, name_periods):
        
        descr = f"scenario {n2} with fixed variables from {n1}" if n1 != n2 else f"scenario {n2}"
        fix = "True" if n1 != n2 else "False"
        folder = f"{PATH_TIMES}AppData\\GAMSSAVE" if n1 != n2 else ""
        bool_apply = True if n1 != n2 else False
        fixyear = "2030" if n1 != n2 else ""
        filename = f"{n1}_{n1}" if n1 != n2 else ""
        filepath = f"{PATH_TIMES}AppData\\GAMSSAVE\\{n1}_{n1}.gdx" if n1 != n2 else ""
        
        dict_case = {"CaseId":n_id,
                     "CreatedOn":"2025-01-17 14:16",
                     "Description":descr,
                     "EditedOn":"2025-01-17 14:16",
                     "EndingYear":"2055",
                     "FixResultFileName":fix,
                     "FixResultInfo":{"WorkTimesFolderPath":folder,
                                      "GdxElasticDermands":{"GdxSelectedFile":{"FileName":"",
                                                                               "FilePath":"",
                                                                               "IsSelected":False},
                                                            "IsApplied":bool_apply},
                                      "GdxIre":{"GdxSelectedFile":{"FileName":"",
                                                                   "FilePath":"",
                                                                   "IsSelected":False},
                                                "IsApplied":bool_apply,
                                                "RadioSelection":1},
                                      "GdxUseSolution":{"FixYearsUpto":fixyear,
                                                        "GdxSelectedFile":{"FileName":filename,
                                                                           "FilePath":filepath,
                                                                           "IsSelected":bool_apply},
                                                        "IsApplied":bool_apply,
                                                        "RadioSelection":1},
                                      "IsApplyFixResult":bool_apply},
                     "GAMSSourceFolder":"GAMS_SrcTIMES.v4.7.6",
                     "Name":f"{n1}_{n2}",
                     "ParametricGroup":None,
                     "ParametricGroupId":None,
                     "PeriodsDefinition":name_periods,
                     "PropertiesGroup":name_properties,
                     "PropertiesGroupId":id_properties,
                     "RegionGroup":name_region,
                     "RegionGroupId":id_region,
                     "ScenarioGroup":f"{n2}_{n2}",
                     "ScenarioGroupId":self.id_scenariogroupe[n2], # get id scenario group
                     "Solver":"cplex",
                     "SolverOptionFile":"cplex_optGeorge",
                     "UserName":"celinep"}
        return dict_case
            

    # def create_subXLS_fixedVars(self, path_data): #TODO: only constrqint the first stqge variables !!!
    #     self.df_var = pd.read_csv(path_data, sep=";", header=0, )
    #     print(self.df_var )
    #     print(self.df_var.iloc[0,0])
    #     print(self.df_var.iloc[0,0].split(";"))
    #     regions = self.df_var['Region'].unique()
    #     self.df_var = pd.pivot(self.df_var, index=["Scenario", "Attribute", "Commodity", "Process", "Period", "Vintage", "Timeslice"], values="Pv", columns="Region").fillna(0) #TODO
    #     self.df_var.reset_index(inplace=True)
    #     self.df_var = self.df_var.replace("-", "")
    #     self.df_var = self.df_var[self.df_var["Period"] < self.year_second_stage]
    #     for n in range(1, self.N+1):
    #         self.write_subXLS_fixedVar(n, regions)

    # def write_subXLS_fixedVar(self, n, regions):
    #     wb_name = f"{FOLDER_SUBXLS}\Scen_{n}_UCfixedVar.xlsx"
    #     df_n = self.df_var[self.df_var["Scenario"] == f"{n}_{n}"]
    #     # df_n = self.df_var #TODO remettre la ligne du dessus hors test
    #     for att in df_n['Attribute'].unique():
    #         print(wb_name, att)
    #         e = wf.ExcelSUPXLS(wb_name, att)
    #         e.Write_table_UC(df_n[df_n["Attribute"] == att], regions)
    #         e.close()


    def get_costMatrix(self, file, coeff=1): 
        self.df_obj = pd.read_csv(f"{PATH_TIMES}Exported_files\\{file}.csv", sep=";", )
        self.cost_matrix = np.empty(shape=(self.N, self.N), dtype='object')
        print(self.df_obj.head(5))
        
        
        
        for idx, row in self.df_obj.iterrows():
            if "_" in row["Scenario"] and row["Scenario"].split("_")[0] != "base":
                i = int(row["Scenario"].split("_")[0])-1
                j = int(row["Scenario"].split("_")[1])-1
                if i < self.N and j < self.N:
                    self.cost_matrix[i,j] = float(row["Pv"].replace(",", "."))*coeff


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

def write_scenarios_SP(scenarios, N=24, style="CSSC"):
    K = len(scenarios)
    sows = {j+1:len(item)/N for j, (key,item) in enumerate(scenarios.items())}
    stages = {2:2030}
    
    path = f"{FOLDER_SUBXLS}\Scen_stocha_uncertainties_{style}_{K}.xlsx"
    shutil.copyfile(PATH_SCENARIOS, path)
    
    sett = wf.ExcelSettings(path, ws_name="stochastic", data_only=False)
    sett.SOW(stages, sows)
    sett.close()
    
    ws_s = wf.ExcelScenarios(path, "source_scenarios", data_only=False, delete_old=False)
    ws_s.write_scenarios(scenarios.keys(), new_scenarios=True)
    ws_s.close()

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
    
    print(get_combination(uncert))

    S = Scenarios(uncert, year_second_stage=2031)
    # S.create_subXLS_scenarios()
    # input("synchronyize the files in TIMES and reload run manager")
    # S.create_scenarios_diag()
    # input("reload run manager")
    # S.reload_case(diag=True)
    # input("reload run manager and run the N use case of TIMES")
        
    # S.move_gdx()
    # S.reload_case()
    
    # input("reload run manager and run the N²-N use case of TIMES")
    # nfile = input("save export, name file:")
    # nfile = "021125_134329569"
    # # # print()
    # S.get_costMatrix(nfile, coeff=10**-5)       
    # # # #### get representatives and pairing
    # # # # df = pd.DataFrame(columns=[f"{i}" for i in range(1, N+1)])
    # for K in [3]:#,10,15,20]:
    #     for i in range(1,2):
    #         print(S.cost_matrix.astype(int))
    #         scenarios1 = cl.ClusterMIP(K).get_scenarios(S.cost_matrix, new=True)
    #         scenarios1b = cl.ClusterMIP(K).get_scenarios(S.cost_matrix, new=False)
    #         scenarios1c = cl.ClusterMedoidDistance(K).get_scenarios(S.cost_matrix)
    #         scenarios2 = cl.ClusterMedoid(K).get_scenarios(get_combination(uncert))
    #         scenarios3 = cl.ClusterRandom(K).get_scenarios(N)
    #         print(scenarios1)
    #         print(scenarios1b)
    #         print(scenarios1c)
    #         print(scenarios2)
    #         print(scenarios3)
            
            
    #         write_scenarios_SP(scenarios1, N=N, style="CSSC")
    #         write_scenarios_SP(scenarios1b, N=N, style="CSSC_ori")
    #         write_scenarios_SP(scenarios1c, N=N, style="medoid_distance")
    #         write_scenarios_SP(scenarios2, N=N, style="medoid")
    #         write_scenarios_SP(scenarios3, N=N, style=f"random{i}")

        # df = pd.concat([df, pd.DataFrame(res_K, index=[K])])
            
            
    # df.to_csv('output_clustering.csv', index=True) 
    

if __name__ == "__main__":
    
    # Possible uncertainties: WIND, CO2TAX, HYDROGEN, DMD, ELEC, BIOMASS, LEARN, EMISSIONCAP, SOCIETE_CHANGE
    # For each uncertainty, possible levels: high, low, med

    uncert = []
    for u in ["WIND", "CO2TAX", "HYDROGEN", "DMD", "ELEC", "BIOMASS"]:#, "LEARN", "EMISSIONCAP", "SOCIETE_CHANGE"]:
        uncert.append([u, []])
        for lvl in ["high", "low"]:
            uncert[-1][1].append(lvl)
    
    
    
    
    main(uncert)

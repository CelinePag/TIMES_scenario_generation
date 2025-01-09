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

PATH_TIMES = r"C:\Veda\Veda_models\IFE-NO-2024.08.27_simplified\IFE-NO-2024.08.27_simplified\\"

FOLDER_SUBXLS = f"{PATH_TIMES}SuppXLS"
PATH_OPTIVAR = f"{PATH_TIMES}Exported_files\010925_104244412.csv"
PATH_OBJ = f"{PATH_TIMES}Exported_files\\010925_104244412.csv"
PATH_UNCERTAINTIES = f"{PATH_TIMES}uncertainties.xlsx"
PATH_GDX = r"C:\Veda\GAMS_WrkTIMES\UC8"
PATH_CASES = f"{PATH_TIMES}AppData\Cases.json"


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
            col_tbd = []
            row_tbd = []
            c_year = []
            for i, el in enumerate(data.iloc[1]):
                if type(el) in [float, np.float64, int] and el < year_second_stage:
                    col_tbd.append(i)
                elif el in ["Year", "Period", "Vintage"]:
                    c_year.append(i)
            
            for idx, row in data.iterrows():
                for col in c_year:
                    if idx > 1 and data.at[idx, col] < year_second_stage:
                        row_tbd.append(idx)
            data.drop(col_tbd,axis=1,inplace=True)
            data.drop(row_tbd,axis=0,inplace=True)
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

    def create_subXLS_scenarios(self):
        # self.get_old_values_uncertainties()
        n = 1
        all_uncert = [[f"{u[0]}_{ur}" for ur in u[1]] for u in self.uncert]
        combinations = all_uncert[0]
        if len(all_uncert) > 1:
            for i in range(1, len(all_uncert)):
                combinations = list(itertools.product(combinations, all_uncert[i]))
        print(combinations)
        
            
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
            
            
            
            
            # e = wf.ExcelSUPXLS(wb_name, str(i))
            # e.Write_table(u["typeTable"], u["df"])
            # e.close()
            
    def move_gdx(self):
        for n in range(1, self.N+1):
            src_file = f"{PATH_GDX}\{n}_{n}\GAMSSAVE\{n}_{n}.gdx"
            dst_file = f"{PATH_TIMES}\AppData\GAMSSAVE\{n}_{n}.gdx"
            shutil.copyfile(src_file, dst_file)

    
    def reload_case(self):
        # PATH_CASES
        with open(PATH_CASES, mode="r", encoding="utf-8") as read_file:
            cases = json.load(read_file)
        
        print(cases)
        
        nstart_id = 1000

        self.id_scenariogroupe = {}
        
        idx_tba = []
        new_cases = []
        for i, case in enumerate(cases):
            for n in range(1, self.N+1):
                if case["Name"] == f"{n}_{n}":
                    self.id_scenariogroupe[n] = case["ScenarioGroupId"]
                    break
            if case["CaseId"] < nstart_id:
                idx_tba.append(i)
        
        for i in idx_tba:
            new_cases.append(cases[i])
                

            
        n_id = nstart_id
        for n1 in range(1, self.N+1):
            for n2 in range(1, self.N+1):
                if n1 != n2:
                    new_cases.append(self.create_case(n_id, n1, n2))
                    # print(case)
                    n_id += 1
        
        with open(PATH_CASES, mode="w", encoding="utf-8") as write_file:
            json.dump(new_cases, write_file)
                    
    def create_case(self, n_id, n1, n2):
        dict_case = {"CaseId":n_id,
                     "CreatedOn":"2025-01-07 14:16",
                     "Description":f"scenario {n2} with fixed variables from {n1}",
                     "EditedOn":"2025-01-07 14:16",
                     "EndingYear":"2055",
                     "FixResultFileName":"True",
                     "FixResultInfo":{"WorkTimesFolderPath":f"{PATH_TIMES}AppData\\GAMSSAVE",
                                      "GdxElasticDermands":{"GdxSelectedFile":{"FileName":"",
                                                                               "FilePath":"",
                                                                               "IsSelected":False},
                                                            "IsApplied":True},
                                      "GdxIre":{"GdxSelectedFile":{"FileName":"",
                                                                   "FilePath":"",
                                                                   "IsSelected":False},
                                                "IsApplied":True,
                                                "RadioSelection":1},
                                      "GdxUseSolution":{"FixYearsUpto":"2030",
                                                        "GdxSelectedFile":{"FileName":f"{n1}_{n1}",
                                                                           "FilePath":f"{PATH_TIMES}AppData\\GAMSSAVE\\{n1}_{n1}.gdx",
                                                                           "IsSelected":True},
                                                        "IsApplied":True,
                                                        "RadioSelection":1},
                                      "IsApplyFixResult":True},
                     "GAMSSourceFolder":"GAMS_SrcTIMES.v4.7.6",
                     "Name":f"{n1}_{n2}",
                     "ParametricGroup":None,
                     "ParametricGroupId":None,
                     "PeriodsDefinition":"msy09_2055",
                     "PropertiesGroup":"DefaultProperties",
                     "PropertiesGroupId":269,
                     "RegionGroup":"AllRegion",
                     "RegionGroupId":268,
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


    def get_costMatrix(self): 
        self.df_obj = pd.read_csv(PATH_OBJ, sep=";", )
        self.cost_matrix = np.empty(shape=(self.N, self.N), dtype='object')
        for idx, row in self.df_obj.iterrows():
            i = int(row["Scenario"].split("_")[0])-1
            j = int(row["Scenario"].split("_")[1])-1
            self.cost_matrix[i,j] = float(row["Pv"].replace(",", "."))





def main(K):
    uncert = [["CO2prices", ("high", "low", "med")], ["EUROelecPrices", ("high", "low", "med")],]
    N = 1
    for el in uncert:
        N = N * len(el[1])
    print(f"{N} scenarios")
    S = Scenarios(uncert, year_second_stage=2031)
    # S.create_subXLS_scenarios()
    # # S.create_subXLS_fixedVars(PATH_OPTIVAR)
    # S.move_gdx()
    # S.reload_case()
    
    S.get_costMatrix()
    
    # get representatives and pairing

    C = cl.ClusterMIP(K)
    C.construct_model(S.cost_matrix)
    C.solve_model()

if __name__ == "__main__":
    main(K=2)

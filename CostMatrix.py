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

# FOLDER_SUBXLS = r"C:\Veda\Veda_models\IFE-NO-2024.08.27_original\IFE-NO-2024.08.27_original\SuppXLS"
# FOLDER_SUBXLS = r"C:\Users\celinep\Documents\GitHub\TIMES_scenario_generation\SuppXLS"
FOLDER_SUBXLS = r"C:\Veda\Veda_models\test_uncertainties\test_uncertainties\SuppXLS"
path_data = r"C:\Veda\Veda_models\test_uncertainties\test_uncertainties\Exported_files\010225_153623151.csv"
path_obj = r"C:\Veda\Veda_models\test_uncertainties\test_uncertainties\Exported_files\010225_153623151.csv"


class Uncertainty():
    """ Characterize 1 uncertainty based on its attributes.
    Allows to:
        - first create 1 appropriate type of table for the uncertainty
        - Fill it with the values provided as input (data["Values"])
        - if those values are just coeff, retrieve the old values and multiply with the coeffs"""

    def __init__(self, data:dict):
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
        
        
        if False:#not data["ReplaceValue"]:
            if self.name == "Demand":
                old_values = self.get_old_values(self.data["Attribute"], self.data["CommTechName"])
            elif self.name == "Tech_Char":
                old_values = self.get_old_values(self.data["Attribute"], self.data["CommTechName"])
                
            for idx, row in self.df.iterrows():
                if self.name == "Demand":
                    for y in data["Periods"]:
                        self.df.at[idx, y] = self.df.at[idx, y] * old_values[row["Region"], y]
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


    def get_old_values(self, data): #TODO
        if data["Attribute"] == "Demand":
            return {2030:1000, 2050:2000}
        elif data["Attribute"] == "INVCOST":
            return {"AllRegions":200000}
        else:
            raise ValueError



class Scenario():
    def __init__(self, uncertainties):
        self.dfs = []
        for u in uncertainties:
            uncert = Uncertainty(u)
            self.dfs.append({"typeTable":uncert.typeTable, "df":uncert.df})
        #ex: self.dfs = [{"typeTable":"~TFM_INS-TS", "df":pd.DataFrame}, {"typeTable":"~TFM_UPD", "df":pd.DataFrame}]

class Scenarios():
    def __init__(self, N:int, routine):
        self.N = N
        self.list_scenarios = []
        self.routine = routine

    def create_subXLS_scenarios(self):
        for n in range(1, self.N+1):
            uncertainties = self.create_uncertainties(self.routine)
            s = Scenario(uncertainties)
            self.list_scenarios.append(s)
            self.write_subXLS_scenario(n, s)

    def create_uncertainties(self, routine): #TODO
        if routine == "test":
            uncertainties = []
            u = {"name":"Demand", "Attribute":"Demand", "CommTechName":"DTCAR",
                   "Regions":["REG1", "REG2"], "Periods":[y for y in range(2030, 2051)],
                   "Values":{}, "ReplaceValue":False}
            c = np.random.normal(1, 0.05)
            for r in u["Regions"]:
                for y in u["Periods"]:
                    u["Values"][(r,y)] = c
            uncertainties.append(u)
            
            u = copy.deepcopy(u)
            u["CommTechName"] = "DTPUB"
            c = np.random.normal(1, 0.05)
            for r in u["Regions"]:
                for y in u["Periods"]:
                    u["Values"][(r,y)] = c
            uncertainties.append(u)
            
            u = {"name":"Tech_Char", "Attribute":"INVCOST", "CommTechName":"TCANELC1",
                   "Regions":["REG1", "REG2"], "Periods":[2030],
                   "Values":{}, "ReplaceValue":False}
            c = np.random.normal(1, 0.05)
            for r in u["Regions"]:
                for y in u["Periods"]:
                    u["Values"][(r,y)] = c
            uncertainties.append(u)

            return uncertainties


    def write_subXLS_scenario(self, n, scenario):
        wb_name = f"{FOLDER_SUBXLS}\Scen_{n}.xlsx"
        for i, u in enumerate(scenario.dfs):
            e = wf.ExcelSUPXLS(wb_name, str(i))
            e.Write_table(u["typeTable"], u["df"])
            e.close()


    def create_subXLS_fixedVars(self, path_data):
        self.df_var = pd.read_csv(path_data, sep=";", )
        regions = self.df_var['Region'].unique()
        self.df_var = pd.pivot(self.df_var, index=["Scenario", "Attribute", "Commodity", "Process", "Period", "Vintage", "Timeslice"], values="Pv", columns="Region").fillna(0) #TODO
        self.df_var.reset_index(inplace=True)
        for n in range(1, self.N+1):
            self.write_subXLS_fixedVar(n, regions)

    def write_subXLS_fixedVar(self, n, regions):
        wb_name = f"{FOLDER_SUBXLS}\Scen_{n}_UCfixedVar.xlsx"
        # df_n = self.df_var[self.df_var["Scenario"] == f"S_{n}_{n}"]
        df_n = self.df_var #TODO remettre la ligne du dessus hors test
        for att in df_n['Attribute'].unique():
            if att == "VAR_Act":
                print(wb_name, att)
                e = wf.ExcelSUPXLS(wb_name, att)
                e.Write_table_UC(df_n[df_n["Attribute"] == att], regions)
                e.close()


    def get_costMatrix(self, file):
        self.df_obj = pd.read_csv(path_data, sep=";", )
        self.cost_matrix = np.empty(shape=(self.N, self.N), dtype='object')
        
        for idx, row in self.df_obj.iterrows():
            i = row["A"]
            j = row["j"]
            self.cost_matrix[i,j] = row["Pv"]
            



# class Clustering():
#     def __init__(self):




def main(K):
    S = Scenarios(N=10, routine="test")
    S.create_subXLS_scenarios()
    # S.create_subXLS_fixedVars(path_data)
    
    S.get_costMatrix(path_obj)
    
    # get representatives and pairing

    C = cl.ClusterMIP(K)
    C.construct_model(S.cost_matrix)
    C.solve_model()

if __name__ == "__main__":
    main(K=2)

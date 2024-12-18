# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:42:52 2024

@author: celinep
"""

import pandas as pd
import WriteFile as wf

# FOLDER_SUBXLS = r"C:\Veda\Veda_models\IFE-NO-2024.08.27_original\IFE-NO-2024.08.27_original\SuppXLS"
FOLDER_SUBXLS = r"C:\Users\celinep\Documents\GitHub\TIMES_scenario_generation\SuppXLS"


class Uncertainty():
    def __init__(self, name):
        self.name = name
        if name == "Demand":
            self.typeTable = "~TFM_INS-TS"
            columns = ["Attribute", "CommName", "Region", 2030, 2050]
        elif name == "Tech_Char":
            self.typeTable = "~TFM_UPD"
            columns = ["Attribute", "LimType", "Year", "AllRegions", "Pset_PN"]
        self.df = pd.DataFrame(columns=columns)
        
        
    def add_values(self, descr):
        old_values = self.get_old_values(descr)
        data = dict(descr)
        if self.name == "Demand":
            data[2030] = old_values[2030] * descr[2030]
            data[2050] = old_values[2050] * descr[2050]
        elif self.name == "Tech_Char":
            data["AllRegions"] = old_values["AllRegions"] * descr["AllRegions"]
        

        for key in data.keys():
            data[key] = [data[key]]
        
        self.df = pd.concat([self.df, pd.DataFrame.from_dict(data)], axis=0, ignore_index=True).fillna("")
        
 
    def get_old_values(self, data): #TODO
        if data["Attribute"] == "Demand":
            return {2030:1000, 2050:2000}
        elif data["Attribute"] == "INVCOST":
            return {"AllRegions":200000}
    
    # def get_acronym(self):
        

class Scenario():
    def __init__(self, uncertainties):
        self.dfs = []
        for u in uncertainties:
            uncert = Uncertainty(u["name"])
            for values in u["list_values"]:
                uncert.add_values(values)
            self.dfs.append({"typeTable":uncert.typeTable, "df":uncert.df})
        #ex: self.dfs = [{"typeTable":"~TFM_INS-TS", "df":pd.DataFrame}, {"typeTable":"~TFM_UPD", "df":pd.DataFrame}]

class Scenarios():
    def __init__(self, N:int, routine): 
        self.N = N
        self.list_scenarios = []
        self.routine = routine
        
        for n in range(1, N+1):
            uncertainties = self.create_uncertainties(routine)
            s = Scenario(uncertainties)
            self.list_scenarios.append(Scenario(uncertainties))
            self.write_subXLS_scenarios(n, s)
    
    def create_uncertainties(self, routine): #TODO
        if routine == "test":
            uncertainties = [{"name":"Demand",
                              "list_values":[{"Attribute":"Demand",
                                              "CommName":"DEM_TRAIL_Bat",
                                              "Region":"NOR1",
                                              2030:0.9,
                                              2050:0.9},
                                             {"Attribute":"Demand",
                                              "CommName":"DEM_TRAIL_Cat",
                                              "Region":"NOR2",
                                              2030:0.9,
                                              2050:0.9}]},
                             {"name":"Tech_Char",
                              "list_values":[{"Attribute":"INVCOST",
                                              "Year":2030,
                                              "AllRegions":0.9,
                                              "Pset_PN":"TTRUCK-LS-H2"}]}]
            return uncertainties
        
        
    
    
    def write_subXLS_scenarios(self, n, scenario):
        wb_name = f"{FOLDER_SUBXLS}\Scen_{n}.xlsx"
        
        for i, u in enumerate(scenario.dfs):
            e = wf.ExcelSUPXLS(wb_name, str(i))
            e.Write_table(u["typeTable"], u["df"])
            e.close()

        
    
    # def get_optimal_values(self, file):
        
    
    # def write_subXLS_fixedVar(self):
    
    
    # def get_costMatrix(self, file):
        
        

# class Clustering():
#     def __init__(self):
        

def main():
    Scenarios(N=10, routine="test")
    
if __name__ == "__main__":
    main()

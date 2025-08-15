# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:43:50 2024

@author: celinep
"""

from openpyxl import load_workbook, Workbook 


class ExcelTIMES:
    """ Master class for dealing with Excel files """

    def __init__(self, wb_name:str, ws_name:str, data_only:bool, delete_old=True):
        """
        parameters:
        data_only: may erase existing formulas in the workbook
        delete_old: when loading the worksheet, erase existing data
        """

        self.wb_name = wb_name
        try:
            self.wb = load_workbook(wb_name , data_only=data_only)
        except FileNotFoundError:
            self.wb = Workbook()
        try:
            self.ws = self.wb[ws_name]
        except KeyError:
            self.wb.create_sheet(ws_name)
            self.ws = self.wb[ws_name]
        if delete_old:
            self.ws.delete_rows(1, self.ws.max_row)
        self.row_nb = 1
    
    def close(self):
        self.wb.save(self.wb_name) 
        
    def write_header(self, name, header:list):
        self.ws.cell(self.row_nb, 1, value=name)
        for i, value in enumerate(header):
            for j, value2 in enumerate(value):
                self.ws.cell(column=j+1, row=self.row_nb+1+i, value=value2)
        self.row_nb += 1 + len(header)

    
    def sow(self, stages:dict, sows:dict):
        self.write_header("~TFM_INS", [["Attribute", "STAGE", "SOW", "Value"]])
        # Define the decision period
        for stage, year in stages.items():
            self.ws.cell(column=1, row=self.row_nb, value="SW_START")
            self.ws.cell(column=2, row=self.row_nb, value=stage)
            self.ws.cell(column=4, row=self.row_nb, value=year)
            self.row_nb += 1
        # Define the number of SOW per stage (only 2 stages considered here, to be changed otherwise)
        self.ws.cell(column=1, row=self.row_nb, value="SW_SUBS")
        self.ws.cell(column=2, row=self.row_nb, value=1)
        self.ws.cell(column=3, row=self.row_nb, value=1)
        self.ws.cell(column=4, row=self.row_nb, value=len(sows.keys()))
        self.row_nb += 1
        # Define the SOW and their probabilities
        for sow, p in sows.items():
            self.ws.cell(column=1, row=self.row_nb, value="SW_SPROB")
            self.ws.cell(column=2, row=self.row_nb, value=2)
            self.ws.cell(column=3, row=self.row_nb, value=sow)
            self.ws.cell(column=4, row=self.row_nb, value=p)
            self.row_nb += 1
        self.row_nb += 2

    
    def write_scenarios(self, scenarios, new_scenarios=True):
        """
        
        """
        col_level = 1
        col_var = 2
        col_full_scenarios = 3
        for row in range(1, 50):
            if self.ws.cell(row=row, column=col_level).value is None:
                var = self.ws.cell(row=row, column=col_var).value
            else:
                level = self.ws.cell(row=row, column=col_level).value
                txt = "0,99,100"
                for j, s in enumerate(scenarios):
                    if (f"{var}_{level.lower()}" in str(s) and not new_scenarios)\
                        or (str(s) in str(self.ws.cell(row=row, column=col_full_scenarios).value).split(",") and new_scenarios):
                        txt += f",{j+1}"
                self.ws.cell(column=col_var, row=row, value=txt)
                if not new_scenarios:
                    self.ws.cell(column=col_full_scenarios, row=row, value=txt)

    def write_scenarios_par_2s(self, pairs):
        
        # scenarios
        col_L, line_L = 3,5
        col_H, line_H = 3, 40
                
        col_level = 13
        col_var = 14
        col_full_scenarios = 15
        
        
        
        for row in range(1, 50):
            if self.ws.cell(row=row, column=col_level).value is None:
                var = self.ws.cell(row=row, column=col_var).value
            else:
                level = self.ws.cell(row=row, column=col_level).value
                
                for k, (i,j) in pairs.items():
                    txt = "0,99,100"
                    txt = ""
                    first = True

                    
                    # ex: 1, (21,44)
                    print(k,i,j)
                    for n,s in enumerate((i,j)):
                        if str(s) in str(self.ws.cell(row=row, column=col_full_scenarios).value).split(","):
                            if not first:
                                txt += ","
                            txt += f"{n+1}"
                            first = False

                    l_base = line_H  if level == "HIGH" else line_L
                    
                    for c in range(col_L+1, col_L+8):
                        if self.ws.cell(row=l_base, column=c).value == var:
                            self.ws.cell(column=c, row=l_base+k, value=txt)
                            break


    def write_scenarios_par(self, cluster, K, N):
        dict_to_false_k = {1:1,2:2,3:3,4:4,5:5,10:6,15:7,20:8,25:9,35:10,45:11,55:12}

        false_K = dict_to_false_k[K]

        # probabilities
        col_11 = 3
        line_11 = 5
        
        s = 1
        for key,value in cluster.items():
            p = len(value) / N
            self.ws.cell(column=col_11+false_K, row=line_11+s, value=p)
            s += 1
        
        # scenarios
        col_11 = 20
        line_L, line_H = 5, 77
        
        scenarios = cluster.keys()
        
        col_level = 30
        col_var = 31
        col_full_scenarios = 32
        for row in range(1, 50):
            if self.ws.cell(row=row, column=col_level).value is None:
                var = self.ws.cell(row=row, column=col_var).value
            else:
                level = self.ws.cell(row=row, column=col_level).value
                txt = "0,99,100"
                txt = ""
                first = True
                for j, s in enumerate(scenarios):
                    if str(s) in str(self.ws.cell(row=row, column=col_full_scenarios).value).split(","):
                        if not first:
                            txt += ","
                        txt += f"{j+1}"
                        first = False
                
               
                l_base = line_H  if level == "HIGH" else line_L
                
                for c in range(col_11+1, col_11+8):
                    if self.ws.cell(row=l_base, column=c).value == var:
                        self.ws.cell(column=c, row=l_base+K, value=txt)
                        break


# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:43:50 2024

@author: celinep
"""

from openpyxl import load_workbook, Workbook 


DICT_ATT_TO_BND = {"VAR_Act":"ACT_BND", "VAR_Cap":"CAP_BND", "VAR_Ncap":"NCAP_BND",
                   "VAR_Comnet":"COM_BNDNET", "VAR_Comprd":"COM_BNDPRD", "VAR_Cumcom":"COM_CUMBND",
                   "VAR_Cumflo":"FLO_CUM", "VAR_Flo":"FLO_BND", "VAR_Ire":"IRE_BND",
                   "VAR_Cumcst":"REG_CUMCST", "VAR_Sin":"STGIN_BND", "VAR_Sout":"STGOUT_BND"}

class ExcelTIMES:
    def __init__(self, wb_name, ws_name, data_only, delete_old=True):
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
        
    def write_header(self, name, header):
        self.ws.cell(self.row_nb, 1, value=name)
        for i, value in enumerate(header):
            for j, value2 in enumerate(value):
                self.ws.cell(column=j+1, row=self.row_nb+1+i, value=value2)
        self.row_nb += 1 + len(header)



class ExcelScenarios(ExcelTIMES):   
    def write_scenarios(self, scenarios, new_scenarios=True):
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


class ExcelSettings(ExcelTIMES):
    def SOW(self, stages:dict, sows:dict):
        self.write_header("~TFM_INS", [["Attribute", "STAGE", "SOW", "Value"]])
        # Define the decision period
        for stage, year in stages.items():
            self.ws.cell(column=1, row=self.row_nb, value="SW_START")
            self.ws.cell(column=2, row=self.row_nb, value=stage)
            self.ws.cell(column=4, row=self.row_nb, value=year)
            self.row_nb += 1
        # Define the number of SOW per stage (only 2 stage considered here, to be changed otherwise)
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


class ExcelSUPXLS(ExcelTIMES):

    def Write_table(self, name, df):
        self.ws.cell(self.row_nb, 1, value=name)
        self.row_nb += 1
        row_head = self.row_nb
        for i, col in enumerate(df.columns):
            self.ws.cell(self.row_nb, i+1, value=col)
        self.row_nb += 1
        
        for idx, row in df.iterrows():
            for col in range(1, len(df.columns)+1):
                self.ws.cell(self.row_nb, col, value=row[self.ws.cell(row_head, col).value])
            self.row_nb += 1
    
    def Write_table_UC(self, df, regions):
        self.ws.cell(self.row_nb, 1, value="~TFM_INS")
        self.row_nb += 1
        row_head = self.row_nb
        columns = ["Attribute", "LimType", "Pset_PN", "Cset_CN", "Year", "TimeSlice"]
        for i, col in enumerate(columns):
            self.ws.cell(self.row_nb, i+1, value=col)
        for i, r in enumerate(regions):
            self.ws.cell(self.row_nb, len(columns)+1+i, value=r)
        self.row_nb += 1
        for idx, row in df.iterrows():
            self.ws.cell(self.row_nb, 1, value=DICT_ATT_TO_BND[row["Attribute"]])
            self.ws.cell(self.row_nb, 2, value="FX")
            self.ws.cell(self.row_nb, 3, value=row["Process"])
            self.ws.cell(self.row_nb, 4, value=row["Commodity"])
            self.ws.cell(self.row_nb, 5, value=row["Period"])
            self.ws.cell(self.row_nb, 6, value=row["Timeslice"])
            for i, r in enumerate(regions):
                self.ws.cell(self.row_nb, len(columns)+1+i, value=row[r])
            self.row_nb += 1
        
    
# "~TFM_UPD"

# ~TFM_UPD
# ~TFM_INS
# ~UC_T
# ~TFM_INS-TS
# ~TFM_DINS
# ~TFM_DINS-TS: curr=kNOK2021

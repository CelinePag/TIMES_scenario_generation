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
    def __init__(self, wb_name, ws_name, data_only):
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
    def write_K_scenarios(self, new_scenarios):
        # new_scenarios = [1,3,8,20]
        self.list_24 = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                        [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                        [1, 2, 3, 4, 13, 14, 15, 16],
                        [5, 6, 7, 8, 17, 18, 19, 20],
                        [9, 10, 11, 12, 21, 22, 23, 24],
                        [1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22],
                        [3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 23, 24],
                        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
                        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]]
        self.rows = [1,3,6,7,8,11,13,16,18]
        for i, l in enumerate(self.list_24):
            txt = "0,99,100"
            for j, s in enumerate(new_scenarios):
                if s in l:
                    txt += f",{j+1}"
            self.ws.cell(column=1, row=self.rows[i], value=txt)


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

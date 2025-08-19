# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 11:04:38 2025

@author: celinep
"""


# Standard library
import copy
import itertools
import json
import os
import shutil

# Third-party
import numpy as np
import pandas as pd

# Local/project
import WriteFile as wf
import Clustering as cl


# FOLDER_SUBXLS = r"C:\Veda\Veda_models\IFE-NO-2024.08.27_original\IFE-NO-2024.08.27_original\SuppXLS"
# FOLDER_SUBXLS = r"C:\Users\celinep\Documents\GitHub\TIMES_scenario_generation\SuppXLS"


NAME_STUDY = "Stocha"
PATH_TIMES = r"C:\Veda\Veda_models\IFE-NO-2024.08.27_simplified\IFE-NO-2024.08.27_simplified\\"

# PATH_TIMES = r"C:\Veda\Veda_models\IFE-NO-2024.08.27_simplified_test_co2\IFE-NO-2024.08.27_simplified_test_co2\\"
# NAME_STUDY = "Stocha_CO2tax"

FOLDER_SUBXLS = f"{PATH_TIMES}SuppXLS"
PATH_UNCERTAINTIES = f"{PATH_TIMES}uncertainties.xlsx"
PATH_GDX = f"C:\Veda\GAMS_WrkTIMES\{NAME_STUDY}"
PATH_CASES = f"{PATH_TIMES}AppData\Cases.json"
PATH_GROUPS = f"{PATH_TIMES}AppData\Groups.json"
PATH_SETTINGS = f"{PATH_TIMES}SysSettings.xlsx"
PATH_SCENARIOS = f"{PATH_TIMES}SuppXLS\\" + "Scen_stocha_uncertainties.xlsx"
PATH_RESULTS = f"{PATH_TIMES}Exported_files\matrix"
PATH_RESULTS2 = f"{PATH_TIMES}Exported_files\matrix_2S"
PATH_TEMPLATE_STOCH_PAR = f"{PATH_TIMES}SuppXLS\\" + "template_uncertainties_par.xlsx"
PATH_TEMPLATE_STOCH_PAR_2s = f"{PATH_TIMES}SuppXLS\\" + "template_uncertainties_par-2S.xlsx"



def get_num(uncert, combi):
    for u in uncert:
        for l in u[1]:
            print(u[0],l)
            num = []
            for i,c in enumerate(combi):
                if u[0] + "_" + l in c:
                    num.append(i+1)
            print(num)

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

def move_gdx(name_case_par="scenarios_diag_1S", n_start=1, n_fin=64):
    """ 
    Move files from the working directory to the model directory for use as first-stage solutions.
    Files must be made with parametric option and of the form: {name_case_par}~0004

    Parameters:
    name_case_par: Name of the case (must use parametric scenarios)
    n_start: 1st scenario to be included ({name_case_par}~n_start)
    n_fin: last scenario to be included ({name_case_par}~n_fin)
    
     """

    print(f"moving gdx files {n_start} to {n_fin} from {PATH_GDX}\{name_case_par}\{name_case_par}~XXXX\GAMSSAVE\{name_case_par}~XXXX to {PATH_TIMES}\AppData\GAMSSAVE\{name_case_par}~XXXX...", end='')
    for n in range(n_start, n_fin+1):
        name_s = f"{name_case_par}~{str(n).zfill(4)}"
        src_file = f"{PATH_GDX}\{name_case_par}\{name_s}\GAMSSAVE\{name_s}.gdx"
        dst_file = f"{PATH_TIMES}\AppData\GAMSSAVE\{name_s}.gdx"
        shutil.copyfile(src_file, dst_file)
    print("Ok")

def create_groups_scenarios(model_group_name="stochastic_full",
                                list_name_new_groups=[]):
    """ Create new scenario groups based on a given template 
    
    Parameters:
    model_group_name: name of the scenario group to use as template
    list_name_new_groups: list of the new groups based on the template, 
    """

    # Extract from the exisiting groups the one that will be used as template
    with open(PATH_GROUPS, mode="r", encoding="utf-8") as read_file:
        groups = json.load(read_file)
    
    model_group = None
    for g in groups:
        if g["GroupName"] == model_group_name:
            model_group = copy.deepcopy(g)
            break

    # create new groups for each method/type of matrix combination
    nstart_id, n = 10000, 0
    for new_name in list_name_new_groups:
        name_group = f"stochastic_{new_name}" # ex: stochastic_CSSC_new_full_5
        new_group = copy.deepcopy(model_group)
        new_group["SavedGroupId"] = nstart_id + n
        n += 1
        new_group["GroupName"] = new_name               
        new_group["Settings"].replace("{\"Name\": \"stocha_uncertainties\", \"Checked\": true",
                                        "{\"Name\": \"stocha_uncertainties\", \"Checked\": false")
        new_group["Settings"] = new_group["Settings"][:-1] + ", {\"Name\": \"" + f"{name_group}" + "\", \"Checked\": true, \"RowOrder\": 15, \"ShortName\": \"RS\"}]"
        groups.append(copy.deepcopy(new_group))
    
    # Save the new groups
    with open(PATH_GROUPS, mode="w", encoding="utf-8") as write_file:
        json.dump(groups, write_file)

def create_cases_fix_first_stage(year_second_stage:int,
                                  model_case_name="scenarios_diag_1S", 
                                   model_solu_fixed_from="scenarios_diag_1S",
                                     n_list=list(range(1, 64+1)),
                                       list_methods=["CSSC_new"], type_matrix=["full"], list_K=1):
    """ Copy the appropriate case and add fixed first stage solutions.
        Will create as many new parametric cases as there are cases in the parametric case model_solu_fixed_from.
    
    Parameters:
    year_second_stage: Year until when the variables are fixed
    model_case_name: name of the case to will be used as template
    model_solu_fixed_from: name of the case whose first solutions will be used
    n_list: list of the parametric cases from {model_solu_fixed_from} that will be used (when applicable)
    list_methods, type_matrix, list_K: reference to non-parametric case to use as first stage solution (when applicable)

    """
    
    print("Creating cases with fixed first stage solutions ...", end='')

    # Extract from the exisiting cases the one that will be used as template
    with open(PATH_CASES, mode="r", encoding="utf-8") as read_file:
        cases = json.load(read_file)
    
    model_case = None
    for c in cases:
        if c["Name"] == model_case_name:
            model_case = copy.deepcopy(c)
            break
    if model_case is None:
        raise ValueError
    
    iterator = []
    # option 1: we have the diagonal case and want to create the non-diagonal cases
    if model_case_name == model_solu_fixed_from:
        iterator = n_list
        func_name = lambda n:f"{model_case_name}_{n}"
        func_descr = lambda n:f"scenarios with fixed variables from {n}"
        func_filename = lambda n:f"{model_solu_fixed_from}~{str(n).zfill(4)}"

    # option 2: we want to create cases for full_scenario with fixed first stage from various methodology (non-random)
    # ex: model_case_name = "stochastic_full" 
    elif model_solu_fixed_from is None:
        iterator = [(met,mat,K) for met in list_methods for mat in type_matrix for K in list_K]
        func_name = lambda met,mat,K: f"{model_case_name}_fixed_{met}_{mat}_{K}"
        func_descr = lambda met,mat,K:f"full scenarios with fixed variables from {met}_{mat}_{K}"
        func_filename = lambda met,mat,K:f"{met}_{mat}_{K}"

    # option 3: we want to create cases for full_scenario with fixed first stage from random methodology
    # ex: model_case_name = "stochastic_full" and model_solu_fixed_from = "stochastic_random"
    elif model_solu_fixed_from == "stochastic_random":
        iterator = n_list
        func_name = lambda n: f"{model_case_name}_fixed_random_{n}"
        func_descr = lambda n:f"full scenarios with fixed variables from random Par. {n}"
        func_filename = lambda n:f"{model_solu_fixed_from}~{str(n).zfill(4)}"

    # create the new cases by just modifying data from template case related to fixing first stage variables
    n_id = 1000
    for i in iterator:
        new_case = copy.deepcopy(model_case)
        new_case["CaseId"] = n_id
        n_id += 1
        new_case["Name"] = func_name(i)
        new_case["Description"] = func_descr(i)
        new_case["FixResultFileName"] = "True"
        new_case["FixResultInfo"]["WorkTimesFolderPath"] = f"{PATH_TIMES}AppData\\GAMSSAVE"
        new_case["FixResultInfo"]["IsApplyFixResult"] = True
        new_case["FixResultInfo"]["GdxElasticDermands"]["IsApplied"] = True
        new_case["FixResultInfo"]["GdxIre"]["IsApplied"] = True
        new_case["FixResultInfo"]["GdxUseSolution"]["FixYearsUpto"] = f"{year_second_stage}"
        new_case["FixResultInfo"]["GdxUseSolution"]["GdxSelectedFile"]["FileName"] = func_filename(i)
        new_case["FixResultInfo"]["GdxUseSolution"]["GdxSelectedFile"]["FilePath"] = f"{PATH_TIMES}AppData\\GAMSSAVE\\{func_filename(i)}.gdx"
        new_case["FixResultInfo"]["GdxUseSolution"]["GdxSelectedFile"]["IsSelected"] = True
        new_case["FixResultInfo"]["GdxUseSolution"]["IsApplied"] = True
        cases.append(copy.deepcopy(new_case))
    
    # Save the new cases
    with open(PATH_CASES, mode="w", encoding="utf-8") as write_file:
        json.dump(cases, write_file)
    print("ok")

def create_cases_scenarios(name_case_stocha="stochastic_full", 
                           list_name_new_cases=[]):
                           # list_methods=["CSSC_new"], type_matrix=["full"], K=1):
    """ Create new scenario cases based on a given template. 
    /!\ The corresponding groups for the new cases must already exist with the same name
    
    Parameters:
    name_case_stocha: name of the case to use as template
    list_name_new_cases: list of the new cases based on the template, 
    """

    # Extract from the exisiting cases the one that will be used as template
    with open(PATH_CASES, mode="r", encoding="utf-8") as read_file:
        cases = json.load(read_file)
    model_case = None
    for c in cases:
        if c["Name"] == name_case_stocha:
            model_case = copy.deepcopy(c)
            break
    
    with open(PATH_GROUPS, mode="r", encoding="utf-8") as read_file:
        groups = json.load(read_file)
    nstart_id, n = 10000, 0
    for name_case in list_name_new_cases:
        # find the appropriate group to associate with the new case
        for g in groups:
            if g["GroupName"] == name_case:
                id_group = g["SavedGroupId"]
                break
        new_case = copy.deepcopy(model_case)
        new_case["CaseId"] = nstart_id + n
        n += 1
        new_case["Name"] = name_case
        new_case["Description"] = name_case
        new_case["ScenarioGroup"] = name_case
        new_case["ScenarioGroupId"] = id_group
        cases.append(copy.deepcopy(new_case))

    with open(PATH_CASES, mode="w", encoding="utf-8") as write_file:
        json.dump(cases, write_file)

def write_parametric_xls(method="CSSC_new", type_matrix="full", cluster=None, N=1):
    """ Create a new parametric Excel file based on the given template.
    Fill the right worksheet with data such as scenarios chosen for each k and corresponding probabilities"""
    origin = PATH_TEMPLATE_STOCH_PAR
    dest = f"{FOLDER_SUBXLS}\Scen_Par-stocha_uncertainties_{method}_{type_matrix}.xlsx"
    shutil.copyfile(origin, dest)

    # open the new parametric file, keep old data and formulas
    ws_s = wf.ExcelTIMES(dest, "source_scenarios", data_only=False, delete_old=False)

    # for each k of the right method/type matrix clusters, we write in the appropriate cells
    for (m,t,k), value in cluster.items():
        if m == method and t == type_matrix:
            ws_s.write_scenarios_par(value, k, N)
    ws_s.close()

def write_parametric2s_xls(cost_matrix):
    origin = PATH_TEMPLATE_STOCH_PAR_2s
    dest = f"{FOLDER_SUBXLS}\Scen_Par-stocha_uncertainties_2S.xlsx"
    shutil.copyfile(origin, dest)
    pairs = cl.get_pairs_scenarios(cost_matrix)
    ws_s = wf.ExcelTIMES(dest, "source_scenarios", data_only=False, delete_old=False)
    ws_s.write_scenarios_par_2s(pairs)
    ws_s.close()

def write_stochastic_xls():
    origin = PATH_SCENARIOS
    # elif type_file == "stochastic":
    # path = f"{FOLDER_SUBXLS}\Scen_stocha_uncertainties_{method}_{type_matrix}_{self.K}.xlsx"
    # shutil.copyfile(dict_source_file[type_file], path)
    # cluster = self.cluster_full if type_matrix == "full" else self.cluster_sparse
    # sows = {j+1:len(item)/self.N for j, (key,item) in enumerate(cluster[method].items())}
    # stages = {2:f"{self.year_second_stage}"}
    # sett = wf.ExcelSettings(path, ws_name="stochastic", data_only=False)
    # sett.SOW(stages, sows)
    # sett.close()
    # ws_s = wf.ExcelScenarios(path, "source_scenarios", data_only=False, delete_old=False)
    # ws_s.write_scenarios(cluster[method].keys(), new_scenarios=True)
    # ws_s.close()


class ApproximateSP():

    def __init__(self, year_second_stage:int, uncertainties:list, basename="scenarios_diag"):
        """ Suposes that you have already created the following cases:
            - stochastic_full (with corresponding scenario group)
            - stochastic_1S (with corresponding scenario group)
        """
        self.year_second_stage = year_second_stage
        self.uncertainties = uncertainties
        self.N = 1
        for el in uncertainties:
            self.N = self.N * len(el[1])
        print(f"{self.N} scenarios")

        self.cost_matrix = np.empty(shape=(self.N, self.N), dtype='object')
        self.scen_to_be_calculated = list(range(1, self.N+1)) # Compute the full matrix
        self.cluster_sparse = {}

        self.basename_1s = f"{basename}_1S"
        self.basename_2s = f"{basename}_2S"

        self.file_diag_first_stage = r"C:\Veda\Veda_models\IFE-NO-2024.08.27_simplified\IFE-NO-2024.08.27_simplified\Exported_files\041425_134220007.csv"

        self.dict_clusters = {}


    def get_approximate(self, method:str, K:int, sparse:bool):
        self.compute_scenarios_diagonal_matrix()
        self.compute_scenarios_matrix(sparse)
        self.clustering_1s(sparse=sparse, list_methods=[method], list_K=[K])


    def compute_scenarios_diagonal_matrix(self, ):

        input(f"Run the Parametric use case of TIMES with name {self.basename_1s}")
        move_gdx(name_case_par=self.basename_1s, n_start=1, n_fin=self.N)

    
    def get_scenarios_for_sparse(self, corr=0.99, new_file_matrix=True):
        """ Choose which lines of the cost-opportunity matrix to compute based on similarity in first stage solutions of individual subproblems """
        print("Clustering first stage solution of individual problems...", end='')
        self.cluster_sparse = cl.cluster_first_stage_solutions(self.file_diag_first_stage, coeff=corr, new_file_matrix=new_file_matrix)
        print("Ok")
        self.scen_to_be_calculated = [k for k in self.cluster_sparse.keys()]
        print(f"Scenarios to be computed: {self.scen_to_be_calculated}")
        print(f"Total number of determnistic instances reduced by {100*(self.N-len(self.scen_to_be_calculated)-1)/self.N} %")
        

    def compute_scenarios_matrix(self, sparse=False):
        if sparse:
            self.get_scenarios_for_sparse(corr=0.99, new_file_matrix=True)
            create_cases_fix_first_stage(self.year_second_stage,
                                            model_case_name=self.basename_1s,
                                            model_solu_fixed_from=self.basename_1s, 
                                            n_list = self.scen_to_be_calculated)
        input(f"reload run manager tab and run the N Parametric use cases of TIMES + put results files containing obj. fct. value in {PATH_RESULTS} folder")


    def clustering_1s(self, sparse=False, list_methods=["CSSC_new"], list_K=[1,2,3,4,5,10]):
        """ Normal method to cluster the scenarios"""
        if sparse and self.cluster_sparse == {}:
            self.get_scenarios_for_sparse(corr=0.99, new_file_matrix=True)
        self.get_cost_matrix(PATH_RESULTS, coeff=10**-5, sparse=sparse)
        self.get_clusters(list_K, list_methods, sparse)
        type_matrix = "sparse" if sparse else "full"
        for m in list_methods:
            write_parametric_xls(method=m, type_matrix=type_matrix, cluster=self.dict_clusters)


    def clustering_2s(self, sparse=False, list_methods=["CSSC_new"], list_K=[1,2,3,4,5,10]):
        """ Experimental: Cluster the scenarios by first pairing them
          and then working on 2S SP as if individual scenarios
            to create cost opportunity matrix and clustering strategies"""
        if sparse and self.cluster_sparse == {}:
            self.get_scenarios_for_sparse(corr=0.99, new_file_matrix=True)
        self.get_cost_matrix(PATH_RESULTS, coeff=10**-5, sparse=sparse)
        # the matrix needs only diagonals elements, so need only to run the individual scenarios once

        # write_scenarios_xls(type_file="parametric-2S", cost_matrix=cost_matrix)
        pairs = cl.get_pairs_scenarios(self.cost_matrix)
        solos = cl.get_repr_distance(self.cost_matrix, list_K)
        cost_matrix_2S = self.get_cost_matrix(PATH_RESULTS2, N_new=int(self.N/2), name_base="2s", coeff=10**-5, sparse=sparse)
        type_matrix = "sparse" if sparse else "full"

        cluster_2s = self.get_clusters(list_K, list_methods, sparse, matrix=cost_matrix_2S)
        cluster_final = {}
        for m in list_methods:
            for k in list_K:
                cluster_mk = {}
                solo_cluster_repr = None

                # we have at leat 1 pair to create (k >= 2)
                if k >= 2:
                    cluster_2s_mk = cluster_2s[(m, type_matrix, k)]
                    paired_scenarios = {s for key in cluster_2s_mk.keys() for s in pairs[key]}

                    # if odd k: find a solo scenario not in a pair
                    if k % 2 == 1:
                        for s_k in range(1, list_K[-1]):
                            candidate = next((s for s in solos[s_k] if s not in paired_scenarios), None)
                            if candidate is not None:
                                solo_cluster_repr = candidate
                                break

                    # build clusters from pairs
                    for _, (key, value) in enumerate(cluster_2s_mk.items()):
                        # each scenario of the selected pairs will be representative of a cluster
                        # each cluster created by a pair will have the same element of scenarios in it (same probaibilities)
                        s1, s2 = pairs[key]
                        cluster_mk[s1] = [pairs[v][0] for v in value
                            if not (k % 2 == 1 and pairs[v][0] == solo_cluster_repr)]
                        cluster_mk[s2] = [pairs[v][1] for v in value
                            if not (k % 2 == 1 and pairs[v][1] == solo_cluster_repr)]
                
                # We have odd k (one cluster will be made from a solo scenario, not a pair
                if k % 2 == 1:
                    if k == 1:
                        # special case: only one cluster with all scenarios
                        solo_cluster_repr = solos[1][0]
                        solo_cluster_scenarios = list(range(self.N))
                    else:
                        # odd k > 1: cluster only contains its representative
                        solo_cluster_scenarios = [solo_cluster_repr]
                    cluster_mk[solo_cluster_repr] = solo_cluster_scenarios
                cluster_final[(m, type_matrix, k)] = {key:v for key,v in cluster_mk.items()}
        
        
        for m in list_methods:
            write_parametric_xls(method=m, type_matrix=type_matrix, cluster=cluster_final)


    def get_clusters(self, list_K, list_methods, sparse, matrix=None):
        cost_matrix = matrix if matrix else self.cost_matrix
        return_cluster = {}
        dict_class = {"CSSC_new":cl.ClusterMIP, "CSSC_old":cl.ClusterMIP,
                       "medoid_distance":cl.ClusterMedoidDistance, "medoid":cl.ClusterMedoid,
                         "random":cl.ClusterRandom, "spectral":cl.ClusterNormalizedSpectral}
        
        dict_class_kwargs = {"CSSC_new":{"cost_matrix": cost_matrix}, "CSSC_old":{"cost_matrix": cost_matrix},
                                "medoid_distance":{"cost_matrix": cost_matrix}, "medoid":{"combi":get_combination(self.uncertainties)},
                                    "random":{"N": self.N}, "spectral":{"cost_matrix": cost_matrix}}

        dict_fct_kwargs = {"CSSC_new":{"new":True, "feas_tol":(10**(-2))}, "CSSC_old":{"new":False, "feas_tol":(10**(-2))},
                            "medoid_distance":{}, "medoid":{},
                                "random":{}, "spectral":{"e":100}}

        type_matrix = "sparse" if sparse else "full"
        for m in list_methods:
            for k in list_K:
                cluster = dict_class[m](**dict_class_kwargs[m]).compute_scenarios(K=k, **dict_fct_kwargs[m])
                return_cluster[(m, type_matrix, k)] = cluster.scenarios

        if not matrix:
            for key, item in return_cluster:
                self.dict_clusters[key] = item
        else:
            return return_cluster


    def get_cost_matrix(self, folder_matrix, name_base="1s", coeff=1, sparse=False, N_new=None):
        """ Create the cost opportunity matrix from Veda results """

        N = N if (N_new and name_base=="2s") else self.N
        matrix = np.empty(shape=(N_new, N_new), dtype='object') if (N and name_base=="2s") else self.cost_matrix

        print(f"Creating {'sparse' if sparse else 'full'} cost-opportunity matrix of size {N}...", end="")
        for file in os.scandir(folder_matrix):  
            if file.is_file():
                print(f"reading {file}")
                df_obj = pd.read_csv(file, sep=";", )
                for _, row in df_obj.iterrows():
                    if name_base in row["Scenario"] and "-" not in row["Scenario"]:
                        j = int(row["Scenario"].split("~")[-1])-1
                        if row["Scenario"].split("~")[0] == f"scenarios_diag_{name_base}":
                            i = j
                        else:
                            i = int(row["Scenario"].split("~")[0].split("_")[-1])-1
                        if i < N and j < N and i+1 in self.scen_to_be_calculated:
                            matrix[i,j] = float(row["Pv"].replace(",", "."))*coeff
                            if sparse:
                                for i_bis in self.cluster_sparse[i+1]:
                                    matrix[i_bis-1,j] = float(row["Pv"].replace(",", "."))*coeff
        for i in range(N):
            for j in range(N):
                if type(matrix[i,j]) != float:
                    print(f"Error in matrix value for i={i}, j={j}: {matrix[i,j]}")
        print("ok")
        if (N_new and name_base=="2s"):
            return matrix
        else:
            self.cost_matrix = matrix
    



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






def get_full_SP_fixed_1st_stage(list_methods, list_k):
    type_matrix = ["full", "sparse"]
    names = [f"{m}_{t}_{k}" for m in list_methods for t in type_matrix for k in list_k]
    create_groups_scenarios(model_group_stocha="stochastic_full", list_name_new_groups=names, )

    input("close/open run manager again")
    # create_cases_scenarios(name_case_stocha="stochastic_full", list_name_new_cases=names)

    for name in names:
        src_file = f"{PATH_GDX}\{name}\GAMSSAVE\{name}.gdx"
        dst_file = f"{PATH_TIMES}\AppData\GAMSSAVE\{name}.gdx"
        shutil.copyfile(src_file, dst_file)                

    create_cases_fix_first_stage(year_second_stage=2035,
                                 model_case_name="stochastic_full",
                                 model_solu_fixed_from=None,
                                list_methods=list_methods, type_matrix=type_matrix, list_K=list_k)

if __name__ == "__main__":
    method = "CSSC_new"
    K = 5
    year_second_stage = 2035

    uncertainties = []
    for u in ["WIND","CO2TAX", "HYDROGEN", "DMD", "ELEC", "BIOMASS"]: 
        uncertainties.append([u, []])
        for lvl in ["HIGH", "LOW"]:
            uncertainties[-1][1].append(lvl)

    # ex: uncertainties = [["WIND", ["HIGH", "LOW"]], ["HYDROGEN", ["HIGH", "LOW"]]]
    approxi_SP = ApproximateSP(year_second_stage, uncertainties)
    approxi_SP.get_approximate(method, K, sparse=True)

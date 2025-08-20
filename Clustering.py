# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 20:53:16 2025

@author: celinep
"""

# Standard library
import logging
import random
import os

# Third-party libraries
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import savetxt, loadtxt
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.cluster import SpectralClustering

# Pyomo
import pyomo.opt  # we need SolverFactory, SolverStatus, TerminationCondition
from pyomo.environ import (
    ConcreteModel,
    Var,
    Constraint,
    NonNegativeReals,
    Binary,
    minimize,
    Objective,
    maximize
)
from pyomo.util.infeasible import log_infeasible_constraints


def get_distance_matrix(matrix):
    """ Return a distance matrix based on differences of diagonal elements of the original matrix"""
    N = len(matrix)
    distance_matrix = np.empty(shape=(N, N), dtype="object")
    for i in range(N):
        for j in range(N):
            distance_matrix[i,j] = abs(matrix[i,i]-matrix[j,j])
    return distance_matrix


def get_repr_distance(cost_matrix, list_k:list):
    """ Allow to get a basic clustering of scenarios based on kmedoids with the distance matrix.
    Parameters:
        - cost_matrix: array of cost for the scenarios
        - list_K: a list of K values for the size of the reduced SP
    Output:
        - solos: a dict associating for each K a list of representative scenarios
        """

    distance_matrix = get_distance_matrix(cost_matrix)
    solos = {k:None for k in list_k}
    # for each k, we get the k scenarios that would be chosen as medoids when clustering
    for K in list_k:
        initial_medoids = [k for k in range(K)]
        kmedoids_instance = kmedoids(distance_matrix, initial_medoids, data_type='distance_matrix')
        kmedoids_instance.process()
        medoids = kmedoids_instance.get_medoids()
        solos[K] = [m+1 for m in medoids]
    return solos


def get_pairs_scenarios(cost_matrix):
    """ Aim to pair scenarios together such that the total distance within pairs is maximized.
    Parameters:
        - cost_matrix: array of cost for the scenarios
    Output:
        - scenarios: a dict containing the pairs of scenarios"""

    N = len(cost_matrix)
    K = int(N/2)
    scenarios = {}
    distance_matrix = get_distance_matrix(cost_matrix)

    model = ConcreteModel()
    list_n = [n for n in range(1, N+1)]
    list_nn = [(n,j) for n in range(1, N+1) for j in range(n+1, N+1)]
    list_nnk = [(n,j,k) for n in range(1, N+1) for j in range(n+1, N+1) for k in range(1, K+1)]

    # the binary variable x_ijk establishes whether or not the pair i,j is in the pair k
    model.x_nnk = Var(list_nnk, within=Binary)

    # OBJECTIVE ----------------------------------------------------------
    def obj(model):
        return sum(model.x_nnk[(i,j,k)]*distance_matrix[i-1,j-1] for (i,j,k) in list_nnk)
    model.objective_function = Objective(rule=obj, sense=maximize)

    # CONSTRAINTS --------------------------------------------------------
    def c1rule(model, k): # each cluster got 2 scenarios as a pair
        return sum(model.x_nnk[(i,j,k)] for (i,j) in list_nn) == 1
    model.C1 = Constraint(range(1, K+1), rule=c1rule)

    def c2rule(model, i): # each scenario is in exactly one pair
        list_pair_with_i = [(a,b) for (a,b) in list_nn if (a==i or b==i)]
        return sum(model.x_nnk[(a,b,k)] for (a,b) in list_pair_with_i for k in range(1,K+1)) == 1
    model.C2 = Constraint(list_n, rule=c2rule)

    # SOLVE MODEL --------------------------------------------------------
    opt = pyomo.opt.SolverFactory('gurobi') #gurobi
    feas_tol = 10**(-2)
    opt.options['FeasibilityTol'] = feas_tol
    results = opt.solve(model, tee=True,
                            symbolic_solver_labels=False, #goes faster,but turn to true with errors!
                            keepfiles=False)
    solver_stat = results.solver.status
    termination_cond = results.solver.termination_condition

    if (solver_stat == pyomo.opt.SolverStatus.ok) and (
            termination_cond == pyomo.opt.TerminationCondition.optimal):
        print('the solution is feasible and optimal')
    else:
        log_infeasible_constraints(model)
        log_infeasible_constraints(model, log_expression=True, log_variables=True)
        logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.INFO)
        raise Exception('Solver Status: ', solver_stat, 'Termination Condition:', termination_cond )
    print('Solution time: ' + str(results.solver.time))

    for k in range(1, K+1):
        for (i,j) in list_nn:
            if 1-feas_tol <= model.x_nnk[(i,j,k)].value <= 1+feas_tol:
                scenarios[k] = (i,j)
                break
    return scenarios


def cluster_first_stage_solutions(folder_results_veda, file_matrix='matrix_corr_first_stage.csv',
                                   coeff=0.9995, K=None,
                                     new_file_matrix=True, year_second_stage=2035):
    """
        Cluster scenarios together based on similarity of first stage solutions
    """

    # Create a new file for the matrix. Turn to False to save time in debug
    if new_file_matrix:
        for file in os.scandir(folder_results_veda):
            if file.is_file():
                print(f"reading {file}")
                df = pd.read_csv(file, sep=";", )
                break

        scenarios = df["Scenario"].unique()
        matrix_corr = np.empty(shape=(len(scenarios), len(scenarios)), dtype='float')

        # Clean the df to keep only interesting variables
        df.drop(df[(~df.Attribute.str.contains('VAR')) | (df.Attribute.str.endswith('M'))].index, inplace = True)
        df["Pv"] = pd.to_numeric(df["Pv"].str.replace(",", "."))
        df["Period"] = pd.to_numeric(df["Period"], errors="coerce").astype("Int64")
        df = df[(df["Period"] <= year_second_stage) & (df["Period"].notna())]

        # Divide the results in dfs for each different scenarios
        dfs = [df[df["Scenario"] == s][["Attribute", "Commodity", "Process", "Region", "Timeslice", "Period", "Pv"]].rename(columns={"Pv":f"Pv_{str(i).zfill(2)}"}) for i,s in enumerate(scenarios)]

        # compare each scenario with one another to get correlation between them
        for i, df_i in enumerate(dfs):
            matrix_corr[i,i] = 1 # correlation of 1 in diagonal
            for j, df_j in enumerate(dfs[(i+1):], start=i+1):
                df_ij = pd.merge(df_i, df_j, on=["Attribute", "Commodity", "Process", "Region", "Timeslice", "Period"], how='outer')[[f"Pv_{str(i).zfill(2)}", f"Pv_{str(j).zfill(2)}"]]
                corr = df_ij.corr()
                matrix_corr[i,j] = corr.at[f"Pv_{str(i).zfill(2)}", f"Pv_{str(j).zfill(2)}"]
                matrix_corr[j,i] = corr.at[f"Pv_{str(i).zfill(2)}", f"Pv_{str(j).zfill(2)}"]
        savetxt(file_matrix, matrix_corr, delimiter=',')

    matrix_corr = loadtxt(file_matrix, delimiter=',')

    sns.heatmap(matrix_corr)
    plt.savefig('figures/heatmap.png', bbox_inches="tight")
    print(f"Figure {'figures/heatmap.png'} saved.")


    # 1st method: put together scenarios with correlation > coeff, unknown number of clusters
    if coeff is not None:
        print(f"using correlation > {coeff} to cluster scenarios")
        used = []
        cluster = {}
        for i in range(len(matrix_corr)):
            if i not in used:
                cluster[i+1] = [i+1]
                used.append(i)
                for j in range(i+1, len(matrix_corr)):
                    if j not in used and matrix_corr[i,j] > coeff:
                        cluster[i+1].append(j+1)
                        used.append(j)

    # 2nd method: cluster scenarios with spectral clustering to from K cluster
    elif K is not None:
        print(f"using spectral clustering to create {K} clusters")
        clustering = SpectralClustering(n_clusters=K, affinity="precomputed").fit(matrix_corr)
        cluster = {i:[] for i in range(K)}
        for i,l in enumerate(clustering.labels_):
            cluster[l].append(i+1)

    # get best representative for each cluster
    cluster_repr = {}
    for item in cluster.values():
        means = [np.mean([matrix_corr[i-1,j-1] for j in item]) for i in item]
        max_index = np.array(means).argmax()
        cluster_repr[item[max_index]] = item

    return cluster_repr



class ClusterNormalizedSpectral():
    """ Class based on spectral clustering """

    def __init__(self, cost_matrix):
        self.cost_matrix = cost_matrix
        self.N = len(cost_matrix)
        self.scenarios = {} # ex: {1:[1,2,3], 4:[4]} for K=2 and N=4


    def compute_scenarios(self, K, e=None, M=None):
        """ """
        list_n = [n for n in range(1, self.N+1)]
        list_nn = [(n,j) for n in range(1, self.N+1) for j in range(1, self.N+1)]
        distance_matrix = np.empty(shape=(self.N, self.N), dtype="object")
        for i in list_n:
            for j in list_n:
                d = (self.cost_matrix[i-1,j-1] + self.cost_matrix[j-1,i-1]) - (self.cost_matrix[i-1,i-1] + self.cost_matrix[j-1,j-1])
                if d/(self.cost_matrix[i-1,j-1] + self.cost_matrix[j-1,i-1]) < -0.05:
                    print(i, j, d, d/(self.cost_matrix[i-1,j-1] + self.cost_matrix[j-1,i-1]))
                    raise ValueError
                elif d < 0:
                    d = 0
                distance_matrix[i-1,j-1] = d # >= 0

        # compute edge set E:
        E = []
            # either E = {(s,t): d(s,t) < e} for small parsameter e (e-neighbourhood graph)
        if e is not None:
            for i,j in list_nn:
                if  distance_matrix[i-1,j-1] < e:
                    E.append([i,j])
            # or E = {(s,t) in SxS: s~t}  where s~t iif s->t and t->s and s->t if d(s,t) one of the M smallest elements of {d(s,u):u!=s} (M-nearest neighbour grap)
        elif M is not None:
            res = {}
            for i in list_n:
                res[i] = list(np.argsort([distance_matrix[i-1,j-1] for j in list_n])[:M+1])
                res[i] = [a-1 for a in res[i]]
                if i in res[i]:
                    res[i] = [a for a in res[i] if a != i]
                else:
                    res[i] = res[i][:-1]
                for j in range(1, i):
                    if j in res[i] and i in res[j]:
                        E.append([i,j])
                        E.append([j,i])
        else:
            raise ValueError

        # compute affinity matrix : A such that A_si,sj = 1 if (si,sj) in E, 0 otherwise
        A = np.empty(shape=(self.N, self.N), dtype="float")
        for i,j in list_nn:
            A[i-1,j-1] = 1 if [i,j] in E else 0
        W = np.empty(shape=(self.N, self.N), dtype="float")
        for i,j in list_nn:
            W[i-1,j-1] = distance_matrix[i-1,j-1] if [i,j] in E else 0.00001

        clustering = SpectralClustering(n_clusters=K, affinity="precomputed").fit(W)
        clusters = {i:[] for i in range(K)}
        for i,l in enumerate(clustering.labels_):
            clusters[l].append(i+1)

        #get best representative of each cluster
        for _, item in clusters.items():
            means = [np.mean([W[i-1,j-1] for j in item]) for i in item]
            max_index = np.array(means).argmax()
            self.scenarios[item[max_index]] = item


class ClusterMedoid():
    """ """

    def __init__(self, combi):
        self.scenarios = {} # ex: {1:[1,2,3], 4:[4]} for K=2 and N=4
        self.combi = combi

    def compute_scenarios(self, K):
        samples = list(self.combi)

        for k, sample in enumerate(samples):
            for j, _ in enumerate(sample):
                if "high" in sample[k][j]:
                    sample[k][j] = 2
                elif "med" in sample[k][j]:
                    sample[k][j] = 1
                elif "low" in sample[k][j]:
                    sample[k][j] = 0

        # set random initial medoids
        initial_medoids = list(range(K))
        # create instance of K-Medoids algorithm
        kmedoids_instance = kmedoids(samples, initial_medoids)
        # run cluster analysis and obtain results
        kmedoids_instance.process()
        clusters = kmedoids_instance.get_clusters()
        medoids = kmedoids_instance.get_medoids()
        # show allocated clusters
        for m in medoids:
            for c in clusters:
                if m in c:
                    self.scenarios[m+1] = [i+1 for i in c]
                    break


class ClusterRandom():
    """ Class based on random clustering """

    def __init__(self, N):
        self.N = N
        self.scenarios = {} # ex: {1:[1,2,3], 4:[4]} for K=2 and N=4

    def compute_scenarios(self, K):
        """ Randomly selct K representative for the K cluster and randomly associate scenarios"""

        list_repr = random.sample(list(range(1, self.N+1)), K)
        self.scenarios = {r:[] for r in list_repr}
        for n in range(1, self.N+1):
            if n not in list_repr:
                self.scenarios[random.choice(list_repr)].append(n)


class ClusterMedoidDistance():
    """ Class based on Hewitt 2022"""

    def __init__(self, cost_matrix):
        self.cost_matrix = cost_matrix
        self.N = len(cost_matrix)
        self.distance_matrix = np.empty(shape=(self.N, self.N), dtype="object")
        self.scenarios = {} # ex: {1:[1,2,3], 4:[4]} for K=2 and N=4


    def compute_scenarios(self, K):
        """ Use kmedoids based on distance matrix computed from cost opportunity matrix"""
        list_n = list(range(1, self.N+1))
        for i in list_n:
            for j in list_n:
                d = (self.cost_matrix[i-1,j-1] + self.cost_matrix[j-1,i-1]) - (self.cost_matrix[i-1,i-1] + self.cost_matrix[j-1,j-1])
                if d/(self.cost_matrix[i-1,j-1] + self.cost_matrix[j-1,i-1]) < -0.05:
                    print(i, j, d, d/(self.cost_matrix[i-1,j-1] + self.cost_matrix[j-1,i-1]))
                    raise ValueError
                elif d < 0:
                    d = 0
                self.distance_matrix[i-1,j-1] = d # >= 0

        initial_medoids = [k for k in range(K)]
        kmedoids_instance = kmedoids(self.distance_matrix, initial_medoids, data_type='distance_matrix')
        kmedoids_instance.process()
        clusters = kmedoids_instance.get_clusters()
        medoids = kmedoids_instance.get_medoids()
        for m in medoids:
            for c in clusters:
                if m in c:
                    self.scenarios[m+1] = [i+1 for i in c]
                    break


class ClusterMIP():
    """ Class based on the methodology by Katchanyan 2023"""

    def __init__(self, cost_matrix, coeff_matrix=1):
        self.cost_matrix = np.dot(coeff_matrix,  cost_matrix)
        self.N = len(self.cost_matrix)
        self.scenarios = {} # ex: {1:[1,2,3], 4:[4]} for K=2 and N=4
        self.model = ConcreteModel()


    def compute_scenarios(self, K, new=True, feas_tol=10**(-2)):
        """
        parameters:
        cost_matrix: Cost-opportunity matrix as definied in Katchanyan 2023
        new: choose to use original methodology or modified one
        FeasTol: Gurobi parameter
        coeff_matrix: To change magnitude of cost_matrix values for improved performance
        """
        self.construct_model(K, new)
        self.solve_model(feas_tol)


    def construct_model(self, K, new=True):
        """ Build the pyomo model correspoonding to the MIP in Katchanyan 2023 (new=False).
            If new is True, the formulation is improved.
        """

        list_n = list(range(1, self.N+1))
        list_nn = [(n,j) for n in range(1, self.N+1) for j in range(1, self.N+1)]

        diff_matrix_abs = np.empty(shape=(self.N, self.N), dtype="float")
        diff_matrix_nonabs = np.empty(shape=(self.N, self.N), dtype="float")

        for i in list_n:
            for j in list_n:
                diff_matrix_abs[i-1,j-1] = abs(self.cost_matrix[i-1,j-1] - self.cost_matrix[i-1,i-1]) # >= 0
                diff_matrix_nonabs[i-1,j-1] = self.cost_matrix[i-1,j-1] - self.cost_matrix[i-1,i-1]
                # distance between scenario i and scenario j with fixed var from i

        # VARIABLES ------------------------------------------------------------------------------------------
        # t_i represents the clustering discrepancy
        self.model.t_n = Var(list_n, within=NonNegativeReals)

        # The binary variable u_j determines if scenario j is picked as a cluster representative,
        self.model.u_n = Var(list_n, within=Binary)

        # the binary variable x_ij establishes whether or not the scenario i is in the cluster with representative j
        self.model.x_nn = Var(list_nn, within=Binary)

        # CONSTRAINTS ------------------------------------------------------------------------------------
        if new:
            def c1rule(model, j):
                return model.t_n[j] >= sum(model.x_nn[(i,j)]*diff_matrix_abs[j-1,i-1] for i in list_n)
            self.model.C1 = Constraint(list_n, rule=c1rule)
        else:
            def c1rule(model, j):
                return model.t_n[j] >= sum(model.x_nn[(i,j)]*self.cost_matrix[j-1,i-1] for i in list_n) \
                        - sum(model.x_nn[(i,j)]*self.cost_matrix[j-1,j-1] for i in list_n)
            self.model.C1 = Constraint(list_n, rule=c1rule)

            def c2rule(model, j):
                return model.t_n[j] >= -sum(model.x_nn[(i,j)]*self.cost_matrix[j-1,i-1] for i in list_n) \
                        + sum(model.x_nn[(i,j)]*self.cost_matrix[j-1,j-1] for i in list_n)
            self.model.C2 = Constraint(list_n, rule=c2rule)

        def c3rule(model, i, j): # Make sure that scenarios are associated with cluster having a representative
            return model.x_nn[(i,j)] <= model.u_n[j]
        self.model.C3 = Constraint(list_nn, rule=c3rule)

        def c4rule(model, i): # Link variables
            return model.x_nn[(i,i)] == model.u_n[i]
        self.model.C4 = Constraint(list_n, rule=c4rule)

        def c5rule(model, i): # Each scenario is in only 1 cluster
            return sum(model.x_nn[(i,j)] for j in list_n) == 1
        self.model.C5 = Constraint(list_n, rule=c5rule)

        def c6rule(model): #Exactly K scenarios will be representatives
            return sum(model.u_n[j] for j in list_n) == K
        self.model.C6 = Constraint(rule=c6rule)


        # OBJECTIVE -------------------------------------------------------------------------------
        def obj(model):
            return (1/self.N) * sum(model.t_n[i] for i in list_n)
        self.model.objective_function = Objective(rule=obj, sense=minimize)


    def solve_model(self, feas_tol=10**(-2)):
        """ solve and log the model resolution"""

        opt = pyomo.opt.SolverFactory('gurobi') #gurobi
        opt.options['FeasibilityTol'] = feas_tol

        results = opt.solve(self.model, tee=True,
                                        symbolic_solver_labels=False, #goes faster, but turn to true with errors!
                                        keepfiles=False)
                                        #https://pyomo.readthedocs.io/en/stable/working_abstractmodels/pyomo_command.html
        solver_stat = results.solver.status
        termination_cond = results.solver.termination_condition
        if (solver_stat == pyomo.opt.SolverStatus.ok) and (
                termination_cond == pyomo.opt.TerminationCondition.optimal):
            print('the solution is feasible and optimal')
        else:
            log_infeasible_constraints(self.model)
            log_infeasible_constraints(self.model, log_expression=True, log_variables=True)
            logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.INFO)
            raise Exception('Solver Status: ', solver_stat, 'Termination Condition: ', termination_cond )
            #print('Solver Status: '), self.results.solver.status
            #print('Termination Condition: '), self.results.solver.termination_condition

        print('Solution time: ' + str(results.solver.time))

        for j in range(1, self.N+1):
            if 1-feas_tol <= self.model.u_n[j].value <= 1+feas_tol:
                self.scenarios[j] = []
                for i in range(1, self.N+1):
                    if 1-feas_tol <= self.model.x_nn[(i,j)].value <= 1+feas_tol:
                        self.scenarios[j].append(i)

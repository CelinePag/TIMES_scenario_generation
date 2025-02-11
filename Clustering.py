# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 20:53:16 2025

@author: celinep
"""

#Pyomo
import pyomo.opt   # we need SolverFactory,SolverStatus,TerminationCondition
from pyomo.environ import ConcreteModel, Var, Constraint, NonNegativeReals, Binary, Reals, minimize, Objective
from pyomo.util.infeasible import log_infeasible_constraints
from pyclustering.cluster.kmedoids import kmedoids
import random

import logging
import numpy as np


class ClusterScenario():
    def __init__(self, K):
        self.scenarios = {}
        self.K = K
    
    def compute_scenarios(self,*args, **kwargs):
        pass
        
    def get_scenarios(self, *args, **kwargs):
        self.compute_scenarios(*args, **kwargs)
        return self.scenarios



class ClusterMedoid(ClusterScenario):
    def compute_scenarios(self, combi):
        sample = [k for k in combi]
        for k in range(len(sample)):
            for j in range(len(sample[k])):
                if "high" in sample[k][j]:
                    if "DMD" in sample[k][j]:
                        sample[k][j] = 2
                    else:
                        sample[k][j] = 1
                    
                elif "med" in sample[k][j]:
                    sample[k][j] = 0.5
                elif "low" in sample[k][j]:
                    sample[k][j] = 0
                    
        # set random initial medoids
        initial_medoids = [k for k in range(self.K)]
        # create instance of K-Medoids algorithm
        kmedoids_instance = kmedoids(sample, initial_medoids)
        # run cluster analysis and obtain results
        kmedoids_instance.process();
        clusters = kmedoids_instance.get_clusters()
        medoids = kmedoids_instance.get_medoids()
        # show allocated clusters
        for m in medoids:
            for c in clusters:
                if m in c:
                    self.scenarios[m+1] = [i+1 for i in c]
                    break


class ClusterRandom(ClusterScenario):
    def compute_scenarios(self, N):
        list_repr = [random.randint(1, N) for k in range(1, self.K+1)]
        seen = []
        for i,r in enumerate(list_repr):
            if r not in seen:
                seen.append(r)
            else:
                new = random.randint(1, N)
                while new in seen:
                    new = random.randint(1, N)
                seen.append(new)
        list_repr = seen
                        
        for k in list_repr:
            self.scenarios[k] = [k]
            
        for n in range(1, N+1):
            if n not in list_repr:
               self.scenarios[random.choice(list_repr)].append(n)

    
class ClusterMedoidDistance(ClusterScenario): # Hewitt 2022
    def compute_scenarios(self, cost_matrix):
        self.cost_matrix = cost_matrix
        self.N = len(cost_matrix)
        
        list_n = [n for n in range(1, self.N+1)]
        self.distance_matrix = np.empty(shape=(self.N, self.N), dtype="object")
        for i in list_n:
            for j in list_n:
                self.distance_matrix[i-1,j-1] = (self.cost_matrix[i-1,j-1] + self.cost_matrix[j-1,i-1]) - (self.cost_matrix[i-1,i-1] + self.cost_matrix[j-1,j-1]) # >= 0

        initial_medoids = [k for k in range(self.K)]
        kmedoids_instance = kmedoids(self.distance_matrix, initial_medoids, data_type='distance_matrix')
        kmedoids_instance.process()
        clusters = kmedoids_instance.get_clusters()
        medoids = kmedoids_instance.get_medoids()
        # show allocated clusters
        for m in medoids:
            for c in clusters:
                if m in c:
                    self.scenarios[m+1] = [i+1 for i in c]
                    break


# class ClusterGraph(ClusterScenario): # Hewitt 2022
#     def compute_scenarios(self, cost_matrix, new=True, FeasTol=(10**(-2))):
#         self.cost_matrix = cost_matrix
#         self.N = len(cost_matrix)
        
#         list_n = [n for n in range(1, self.N+1)]
#         self.distance_matrix = np.empty(shape=(self.N, self.N), dtype="object")
#         for i in list_n:
#             for j in list_n:
#                 self.distance_matrix[i-1,j-1] = (self.cost_matrix[i-1,j-1] + self.cost_matrix[j-1,i-1]) - (self.cost_matrix[i-1,i-1] + self.cost_matrix[j-1,j-1]) # >= 0

#         initial_medoids = [k for k in range(self.K)]
#         kmedoids_instance = kmedoids(self.distance_matrix, initial_medoids, data_type='distance_matrix')
#         kmedoids_instance.process()
#         clusters = kmedoids_instance.get_clusters()
#         medoids = kmedoids_instance.get_medoids()
#         # show allocated clusters
#         for m in medoids:
#             for c in clusters:
#                 if m in c:
#                     self.scenarios[m+1] = [i+1 for i in c]
#                     break


class ClusterMIP(ClusterScenario): # Katchanyan 2023
    def compute_scenarios(self, cost_matrix, new=True, FeasTol=(10**(-2))):
        self.construct_model(cost_matrix, new)
        self.solve_model(FeasTol)
        
    
    def construct_model(self, cost_matrix, new=True):
        self.model = ConcreteModel()
        self.cost_matrix = cost_matrix
        self.N = len(cost_matrix)
        
        list_n = [n for n in range(1, self.N+1)]
        list_nn = [(n,j) for n in range(1, self.N+1) for j in range(1, self.N+1)]
        
        self.diff_matrix_abs = np.empty(shape=(self.N, self.N), dtype="object")
        self.diff_matrix_nonabs = np.empty(shape=(self.N, self.N), dtype="object")

        for i in list_n:
            for j in list_n:
                self.diff_matrix_abs[i-1,j-1] = abs(self.cost_matrix[i-1,j-1] - self.cost_matrix[i-1,i-1]) # >= 0
                self.diff_matrix_nonabs[i-1,j-1] = self.cost_matrix[i-1,j-1] - self.cost_matrix[i-1,i-1]
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
            def C1rule(model, j):
                return (self.model.t_n[j] >= sum(self.model.x_nn[(i,j)]*self.diff_matrix_abs[j-1,i-1] for i in list_n))
            self.model.C1 = Constraint(list_n, rule=C1rule)
        else:
            def C1rule(model, j):
                return (self.model.t_n[j] >= sum(self.model.x_nn[(i,j)]*self.diff_matrix_nonabs[j-1,i-1] for i in list_n))
            self.model.C1 = Constraint(list_n, rule=C1rule)
            
            def C2rule(model, j):
                return (self.model.t_n[j] >= sum(self.model.x_nn[(i,j)]*(-1)*self.diff_matrix_nonabs[j-1,i-1] for i in list_n))
            self.model.C2 = Constraint(list_n, rule=C2rule)
        
        def C3rule(model, i, j): # Make sure that scenarios are associated with cluster having a representative
            return (self.model.x_nn[(i,j)] <= self.model.u_n[j])
        self.model.C3 = Constraint(list_nn, rule=C3rule)
        
        def C4rule(model, i): # Link variables
            return (self.model.x_nn[(i,i)] == self.model.u_n[i])
        self.model.C4 = Constraint(list_n, rule=C4rule)
        
        def C5rule(model, i): # Each scenario is in only 1 cluster
            return (sum(self.model.x_nn[(i,j)] for j in list_n) == 1)
        self.model.C5 = Constraint(list_n, rule=C5rule)
        
        def C6rule(model, i): #Exactly K scenarios will be representatives
            return (sum(self.model.u_n[j] for j in list_n) == self.K)
        self.model.C6 = Constraint(list_n, rule=C6rule)
        
        
        # OBJECTIVE -------------------------------------------------------------------------------------
        def obj(model):
            return (1/self.N) * sum(self.model.t_n[i] for i in list_n)
        self.model.objective_function = Objective(rule=obj, sense=minimize)
    
    
    def solve_model(self, FeasTol=(10**(-2))):       
        opt = pyomo.opt.SolverFactory('gurobi') #gurobi
        opt.options['FeasibilityTol'] = FeasTol 

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
        df = {}#"K":self.K}
        for k in range(1, self.N+1):
            print(k, self.model.t_n[k].value)
        for j in range(1, self.N+1):
            if 1-FeasTol <= self.model.u_n[j].value <= 1+FeasTol:
                # for i in range(1, self.N+1):
                #     print("diff", i, self.cost_matrix[j-1,i-1]-self.cost_matrix[j-1,j-1])
                #     print("diff abs", i, self.distance_matrix[j-1,i-1])
                df[str(j)] = sum(self.model.x_nn[(i,j)].value for i in range(1, self.N+1))/self.N
                self.scenarios[j] = []
                # print(j, "is representative:", self.model.u_n[j].value==1)
                for i in range(1, self.N+1):
                    if 1-FeasTol <= self.model.x_nn[(i,j)].value <= 1+FeasTol:
                        self.scenarios[j].append(i)
                        # print(i, "is in the same cluster as non-representative:", self.model.x_nn[(i,j)].value==1)
        self.df = df
        
        
if __name__ == "__main__":
    K = 2
    cost_matrix = np.array([[0.9, 1.1, 4.2, 3.9], [1.4,1,4.3,4], [1.8,2,1.1,1], [1.8,2,1.1,1]])
    scenarios1 = ClusterMIP(K).get_scenarios(cost_matrix, new=True)
    scenarios2 = ClusterMIP(K).get_scenarios(cost_matrix, new=False)
    print(scenarios1)
    print(scenarios2)
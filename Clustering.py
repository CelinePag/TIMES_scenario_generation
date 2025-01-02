# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 20:53:16 2025

@author: celinep
"""

#Pyomo
import pyomo.opt   # we need SolverFactory,SolverStatus,TerminationCondition
from pyomo.environ import ConcreteModel, Var, Constraint, NonNegativeReals, Binary, Reals, minimize, Objective
from pyomo.util.infeasible import log_infeasible_constraints
import logging
import numpy as np

class ClusterMIP():
    def __init__(self, K):
        self.model = ConcreteModel()
        self.K = K
        
    def construct_model(self, cost_matrix):
        
        self.N = len(cost_matrix)
        list_n = [n for n in range(1, self.N+1)]
        list_nn = [(n,j) for n in range(1, self.N+1) for j in range(1, self.N+1)]
        
        # VARIABLES
        self.model.t_n = Var(list_n, within=NonNegativeReals)
        # The binary variable u_j determines if scenario j is picked as a cluster representative,
        self.model.u_n = Var(list_n, within=Binary)
        # the binary variable x_ij establishes whether or not the scenario i is in the cluster with representative j
        self.model.x_nn = Var(list_nn, within=Binary)

        # Constraints
        def C1rule(model, j):

            return (self.model.t_n[j] >= sum(self.model.x_nn[(i,j)]*cost_matrix[j-1,i-1] for i in list_n) \
                    - sum(self.model.x_nn[(i,j)]*cost_matrix[j-1,j-1] for i in list_n))
        self.model.C1 = Constraint(list_n, rule=C1rule)
        
        def C2rule(model, j):
            return (self.model.t_n[j] >= sum(self.model.x_nn[(i,j)]*cost_matrix[j-1,j-1] for i in list_n) \
                    - sum(self.model.x_nn[(i,j)]*cost_matrix[j-1,i-1] for i in list_n))
        self.model.C2 = Constraint(list_n, rule=C2rule)
        
        def C3rule(model, i, j):
            return (self.model.x_nn[(i,j)] <= self.model.u_n[j])
        self.model.C3 = Constraint(list_nn, rule=C3rule)
        
        def C4rule(model, i):
            return (self.model.x_nn[(i,i)] == self.model.u_n[i])
        self.model.C4 = Constraint(list_n, rule=C4rule)
        
        def C5rule(model, i):
            return (sum(self.model.x_nn[(i,j)] for j in list_n) == 1)
        self.model.C5 = Constraint(list_n, rule=C5rule)
        
        def C6rule(model, i):
            return (sum(self.model.u_n[j] for j in list_n) == self.K)
        self.model.C6 = Constraint(list_n, rule=C6rule)
        
        
        # Objectif
        def obj(model):
            obj_value = (1/self.N) * sum(self.model.t_n[i] for i in list_n)
            return obj_value
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
        for j in range(1, self.N+1):
            print(j, "is representative:", self.model.u_n[j].value)
            for i in range(1, self.N+1):
                print(i, "is in the same cluster as non-representative:", self.model.x_nn[(i,j)].value)

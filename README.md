# TIMES Scenario Reduction


## Requisites

- Put the files Scen_Par_Uncertainties_1S.xls and template_uncertainties_par.xls in the SuppXLS folder of the TIMES model
- modifiy both files to adjust the uncertainties, the number of scenarios, and the scenarios themself
- Create the folders Matrix and diagonal in the Exported_files folder of the TIMES model
- Synchronize the files in VEDA FE
- Create a scenario group with the files to include in the analysis
- Create a property group that includes ther Stochastic option
- Create a case scenarios_diag_1S whose parameters are:
    - Scenario group as previsouly created
    - parametric group as all_Uncertainties_1S
    - Match the study in settings with the one in the code
    - the property group should the one created before with stochasticity
    - 


## Running the code

- Adjust the global variables related to the path of the model
- Run Main.py and follow instructions, press enter after each actiono to procede
- Once the final .xls is created:
    - synchronize the files in VEDA FE
    - run the newly created file with the chosen parametric options (running all of them means running multiple reduced SP with different K values, might take time)
-  



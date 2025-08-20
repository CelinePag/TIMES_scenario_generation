# TIMES Scenario Reduction

## Requisites

1. Place the following files in the `SuppXLS` folder of the TIMES model:
   - `Scen_Par_Uncertainties_1S.xls`  
   - `template_uncertainties_par.xls`  

2. Modify both files to adjust:  
   - uncertainties  
   - number of scenarios  
   - scenarios themselves  

3. Create the following folders inside the `Exported_files` folder of the TIMES model:  
   - `Matrix`  
   - `diagonal`  

4. Synchronize the files in **VEDA FE**.  

5. Create a **scenario group** with the files to include in the analysis.  

6. Create a **property group** that includes the **Stochastic** option.  

7. Create a case `scenarios_diag_1S` with the following parameters:  
   - Scenario group: previously created group  
   - Parametric group: `all_Uncertainties_1S`  
   - Match the study in settings with the one in the code  
   - Property group: the stochastic property group created earlier  

---

## Running the Code

1. Adjust the global variables related to the path of the model.  

2. Run `Main.py` and follow the instructions:  
   - Press **Enter** after each action to proceed.  

3. Once the final `.xls` is created:  
   - Synchronize the files in **VEDA FE**  
   - Run the newly created file with the chosen parametric options.  
     -  Running all options means multiple reduced SP runs with different **K** values → this can take time.  

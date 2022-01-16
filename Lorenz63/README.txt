This directory takes care of the Lorenz63 part, chapter 2 of the thesis PDF

Running Lorenz_Experiments.py will generate one result figures directory 'figs' containing:
	- RMSE plots for all methods but conditional under the name format "rmse"_\Delta-x_method
	- RMSE plots for conditional method under the name format "entrop_rmse_Delta-x_(alpha_value*100)"
	- Average entropy of likelihood vector at resampling for cumulative method under the name format "entrop_cumul"_Delta-x
	- Average Gini coefficient of likelihood vector at resampling for cumulative method under the name format "gini_cumul"_Delta-x
	- An example run of Lorenz63 on dt=0.01 for 5000 steps and equilibria

There is a 'fixed-SAVED' file which shows the figures as they appear in the report.

The 'preload' directory, contains preloaded multivariate analysis results for the generation of the figures and the printing of correlation tables. The user will be prompted if he wants to use these results to generate the figures, if he refuses they will generate and overwrite the previous results. The generation is very time consuming since it will run close to 200000 total assimilations with the base experiment set:

methods_tested=["Base", "Base_Interpolation", "Cumulative", "Backtrack", "Conditional", "Conditional", "Conditional"]#Methods tested
alpha_entropy_tested=[0,0,0,0,0,1,0.85]#For clarity, the first values dont matter since they are tested with methods that do not take them into account
N_values=[10,18,30,56,100,180,300,560,1000,1800,3000]#N values tested
y_factor=[2,3,4,6,8,10]#y_factors tested
scales=[10,100]#scales of x_assim tested
runs=200#number of runs on each parameter set

Some results have been tiled to reach the 200 runs, which means they weren't necessarily run 200 times (As explained in the report). The code has been modified since the generation of these results and will now generate tables of results on 200 runs.
This directory takes care of the LOVECLIM part, chapter 3 of the thesis PDF

In order to get all the figures used in the paper generated, please run Post_Trait_Mult_Freq.py. Make sure you have the 18GB 'data' directory contact me if you want it antoine.gilliard@hotmail.fr, it contains all the netCDF files used in the evaluation.

Running Post_Trait_Mult_Freq.py will generate three result figures directories:
	- bar_plots: The directory on the correlation bar plots containing
		-Bar plots on the average correlations under the name format "bar_plot"_var_region-grid_timescale_"yr"
		-Bar plots on the correlation on average temperatures under the name format "bar_plot"_var_region-grid_timescale_"yr_zonally"

	- maps: The directory on the maps plots used in this thesis containing
		-Map plots of proxy zone illustrations under the name format var_region-grid
		-Map plots of certain evaluation metrics under under the name format "map"_var_method_timescale_metric

	-time_series: The directory on the time series plots used in this thesis under the name format "time_series"_method_var_timescale_region-grid

All the result figure files are already saved in a directory called "figs-SAVED" to see what should be obtained when running the code

One can also run the file Compute_degeneracies.py, it can be used to look at the degeneracy tendencies of each method (with the exception of particle backtracking)

step 1: 3 files to deal with aq data
	1) aq_update.ipynb: load all aq data from January, 2017 to April, 2018 and merge them into one hdf file
	2) aq_exploration.ipynb: brief overview of aq data (missing value, label correlation)
	3) aq_preprocess.ipynb: fill in missing values

step 2: 5 files to deal with grid weather data
	1) grid_meo_update.ipynb: load all aq data from January, 2017 to May 2nd, 2018 and merge them into one hdf file
	2) grid_meo_preprocess.ipynb: brief overview of grid weather data (missing value)
	3) grid_meo_split.ipynb: split grid weather data to daily data
	4) nearest_grid_meo.ipynb: find the nearest grid weather station for each air quality station
	5) grid_meo_to_aq_meo.ipynb: generate weather data for each air quality station from its nearest grid weather station

step 3:  feature engineering
	1) feature_engineering.ipynb: generate feature
	2) util.py: functions that can be imported

step 4: train and test model
	1) train_model.ipynb: train lightgbm model
	2) prediction.ipynb: make prediction
	
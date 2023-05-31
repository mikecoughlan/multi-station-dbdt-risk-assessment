# multi-station-dbdt-risk-assessment
Repository for performing dB/dt risk assessment at mid-latitude magnetometer stations. Determine the probability that the dB/dt will exceed the 99th percentile threshold 30-60 minutes into the future. Stations analyzed are taken from the SuperMAG repository and include BFE, WNG, LER, ESK, STJ, OTT, NEW, and VIC. Solar wind data was taken from the Advanced Composition Explorer (ACE) L1 monitor. SHAP values are calculated from the resulting Convolutional Neural Network models to help interpret the models as a function of their input parameters. Paper abstract can be found below the file descriptions. Paper will be avalible in this repository after its publication. For all inquiries concerning this repository or this work, or for data used in this research, please contact mike.k.coughlan@gmail.com. [![DOI](https://zenodo.org/badge/536595034.svg)](https://zenodo.org/badge/latestdoi/536595034)

## Publication Abstract
The prediction of large fluctuations in the ground magnetic field (dB/dt) is essential for preventing damage from Geomagnetically Induced Currents. Directly forecasting these fluctuations has proven difficult, but accurately determining the risk of extreme events can allow for the worst of the damage to be prevented. Here we trained Convolutional Neural Network models for eight mid-latitude magnetometers to predict the probability that dB/dt will exceed the 99^{th} percentile threshold 30-60 minutes in the future. Two model frameworks were compared, a model trained using solar wind data from the Advanced Composition Explorer (ACE) satellite, and another model trained on both ACE and SuperMAG ground magnetometer data. The models were compared to examine if the addition of current ground magnetometer data significantly improved the forecasts of $dB/dt$ in the future prediction window. A bootstrapping method was employed using a random split of the training and validation data to provide a measure of uncertainty in model predictions. The models were evaluated on the ground truth data during eight geomagnetic storms and a suite of evaluation metrics are presented. The models were also compared to a persistence model to ensure that the model using both datasets did not over-rely on dB/dt values in making its predictions. Overall, we find that the models using both the solar wind and ground magnetometer data had better metric scores than the solar wind only and persistence models, and was able to capture more spatially localized variations in the dB/dt threshold crossings.

## Non-technical Summary
The interaction between the Sun and the Earth's magnetic field can cause the magnetic field measured on the Earth's surface to change rapidly. These changes can cause damaging currents to run through critical infrastructure such as power lines. Furthermore, the magnetic field can change a large amount in a small distance, meaning that stations that record the magnetic field can have drastically different readings even if they are located close to one another. We can use machine learning algorithms to forecast the risk of these changes in the magnetic field measured on the ground. Using data from stations that measure the magnetic field in addition to data from satellites in between the Sun and the Earth could improve forecast. Here, we created two different models, one using just the data from in between the Sun and the Earth, and one that also used data from the stations that measure the magnetic field on the Earth's surface, to see how they compare in their ability to perform risk assessments.

## File/directory descriptions

### [Conference Posters](/conference%20posters/)
Posters created for this work, presented at various conferences. Posters are in pdf format.

### [env_and_software_requirements](/env_and_software_requirements/)
Requirements.txt and env.yml files that can be used for created a virtual environment to recreate this work.

### [paper submission figures](/paper%20submission%20figures/)
Figures included in either the main text of the publication of this work, or in the supplimental information file. Figures are in jpg format.

### [CNN_cross_val.py](/CNN_cross_val.py)
Performs a cross validation for the CNN models for hyperparameter turning. Displays the value for several metric scores for anaylsis.

### [analyzing_results.py](/analyzing_results.py)
Calculates metric scores from predicted data and defines the median, mean and 95th percentiles for plotting. Saves results to a dictonary for plotting in the plotting.py script. Plots some supplimental information such as correlations betwwen model input values and model outputs.

### [calculating_shap_values.py](/calculating_shap_values.py)
Calculates the SHAP values for the solar wind and combined models for each station. 10 randomly chosen of the 100 split models have their resulting 2D SHAP value arrays averaged. The sum of the values across the time dimension producing only one value per input parameter. The total values are then normalized to a percentage of total SHAP values for that input array, giving a percent contribution for each parameter. Plots the results in a stackplot for the combined and the solar wind models. Also calculates the rolling average of these percent contributions to smooth the plots. This was not used in analysis.

### [combining_and_processing_data.py](/combining_and_processing_data.py)
Combines the preprocessed magnetometer and solar wind data, segments out the training and testing data and creates the arrays that will be used for input to the models. Uses dictonaries to store the np.arrays and the relevent data.

### [config.json](/config.json)
File containing dictonary of values used across multiple files.

### [data_distributions.py](/data_distributions.py)
Calculates and plots the data distributions for all available data, the data used for training, and the testing data.

### [examining_training_data.py](/examining_training_data.py)
Calculates the amount of data available for different limits on linear interpolation over missing data. Does this for each station and plots the results.

### [model_config.json](/model_config.json)
File containing dictonary of values used in multiple places during the modeling process. Contains variables specific to the CNN (hyperparameters, number of models, etc.)

### [modeling.py](/modeling.py)
Script for defining the neural network, fitting the model, and making a prediction on the testing data. Fits models for the 100 unique train-val splits and makes 100 corresponding predictions for each testing storm. Saves the results to a csv/feather file used for analyzing results in the analyzing_results.py script. Uses an argparser for choosing the magnetometer station the models will be created for.

### [plotting.py](/plotting.py)
Takes the results dictonaries and files and creates plots using matplotlib to display the results. Saves the plots to the plots directory.

### [preparing_SW_data.py](/preparing_SW_data.py)
File for preparing the raw solar wind data from both ACE and OMNI. Takes the source files, up or down samples the ACE data as necessary to the 1-minute resoultion. Changes missing data format from eg. 999.999 to np.nan. Interpolates up to 15 minutes of missing data. Saves the data as an external file for use in later scripts. SCRIPT ADAPTED FROM SIMILAR SCRIPT WRITTEN BY VICTOR A. PINTO

### [preparing_mag_data.py](/preparing_mag_data.py)
File for preparing the ground magnetometer data from the SuperMAG stations. Standardizes column names and interpolates up to 15 minutes of missing data. Saves the data as an external file for use in later scripts. SCRIPT ADAPTED FROM SIMILAR SCRIPT WRITTEN BY VICTOR A. PINTO

### [shap_feature_contributions.py](/shap_feature_contributions.py)
Plots the SHAP percentage contributions against the input parameter values. Does so by loading the pre-calculated SHAP values and normalizing the values to a percentage. Does this for all of the time history, so each value is a single input not the sum over the time history. Plots the results for all of the 10 randomly choosen models for all 8 stations and all 8 test storms. This can cause outliers to appear more significant than they are. Plots the results in a  2D histogram for evaluation.

### [shap_values.ipynb](/shap_values.ipynb)
Notebook for exploring the use of SHAP values for interpreting the model outputs. This is an experimental notebook used for learning about the SHAP values, how they can best be displayed, and to probe their robustness. This notebook is not well documented and will be continully updated as I continue to experiment with these values, the SHAP package, and overall model interpretibility.

### [space_weather_2023.pdf](/space_weather_2023.pdf)
PDF of publication in the Journal of Space Weather.

### [storm_list.csv](/stormList.csv)
List of the SYM-H minimums for the storms used in training. Lead and recovery time are added to the SYM-H minimum and then extracted from the larger dataset to be used in training. File is used in the [combining_and_processing_data.py](/combining_and_processing_data.py) script.


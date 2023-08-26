
# Siesmic_traces preprocessing and feature engineering

This repository contains code, to preprocess the seismic input traces and forming new features.


## Files

* `preprocess`: contain subfolders such as individual_trace.py, filtered_merged.py and merged_filter.py. 
  
* `individual_trace.py`(subfolder): This notebook contain code for how the individual traces are preprocessed using different filters.

* `filtered_merged.py`(subfolder): This notebook contain code for how all the individual traces in a cdp are filtered and then merged .
  
* `merged_filter.py`(subfolder):This notebook contain code for how the individual traces in a cdp are first merged and then filtered .
  
* `FE.py`: This notebook contain code how the individual traces in cdp are filtered and then merged and different features are formed and correlations is finded.

* `store_data_folder.py`: This notebook contain code how the preprocessed and new features are stored in each cdp separate folder.

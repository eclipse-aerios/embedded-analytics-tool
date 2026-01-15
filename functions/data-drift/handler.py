# Required imports
import importlib.util
import sys
import json

# Testing imports
from numpy.random import seed
from numpy.random import randn
from numpy.random import lognormal

# KS_TEST imports
from scipy.stats import ks_2samp

#PSI imports
import numpy as np
import pandas as pd

# Load Metric Reporter
spec = importlib.util.spec_from_file_location("metric_reporter", "/home/app/function/metric_reporter.py")
metricReporterModule = importlib.util.module_from_spec(spec)
sys.modules["metric_reporter"] = metricReporterModule
spec.loader.exec_module(metricReporterModule)
reporter = metricReporterModule.MetricReporter()

# Function Handler
def handle(req):
    req = json.loads(req)
    #{"testFunction" : "true"}
    if 'testFunction' in req and req['testFunction'] == 'true':  
        #set seed (e.g. make this example reproducible)
        seed(0)
        
        #generate two datasets
        data1 = randn(100)
        data2 = lognormal(3, 1, 100)
        
        #perform Kolmogorov-Smirnov test with test data
        result = ks_2samp(data1, data2)

        #perform PSI with test data
        dev_data = pd.DataFrame({'score': [15, 20, 25, 30, 20, 15, 10, 5, 30, 10]})
        val_data = pd.DataFrame({'score': [15, 20, 24, 25, 20, 15, 10, 5, 30, 10]})
        psi_value = population_stability_index(dev_data, val_data, 'score')

        #return results
        return {
            "KS_Test_result" : str(result),
            "PSI Value" : str(psi_value)
            }

    else:
        #Custom Code Written Here.

        # Example of pushing metric to Prometheus
        pushResult = reporter.report_metric(
            value=1.23,
            metric_id='some_id',
            model_name='some_model_name',
            description='Some description of the metric.'
            )
        return {
            "metricReported" : pushResult
            }
    
def population_stability_index(dev_data, val_data,col_name, num_bins=10):
    #The next line calculates the minimum and maximum scores in the development dataset and creates num_bins + 1 equally spaced bin edges.
    bins = np.linspace(dev_data[col_name].min(), dev_data[col_name].max(), num_bins + 1)
    
    #Next, we assign each score in both datasets to the corresponding bin. “pd.cut” is a pandas function that categorizes a continuous variable into discrete bins based on specified edges.
    dev_data['bin'] = pd.cut(dev_data[col_name], bins=bins, include_lowest=True)
    val_data['bin'] = pd.cut(val_data[col_name], bins=bins, include_lowest=True)
    
    #We group the data by bins, and the number of scores in each bin is counted for both datasets. The result is stored in two separate data frames, dev_group and val_group.
    dev_group = dev_data.groupby('bin')[col_name].count().reset_index(name='dev_count')
    val_group = val_data.groupby('bin')[col_name].count().reset_index(name='val_count')
    
    #The next line merges the dev_group and val_group data frames based on the ‘bin’ column. The ‘left’ join ensures that all bins from the development dataset are included in the final merged data frame, even if there are no scores in the validation dataset for those bins.
    merged_counts = dev_group.merge(val_group, on='bin', how='left')
    
    #Next, we calculate the PSI value for each bin. First, a small constant is added to both the development and validation percentages to avoid division by zero. Then the percentage of scores in each bin is calculated for both datasets. Finally, the PSI is calculated for each bin using the formula (val_pct – dev_pct) * log(val_pct / dev_pct).
    small_constant = 1e-10
    merged_counts['dev_pct'] = (merged_counts['dev_count'] / len(dev_data)) + small_constant
    merged_counts['val_pct'] = (merged_counts['val_count'] / len(val_data)) + small_constant
    merged_counts['psi'] = (merged_counts['val_pct'] - merged_counts['dev_pct']) * np.log(merged_counts['val_pct'] / merged_counts['dev_pct'])
    
    #Lastly, we return the sum value of all the bins.
    return merged_counts['psi'].sum()
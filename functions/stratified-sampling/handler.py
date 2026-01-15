# Required imports
import importlib.util
import sys
import json

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
        # Create a dictionary of students 
        students = {
            'Name': ['Lisa', 'Kate', 'Ben', 'Kim', 'Josh', 'Alex', 'Evan', 'Greg', 'Sam', 'Ella'], 
            'ID': ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010'],
            'Grade': ['A', 'A', 'C', 'B', 'B', 'B', 'C', 'A', 'A', 'A'],
            'Category': [2, 3, 1, 3, 2, 3, 3, 1, 2, 1]
            } 
         
            # Create dataframe from students dictionary 
        df = pd.DataFrame(students) 
        
        #Disproportionate Sampling: Using pandas groupby, separate the students into groups based on their grade i.e A, B, C and randomly sample 2 students from each grade group using the sample function
        disproSample = df.groupby('Grade', group_keys=False).apply(lambda x: x.sample(2))
        
        #Proportionate Sampling: Using pandas groupby, separate the students into groups based on their grade i.e A, B, C, and random sample from each group based on population proportion. The total sample size is 60%(0.6) of the population
        proSample = df.groupby('Grade', group_keys=False).apply(lambda x: x.sample(frac=0.6))
        
        return {
            "disproSample": disproSample.to_string(),
            "proSample": proSample.to_string()
            }
    else:
        # Custom Code Written Here.

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

# Required imports
import importlib.util
import sys
import json
import os
import gdown
import pandas as pd
from ydata_profiling import ProfileReport
# ADD new libraries when we decide about mintaka

# Load Metric Reporter
spec = importlib.util.spec_from_file_location("metric_reporter", "/home/app/function/metric_reporter.py")
metricReporterModule = importlib.util.module_from_spec(spec)
sys.modules["metric_reporter"] = metricReporterModule
spec.loader.exec_module(metricReporterModule)
reporter = metricReporterModule.MetricReporter()

def getHTML():
    try:
        with open('/home/app/Analysis.html') as analysis_file:
            ret = analysis_file.read()
    except IOError as e:
        ret = "<!DOCTYPE html><html><body><p>(function)EDA Report has not been generated yet.</p><p>Please invoke function.</p></body></html>"
    return "Analysis.html"

# Function Handler
def handle(req):
    print("Handler has triggered.")
    try:
        req = json.loads(req)
    except:
        print("JSON load failed")
        print(req)

    # Perform EDA on csv or excel file from google drive - START
    if 'gdrive' in req:
        data_file_url = req['gdrive']['url']
        
        # Download file from url
        data_file = gdown.download(data_file_url, fuzzy=True)

        # Check file extension
        file_extension = os.path.splitext(data_file)[1]
    
        if file_extension == '.csv':
            # Open CSV file
            try:
                df = pd.read_csv(data_file)
            except Exception as e:
                print(f"Error opening CSV file: {e}")
                return None
        elif file_extension == '.xlsx' or file_extension == '.xls':
            # Open XLSX/XLS file
            try:
                df = pd.read_excel(data_file)
                print("XLSX file opened successfully!")
            except Exception as e:
                print(f"Error opening XLSX/XLS file: {e}")
                return None
        else:
            print("Unsupported file format!")
            return None
            
        report = ProfileReport(df, title="EDA")
        report.to_file("Analysis.html")
        ret = { "result" : "Analysis.html created from gdrive"}
        return ret
    # Perform EDA on csv or excel file from google drive - END

    # Perform EDA on measurements stored in a db using mintaka - START
    elif 'data_fabric' in req:
        # waiting for input from other partners
        # mintaka, available measurements, etc
        print('Under Development')
        ret = { "result" : "data fabric under development"}
        return ret

    elif 'test' in req:
        # use test data
        with open('/home/app/function/gps_data.csv') as data_file:
            df = pd.read_csv(data_file)
            report = ProfileReport(df, title="EDA")
            report.to_file("/home/app/templates/Analysis.html")
        print('Test Complete')
        ret = { "result" : "Analysis.html created from test data"}
        return ret
    # Perform EDA on measurements stored in a db using mintaka - END
    
    else:
        print('No valid keys in HTTP request Body')
        ret = { "result" : "EDA not generated"}
        return ret
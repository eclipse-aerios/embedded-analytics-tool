# Required imports
import importlib.util
import sys
import json

from sklearn.cluster import KMeans

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
        x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
        y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
        data = list(zip(x, y))
        print(data)
        
        inertias = []
        for i in range(1,11):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
            print(f"Clusters: {i}, Inertias: {kmeans.inertia_}")
        
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(data)
        print(kmeans.labels_)

        #return results
        return {
            "Data" : str(data),
            "KMeans_Label" : str(kmeans.labels_)
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
    
def population_stability_index(dev_data, val_data,col_name, num_bins=10):
    return merged_counts['psi'].sum()
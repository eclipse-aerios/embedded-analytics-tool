import importlib.util
import sys
spec = importlib.util.spec_from_file_location("metric_reporter", "/home/app/function/metric_reporter.py")
metricReporterModule = importlib.util.module_from_spec(spec)
sys.modules["metric_reporter"] = metricReporterModule
spec.loader.exec_module(metricReporterModule)
reporter = metricReporterModule.MetricReporter()

def handle(req):
    # Send some example metrics to the pushgateway
    print("handle triggered")
    pushResult = reporter.report_metric(
        value=1.23,
        metric_id='some_id',
        model_name='some_model_name',
        description='Some description of the metric.'
    )
    
    return {
        "metricReported": pushResult
    }

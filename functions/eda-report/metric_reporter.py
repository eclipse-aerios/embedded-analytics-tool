""" Contains class that enables pushing metrics to the Prometheus PushGateway """
from prometheus_client import push_to_gateway, Gauge, CollectorRegistry

class MetricReporter:
    """ MetricReporter is a class to push metrics (ML and other) to the Prometheus PushGateway. """

    def __init__(self):
        self.metrics = {}
        self.registry = CollectorRegistry()

    def report_metric(self, value, metric_id='unidentified_metric', model_name='unidentified_model', description=''):
        """ Pushes a metric to the PushGateway """
        if metric_id not in self.metrics:
            self.metrics[metric_id] = Gauge(metric_id, description, registry=self.registry)
        self.metrics[metric_id].set(value)
        push_to_gateway('eat-push-prometheus-pushgateway.embedded-analytics-tool:9091', job=model_name, registry=self.registry)
        return True
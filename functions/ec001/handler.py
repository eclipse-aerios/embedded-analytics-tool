import importlib.util
import sys
import os
import requests
import json
spec = importlib.util.spec_from_file_location("metric_reporter", "/home/app/function/metric_reporter.py")
metricReporterModule = importlib.util.module_from_spec(spec)
sys.modules["metric_reporter"] = metricReporterModule
spec.loader.exec_module(metricReporterModule)
reporter = metricReporterModule.MetricReporter()

def notif(token,ie):
    url = "https://portal.aerios-project.eu/portal-backend/notifications"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer "+token
    }
    payload = {
        "category": "EAT-Self-orchestrator",
        "description": "Message from EAT: Notification of type EC001 coming from IE: "+ie,
        "alertSource": "CloudFerro",
        "severity": "critical"
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    print(response.status_code)
    print(response.text)


def get_token():
    keycloak_payload = {
        'grant_type': 'client_credentials',
        'client_id': 'ContextBroker',
        'client_secret': '<secret>'
    }

    krakend_headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    response = requests.post(url='https://keycloak.aerios-project.eu/auth/realms/<realm>/protocol/openid-connect/token', data=keycloak_payload, headers=krakend_headers, timeout=3)
    return json.loads(response.text)['access_token']

def handle(req):

    token = get_token()
    data = json.loads(req)
    ie=data.get("infrastructureElementId","default_value")
    notif(token,ie)
       
    return req




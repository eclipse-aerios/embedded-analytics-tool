import requests
import json
import os

#Start Grafana Dashboard
reqUrl = 'http://eat-grafana.embedded-analytics-tool.svc.cluster.local:80/api/dashboards/db'

defaultReqBody = {
  "dashboard": {
    "id": None,
    "uid": None,
    "title": "Production Overview",
    "tags": [ "templated" ],
    "timezone": "browser",
    "schemaVersion": 16,
    "refresh": "25s"
  },
  "message": "Made changes to xyz",
  "overwrite": False
}

#To import from json file
with open('/home/app/function/dashboard.json') as dashboard_file:
    dashboard = json.load(dashboard_file)
reqBody = {
  "dashboard": dashboard,
  "message": "Updating Dashboard from File",
  "overwrite": False
}

reqHeaders = {'Accept': 'application/json', 'Content-Type': 'application/json'}

#Post Dashboard
r = requests.post(reqUrl, json=reqBody, headers=reqHeaders, auth=(os.environ['GRAFANA_USER'], os.environ['GRAFANA_PASS']))
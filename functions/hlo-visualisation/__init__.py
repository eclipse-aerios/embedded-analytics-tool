import requests
import json
import os

#Start Grafana DataSource
reqUrl = 'http://eat-grafana.embedded-analytics-tool.cluster.local:80/api/datasources'

defaultDataSourceReqBody = {
  "name": "test_datasource",
  "type": "graphite",
  "url": "http://mydatasource.com",
  "access": "proxy",
  "basicAuth": True,
  "basicAuthUser": "basicuser",
  "secureJsonData": {
    "basicAuthPassword": "basicpassword"
  }
}

with open('/home/app/function/embedded_files/datasource.json') as datasource_file:
    datasource = json.load(datasource_file)
    # Update the specified field value
    if "basicAuthPassword" in datasource["jsonData"]:
      datasource["jsonData"]["basicAuthPassword"] = os.environ['OPENFAAS_PASS']
reqBody = datasource

reqHeaders = {'Accept': 'application/json', 'Content-Type': 'application/json'}

r = requests.post(reqUrl, json=reqBody, headers=reqHeaders, auth=(os.environ['GRAFANA_USER'], os.environ['GRAFANA_PASS']))

#Start Grafana Dashboard
reqUrl = 'http://eat-grafana.embedded-analytics-tool.cluster.local:80/api/dashboards/db'

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
with open('/home/app/function/embedded_files/dashboard.json') as dashboard_file:
    dashboard = json.load(dashboard_file)
reqBody = {
  "dashboard": dashboard,
  "message": "Updating Dashboard from File",
  "overwrite": False
}

reqHeaders = {'Accept': 'application/json', 'Content-Type': 'application/json'}

r = requests.post(reqUrl, json=reqBody, headers=reqHeaders, auth=(os.environ['GRAFANA_USER'], os.environ['GRAFANA_PASS']))

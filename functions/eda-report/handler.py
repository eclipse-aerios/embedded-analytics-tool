# Required imports
import importlib.util
import sys
import json
import os
import gdown
import pandas as pd
from ydata_profiling import ProfileReport


# Set DEV to False when building production container images
DEV = True
if DEV:
    # API_URL = 'https://test-domain.aerios-project.eu'
    # API_PORT = '443'
    # URL_VERSION = ''
    # Orion-LD configuration
    API_URL = 'https://test-domain.aerios-project.eu'
    API_PORT = '443'
    URL_VERSION = 'entities'
    K8S_SHIM_URL = 'http://localhost'
    K8S_SHIM_PORT = '8000'
    TOKEN = 'TOKEN'
else:
    API_URL = "http://orion-ld-broker"
    API_PORT = "1026"
    URL_VERSION = 'ngsi-ld/v1/entities'
    K8S_SHIM_URL = "http://aerios-k8s-shim-service"
    K8S_SHIM_PORT = "8085"
ORION_ENTITY_ID = 'urn:ngsi-ld:vehicle:5g-car:1'
TOKEN_URL = f"{K8S_SHIM_URL}:{K8S_SHIM_PORT}/token/cb"

import logging

def get_app_logger():
    """
    Returns a logger instance that only logs to the console.
    """
    logger = logging.getLogger("app_logger")

    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S'
        )

        # Console (Stream) handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)

    return logger


# ADD new libraries when we decide about mintaka
import requests

logger = get_app_logger()


def get_m2m_cb_token():
    """
    Get M2M token for Orion-LD queries.
    """
    url = f"{TOKEN_URL}"

    try:
        response = requests.get(url=url, timeout=2)
        response.raise_for_status()

        token_data = response.json()
        token_value = token_data.get("token")

        if token_value:
            return token_value
        else:
            logger.info("Token value not found in response.")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to retrieve token: {e}")
        return None

import json
import csv
import requests
import random
from datetime import datetime

logger = get_app_logger()


CSV_HEADERS = ["latitude", "longitude", "date", "utc_time", "altitude", "speed", "heading", "signal_quality", "distance"]

def fetch_car_entity():
    """
    Fetch car entity data from Orion-LD.
    Uses a custom token when DEV=True, otherwise fetches a token dynamically.
    """
    if DEV:
        try:
            token = TOKEN
            logger.info("Using custom token (DEV mode).")
        except ImportError:
            logger.error("TOKEN is missing in config.py, but DEV=True requires it.")
            return None
    else:
        token = get_m2m_cb_token()
        if not token:
            logger.error("Failed to retrieve token.")
            return None

    url = f"{API_URL}:{API_PORT}/{URL_VERSION}/{ORION_ENTITY_ID}"
    headers = {
        "Authorization": f"Bearer {token}"
    }

    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()

        entity_data = response.json()
        formatted_json = json.dumps(entity_data, indent=4, sort_keys=True)
        logger.info(f"Entity Data:\n{formatted_json}")

        return entity_data 

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch entity data: {e}")
        return None


def format_and_save_data(json_data, output_file='/home/app/function/embedded_files/entity_data.csv'):
    """
    Extracts required attributes, logs the formatted output, and saves it to a CSV file.
    """
    if not json_data:
        logger.error("No data to format.")
        return

    try:
        latitude = round(json_data.get("http://uri.etsi.org/ngsi-ld/location", {}).get("value", {}).get("coordinates", [None, None])[1], 6)
        longitude = round(json_data.get("http://uri.etsi.org/ngsi-ld/location", {}).get("value", {}).get("coordinates", [None, None])[0], 6)
        
        speed = json_data.get("http://uri.fiware.org/ns/data-models#speed", {}).get("value", 0)  
        distance = json_data.get("http://uri.fiware.org/ns/data-models#distanceCollision", {}).get("value", 0)

        raw_heading = json_data.get("http://uri.fiware.org/ns/data-models#heading", {}).get("value", "0")  
        if isinstance(raw_heading, str):
            heading = int(raw_heading.replace("degrees", "").strip()) if "degrees" in raw_heading else int(raw_heading)
        else:
            heading = raw_heading

        signal_quality = json_data.get("http://uri.fiware.org/ns/data-models#signalQuality", {}).get("value", 0)

        date = datetime.utcnow().strftime("%d%m%y")

        utc_time = datetime.utcnow().strftime("%I:%M%p")

        altitude = 286 + random.choice([-1, 0, 1])

        if os.stat(output_file).st_size == 0:
            with open(output_file, mode="w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(CSV_HEADERS)  
                writer.writerow([latitude, longitude, date, utc_time, altitude, speed, heading, signal_quality, distance])
        else:
            with open(output_file, mode="a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([latitude, longitude, date, utc_time, altitude, speed, heading, signal_quality, distance])

        logger.info("Data saved to CSV")

    except Exception as e:
        logger.error(f"Failed to format and save JSON data: {e}")


spec = importlib.util.spec_from_file_location("metric_reporter", "/home/app/function/metric_reporter.py")
metricReporterModule = importlib.util.module_from_spec(spec)
sys.modules["metric_reporter"] = metricReporterModule
spec.loader.exec_module(metricReporterModule)
reporter = metricReporterModule.MetricReporter()

def getOutputName():
    outputName = "output.html"
    return outputName

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
        report.to_file("output.html")
        ret = { "result" : "output.html created from gdrive"}
        return ret
    # Perform EDA on csv or excel file from google drive - END

    # Perform EDA on measurements stored in a db using mintaka - START
    elif 'data_fabric' in req:
        # waiting for input from other partners
        # mintaka, available measurements, etc
        print('Under Development')
        ret = { "result" : "data fabric queried"}

        entity_data = fetch_car_entity()
        if entity_data:
            format_and_save_data(entity_data)

        with open('/home/app/function/embedded_files/entity_data.csv') as data_file:
            df = pd.read_csv(data_file)
            report = ProfileReport(df, title="EDA")
            report.to_file("/home/app/function/embedded_files/output.html")
        print('Write to output Complete')
        ret = { "result" : "EDA report created from datafabric data"}
        return ret

    elif 'test' in req:
        # use test data
        with open('/home/app/function/embedded_files/gps_data.csv') as data_file:
            df = pd.read_csv(data_file)
            report = ProfileReport(df, title="EDA")
            report.to_file("/home/app/function/embedded_files/output.html")
        print('Test Complete')
        ret = { "result" : "EDA report created from test data"}
        return ret
    # Perform EDA on measurements stored in a db using mintaka - END
    
    else:
        print('No valid keys in HTTP request Body')
        ret = { "result" : "EDA not generated"}
        return ret
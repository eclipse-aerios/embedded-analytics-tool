import json
import csv
import requests
import random
from datetime import datetime
from log import get_app_logger
from get_m2m_token import get_m2m_cb_token
from config import API_URL, API_PORT, URL_VERSION, ORION_ENTITY_ID, DEV

logger = get_app_logger()


CSV_HEADERS = ["latitude", "longitude", "date", "utc_time", "altitude", "speed", "heading", "signal_quality", "distance"]

def fetch_car_entity():
    """
    Fetch car entity data from Orion-LD.
    Uses a custom token when DEV=True, otherwise fetches a token dynamically.
    """
    if DEV:
        try:
            from config import TOKEN 
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


def format_and_save_data(json_data, output_file="entity_data.csv"):
    """
    Extracts required attributes, logs the formatted output, and saves it to a CSV file.
    """
    if not json_data:
        logger.error("No data to format.")
        return

    try:
        latitude = round(json_data.get("location", {}).get("value", {}).get("coordinates", [None, None])[1], 6)
        longitude = round(json_data.get("location", {}).get("value", {}).get("coordinates", [None, None])[0], 6)
        
        speed = json_data.get("speed", {}).get("value", 0)  
        distance = round(json_data.get("distanceCollision", {}).get("value", 0), 2)

        raw_heading = json_data.get("heading", {}).get("value", "0")  
        if isinstance(raw_heading, str):
            heading = int(raw_heading.replace("degrees", "").strip()) if "degrees" in raw_heading else int(raw_heading)
        else:
            heading = raw_heading

        signal_quality = json_data.get("signalQuality", {}).get("value", 0)

        date = datetime.utcnow().strftime("%d%m%y")

        utc_time = datetime.utcnow().strftime("%I:%M%p")

        altitude = 286 + random.choice([-1, 0, 1])

        with open(output_file, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(CSV_HEADERS)  
            writer.writerow([latitude, longitude, date, utc_time, altitude, speed, heading, signal_quality, distance])

        logger.info("Data saved to CSV")

    except Exception as e:
        logger.error(f"Failed to format and save JSON data: {e}")


if __name__ == "__main__":
    entity_data = fetch_car_entity()
    if entity_data:
        format_and_save_data(entity_data)

import requests
from config import TOKEN_URL
from log import get_app_logger

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


if __name__ == "__main__":
    token = get_m2m_cb_token()
    if token:
        print(f"Retrieved token: {token}")
    else:
        print("Failed to retrieve token.")

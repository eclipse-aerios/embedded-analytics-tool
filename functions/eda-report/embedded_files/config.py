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

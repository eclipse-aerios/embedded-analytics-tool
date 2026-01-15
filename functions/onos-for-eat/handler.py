# Required imports
import importlib.util
import sys
import json
import os
import requests



# Load Metric Reporter
spec = importlib.util.spec_from_file_location("metric_reporter", "/home/app/function/metric_reporter.py")
metricReporterModule = importlib.util.module_from_spec(spec)
sys.modules["metric_reporter"] = metricReporterModule
spec.loader.exec_module(metricReporterModule)
reporter = metricReporterModule.MetricReporter()


# Function Handler
def handle(req):
    req = json.loads(req)

    # Initial OpenFlow command structure
    openflow_cmd = {
        "flows": [{
            "priority": 100,
            "timeout": 0,
            "isPermanent": True,
            "deviceId": None,
            "treatment": {
                "instructions": []
            },
            "selector": {
                "criteria": []
            }
        }]
    }

    # 0. Check of ONOS credentials are provided
    if ('onos_username' and 'onos_password') in req:
        
        onos_username = req['onos_username']
        onos_password = req['onos_password']
        
    else:
        print('You MUST provide:\n1) onos_username \n2) onos_password')
        return

    # 1a. Check if Controller IP, Controller Port, and deviceId are provided
    if ('controller_ip' and 'controller_port' and 'deviceId') in req:
        
        # Validate input (ask about extra functions in order to implement it)

        # Create the ONOS API url
        controller_ip = req['controller_ip']
        controller_port = req['controller_port']
        url = f"http://{controller_ip}:{controller_port}/onos/v1/flows?appId=org.onosproject.core"

        # Update the deviceId in openflow_cmd
        openflow_cmd["flows"][0]["deviceId"] = req['deviceId']
    else:
        print('You MUST provide:\n1) controller_ip \n2) controller_port \n3) deviceId')
        return
    
    # 1b. Check if priority is provided
    if 'priority' in req:
        # Update the priority in openflow_cmd
        openflow_cmd["flows"][0]["priority"] = req['priority']

    # 1c. Check if lldp_to_controller is provided
    if 'lldp_to_controller' in req:
        if req['lldp_to_controller'] == True:
            # add the appropriate fields in "Filtering"
            openflow_cmd["flows"][0]["treatment"]["instructions"].append({
                    "type": "OUTPUT",
                    "port": "CONTROLLER"
                })
            # add the appropriate fields in "Actions"
            openflow_cmd["flows"][0]["selector"]["criteria"].append({
                    "type": "ETH_TYPE",
                    "ethType": "0x88cc"
                })

    # 2a. Check if src_ip is provided
    if 'src_ip' in req:
        temp_src_ip = req['src_ip']
        src_ip = {
            "type": "IPV4_SRC",
            "ip": f"{temp_src_ip}/32"
        }

        # Add the src_ip in openflow_cmd
        openflow_cmd["flows"][0]["selector"]["criteria"].append(src_ip)

    # 2b. Check if src_mac is provided
    if 'src_mac' in req:
        temp_src_mac = req['src_mac']
        src_mac = {
            "type": "ETH_SRC",
            "mac": temp_src_mac
        }

        # Add the src_mac in openflow_cmd
        openflow_cmd["flows"][0]["selector"]["criteria"].append(src_mac)

    # 2c. Check if dst_ip is provided
    if 'dst_ip' in req:
        temp_dst_ip = req['dst_ip']
        dst_ip = {
            "type": "IPV4_DST",
            "ip": f"{temp_dst_ip}/32"
        }

        # Add the dst_ip in openflow_cmd
        openflow_cmd["flows"][0]["selector"]["criteria"].append(dst_ip)

    # 2d. Check if dst_mac is provided
    if 'dst_mac' in req:
        temp_dst_mac = req['dst_mac']
        dst_mac = {
            "type": "ETH_DST",
            "mac": temp_dst_mac
        }

        # Add the dst_ip in openflow_cmd
        openflow_cmd["flows"][0]["selector"]["criteria"].append(dst_mac)

    # 3a. Check if port_out is provided
    if 'port_out' in req:
        temp_port_out = req['port_out']
        port_out = {
            "type": "OUTPUT",
            "port": temp_port_out
        }

        # Add the port_out in openflow_cmd
        openflow_cmd["flows"][0]["treatment"]["instructions"].append(port_out)

    # 3b. Check if mod_dst_ip is provided
    if 'mod_dst_ip' in req:
        temp_mod_dst_ip = req['mod_dst_ip']
        mod_dst_ip = {
            "type": "L3MODIFICATION",
            "subtype": "IPV4_DST",
            "ip": f"{temp_mod_dst_ip}"
        }

        # Add the mod_dst_ip in openflow_cmd
        openflow_cmd["flows"][0]["treatment"]["instructions"].append(mod_dst_ip)

    # 3c. Check if mod_dst_mac is provided
    if 'mod_dst_mac' in req:
        temp_mod_dst_mac = req['mod_dst_mac']
        mod_dst_mac = {
            "type": "L2MODIFICATION",
            "subtype": "ETH_DST",
            "mac": temp_mod_dst_mac
        }

        # Add the mod_dst_mac in openflow_cmd
        openflow_cmd["flows"][0]["treatment"]["instructions"].append(mod_dst_mac)


    # Create the header
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # Authorisation
    auth = (req['onos_username'], req['onos_password'])

    # Make the API call to install the constructed Flow
    # response = requests.post(url, json=openflow_cmd, headers=headers, auth=auth)

    # if response.status_code == 201 or response.status_code == 200:
    #     print("Flow installed succesfully!")
    # else:
    #     print(f"Error: API call failed with Status Code {response.status_code}.")
    #     print("Response:", response.text)
    
    # Print Functions for Joe to test the output
    print(openflow_cmd)
    return openflow_cmd
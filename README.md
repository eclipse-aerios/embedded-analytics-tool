# aeriOS Embedded Analytics Tool (EAT)

The aeriOS Embedded Analytics Tool (EAT) can be compartmentalised into three roles; these are the analytics framework, function authoring and visualisation. This section will reiterate these primary roles to provide a final architectural view of EAT with additional detailing of the aeriOS Function Template structure. As a result, a holistic EAT architecture is presented.

<img src="./docs/figures/embedded_analytics_tool_arch.drawio.svg" alt="Embedded Analytics Tool Architecture" height="480"/>

EAT provides a framework for the design, implementation, and deployment of specialised functions, which is mainly based on [OpenFaaS](https://docs.openfaas.com/). These may be straightforward policy-based functions for validation use cases or intelligence-based models for smart decision making. The framework supports multiple dashboards for operations (*gateway*) and visualisation (Grafana). The *pushgateway* allows in-function metrics to be exposed to prometheus monitoring which is then visualised through grafana to the user. The *alertmanager* component monitors Prometheus metrics related to the health of the *gateway*, alerting the user if required. 

To access these features of EAT, functions must be created using the [aeriOS template](#aerios-eat-functions).

## Installation
The first step in installation is to clone EAT from the aeriOS GitHub repository and then step into the folder using the change directory command.

```sh
git clone https://github.com/eclipse-aerios/embedded-analytics-tool.git
```

### Running on Windows using KinD

Next, if installing EAT onto a Windows machine for testing.
We support the installation for EAT onto Kubernetes in Docker (KinD).
This step can be skipped if installing onto existing Kubernetes cluster.
These steps ensure the context is using kind-kind and concludes by printing the cluster information.

```sh
chmod +x kind-with-registry.sh
./kind-with-registry.sh
kubectl config use kind-kind
kubectl cluster-info
```

### Installing through Helm
Before installation of the Helm charts you must create the required namespaces for EAT and its functions:
- **embedded-analytics-tool**: namespaces in which the EAT components are installed.
- **openfaas-fn**: namespace in which the EAT functions are deployed.
  
Just apply the [namespaces.yml](./namespaces.yml) file.

```sh
kubectl apply -f namespaces.yml
```

Three *helm install* commands are executed for the *embedded-analytics-tool namespace*, these are for OpenFaaS, Prometheus PushGateway and Grafana.
You can check the successfull creation of the pods using the *kubectl get all* command.

```sh
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

helm install --namespace=embedded-analytics-tool eat-openfaas helm-charts/openfaas2-12.0.5.tgz
helm install --namespace=embedded-analytics-tool eat-push prometheus-community/prometheus-pushgateway --version 2.3.0
helm install --namespace=embedded-analytics-tool eat-grafana grafana/grafana --version 6.57.3

kubectl get all -n embedded-analytics-tool
kubectl get all -n openfaas-fn
```

#### Expose ports
To provide access to the GUIs of the OpenFaaS gateway, Prometheus and Grafana their ports must be exposed. This can achieved through different methods (NodePorts, Ingress...) but the quickest way is using port forwarding to expose them locally in your machine:

```sh
kubectl port-forward --namespace=embedded-analytics-tool svc/gateway 8080:8080 &
kubectl port-forward --namespace=embedded-analytics-tool svc/prometheus 9090:9090 &
kubectl port-forward --namespace=embedded-analytics-tool svc/eat-push-prometheus-pushgateway 9091:9091 &
kubectl port-forward --namespace=embedded-analytics-tool svc/eat-grafana 3000:80 &
```

#### Get passwords
To access the GUIs you must provide authentication for both the OpenFaaS gateway and Grafana.
This execution retrieves the secret for OpenFaaS gateway and uses the OpenFaaS CLI tool to login to the gateway. This is required if you wish to create and deploy functions on the EAT.

```sh
OPENFAAS_USER=$(echo admin)
OPENFAAS_PASS=$(kubectl get secret --namespace=embedded-analytics-tool basic-auth -o jsonpath="{.data.basic-auth-password}" | base64 --decode; echo)
echo -n $OPENFAAS_PASS | faas-cli login --username admin --password-stdin
echo $OPENFAAS_PASS
```

Grafana requires its secret for login, this can be retrieved with the command below.

```sh
GRAFANA_USER=$(echo admin)
GRAFANA_PASS=$(kubectl get secret --namespace=embedded-analytics-tool eat-grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo)
echo $GRAFANA_PASS
```

Both GUIs usernames are set to admin by default.

#### Installation Restart
To clean up all EAT related executions the following commands removed all EAT pods, services and helm installs.
Helm install is retriggered and all pods/services are printed to screen.

```sh
kubectl delete --all deployments --namespace=embedded-analytics-tool
kubectl delete --all services --namespace=embedded-analytics-tool
helm delete --namespace embedded-analytics-tool eat-openfaas
helm delete --namespace embedded-analytics-tool eat-push
helm delete --namespace embedded-analytics-tool eat-grafana
helm install --namespace=embedded-analytics-tool eat-openfaas chart/openfaas2-12.0.5.tgz
helm install --namespace=embedded-analytics-tool prometheus-pushgateway prometheus-community/prometheus-pushgateway --version 2.3.0
helm install --namespace=embedded-analytics-tool grafana grafana/grafana --version 6.57.3
kubectl get all -n embedded-analytics-tool
kubectl get all -n openfaas-fn
```

This completes the installation of EAT.

We welcome your feedback regarding the instruction quality, please communicate any issue you have during the installation process as we can update and improve our documentation.

## aeriOS EAT functions
Detailed information is provided [here](./functions/README.md).

Eclipse aeriOS provides a useful [template](./functions/template/) to develop new functions to be executed by the Embedded Analytics Tool.

In addition, the already developed functions, such as the [HLO's AI explainability function](./functions/hlo-explainer/), are included inside [functions folder](./functions/) of this repository.

## License
This project is licensed under the MIT License.

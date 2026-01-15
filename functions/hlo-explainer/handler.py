import importlib.util
import sys
import json
spec = importlib.util.spec_from_file_location("metric_reporter", "/home/app/function/metric_reporter.py")
metricReporterModule = importlib.util.module_from_spec(spec)
sys.modules["metric_reporter"] = metricReporterModule
spec.loader.exec_module(metricReporterModule)
reporter = metricReporterModule.MetricReporter()

import requests
import json
import os

#Function Imports
import dataclasses as dc
import logging
import pathlib
import pickle
from typing import Any, List, Optional

import dataclasses_json as dcj
import dill
import numpy as np
import sb3_contrib as sb3c
import sb3_contrib.common.maskable.policies as sb3cp
import shap
import torch
import torch.nn as nn

from matplotlib import pyplot as plt
import base64



@dcj.dataclass_json
@dc.dataclass(frozen=True)
class Explanation:
    shapley_values: List[List[float]]
    shapley_expected_values: List[float]
    unmasked_probs: List[float]
    masked_probs: List[float]
    action_mask: List[float]
    feature_names: List[str]
    target_names: List[str]

    def assert_correctness(self) -> None:
        assert isinstance(self.shapley_values, list), type(self.shapley_values)
        assert len(self.shapley_values) == 20604, len(self.shapley_values)
        assert isinstance(self.shapley_values[0], list), type(self.shapley_values[0])
        assert len(self.shapley_values[0]) == 100, len(self.shapley_values[0])
        assert isinstance(self.shapley_values[0][0], float), type(
            self.shapley_values[0][0]
        )

        assert isinstance(self.shapley_expected_values, list), type(
            self.shapley_expected_values
        )
        assert len(self.shapley_expected_values) == 100, len(
            self.shapley_expected_values
        )
        assert isinstance(self.shapley_expected_values[0], float), type(
            self.shapley_expected_values[0]
        )

        assert isinstance(self.unmasked_probs, list), type(self.unmasked_probs)
        assert len(self.unmasked_probs) == 100, len(self.unmasked_probs)
        assert isinstance(self.unmasked_probs[0], float), type(self.unmasked_probs[0])

        assert isinstance(self.masked_probs, list), type(self.masked_probs)
        assert len(self.masked_probs) == 100, len(self.masked_probs)
        assert isinstance(self.masked_probs[0], float), type(self.masked_probs[0])

        assert isinstance(self.action_mask, list), type(self.action_mask)
        assert len(self.action_mask) == 100, len(self.action_mask)
        assert isinstance(self.action_mask[0], int), type(self.action_mask[0])

        assert isinstance(self.feature_names, list), type(self.feature_names)
        assert len(self.feature_names) == 20604, len(self.feature_names)
        assert isinstance(self.feature_names[0], str), type(self.feature_names[0])

        assert isinstance(self.target_names, list), type(self.target_names)
        assert len(self.target_names) == 100, len(self.target_names)
        assert isinstance(self.target_names[0], str), type(self.target_names[0])


class PolicyNetworkWrapper(nn.Module):
    _EPS: float = 1e-12

    def __init__(self, model: sb3c.MaskablePPO, action_mask: np.ndarray) -> None:
        super().__init__()
        self._policy: sb3cp.MaskableActorCriticPolicy = model.policy
        self._action_mask: np.ndarray = action_mask

    def get_unmasked_probs(self, x: torch.Tensor) -> torch.Tensor:
        # get latent representation which is used to determine the action distribution
        with torch.no_grad():
            latent_pi = self._policy.mlp_extractor.forward_actor(x)
            distribution = self._policy._get_action_dist_from_latent(latent_pi)
        return distribution.distribution.probs

    def _get_model_device(self) -> torch.device:
        # assume that all model parameters are on the same device
        return next(self._policy.parameters()).device

    def forward(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        # the data for the model can either be a numpy array or a torch tensor
        # - numpy array,  i.e. if Kernel SHAP is used
        # - torch tensor, i.e. if Deep SHAP is used, or
        #                      if the model inference is done for the data from the environment
        is_using_numpy = isinstance(x, np.ndarray)
        if is_using_numpy:
            x = (
                torch.from_numpy(x)
                .type(torch.float32)
                .to(device=self._get_model_device())
            )

        unmasked_probs = self.get_unmasked_probs(x)

        # repeat the action mask for each example in the batch
        # [ACTION] -> [BATCH, ACTION]
        batch_size = x.shape[0]
        action_mask = np.tile(self._action_mask, reps=(batch_size, 1))
        mask_tensor = (
            torch.from_numpy(action_mask)
            .type(torch.bool)
            .to(device=unmasked_probs.device)
        )
        # replace probabilities that should be masked with a small value (eps)
        masked_probs = torch.where(
            condition=mask_tensor, input=unmasked_probs, other=torch.tensor(self._EPS)
        ).requires_grad_(True)

        if is_using_numpy:
            return masked_probs.cpu().detach().numpy()

        return masked_probs


class DeepExplainerWrapper:
    def __init__(self, model: PolicyNetworkWrapper, ground_dataset: List[np.ndarray]):
        self._explainer: shap.DeepExplainer = shap.DeepExplainer(model, ground_dataset)
        self.expected_values: np.ndarray = self._explainer.expected_value

    def shap_values(self, x: torch.Tensor) -> np.ndarray:
        return self._explainer.shap_values(x, check_additivity=False)


class KernelExplainerWrapper:
    def __init__(self, model: PolicyNetworkWrapper, ground_dataset: List[np.ndarray]):
        self._explainer: shap.KernelExplainer = shap.KernelExplainer(
            model, ground_dataset
        )
        self.expected_values: np.ndarray = self._explainer.expected_value

    def shap_values(self, x: torch.Tensor) -> np.ndarray:
        x = x.cpu().numpy()
        return self._explainer.shap_values(x, silent=True, gc_collect=False)


class PolicyNetworkExplainer:
    def __init__(
        self,
        model: Optional[sb3c.MaskablePPO],
        ground_dataset: np.ndarray,
        feature_names: List[str],
        target_names: List[str],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        # the model is optional because it can be set later
        # because during pickling the model must be set to None
        self._model: Optional[sb3c.MaskablePPO] = model
        self._logger: Optional[logging.Logger] = logger

        # array with example input states for the policy network (1D arrays)
        # these are observations transformed to the input for the policy network
        # thus its a 2D array, (num samples) x (num features)
        # here, its transformed to a torch tensor, because only the deep explainer is actually used
        self._ground_dataset: torch.Tensor = torch.tensor(ground_dataset)

        # list with feature names corresponding to the input for the model
        self._feature_names: List[str] = feature_names

        # list with target names corresponding to the output of the model
        self._target_names: List[str] = target_names

    def set_logger(self, logger: logging.Logger) -> None:
        self._logger = logger

    def set_model(self, model: sb3c.MaskablePPO) -> None:
        self._model = model

    def export_as_dill(self, save_path: pathlib.Path) -> None:
        explainer_path = save_path / "explainer.dill"
        model = self._model
        logger = self._logger
        self.set_model(None)
        self.set_logger(None)
        with open(explainer_path, "wb") as f:
            dill.dump(self, f)
        self.set_model(model)
        self.set_logger(logger)

    @classmethod
    def load(
        cls, data_path: pathlib.Path, logger: Optional[logging.Logger] = None
    ) -> "PolicyNetworkExplainer":
        model_path = data_path / "allocation_model_trained"
        model = sb3c.MaskablePPO.load(model_path, device="cpu")

        feature_names_path = data_path / "feature_names.pkl"
        with open(feature_names_path, "rb") as f:
            feature_names = pickle.load(f)

        target_names_path = data_path / "target_names.pkl"
        with open(target_names_path, "rb") as f:
            target_names = pickle.load(f)

        ground_dataset_path = data_path / "ground_dataset.pkl"
        with open(ground_dataset_path, "rb") as f:
            ground_dataset = pickle.load(f)

        return PolicyNetworkExplainer(
            model=model,
            ground_dataset=ground_dataset,
            feature_names=feature_names,
            target_names=target_names,
            logger=logger,
        )

    def explain(
        self,
        env_obs: Any,
        action_mask: np.ndarray,
        explainer_type: str = "deep",
    ) -> Explanation:
        assert self._model is not None

        action_mask = action_mask.astype(np.int8)
        policy_network = PolicyNetworkWrapper(
            model=self._model, action_mask=action_mask
        )

        if self._logger is not None:
            self._logger.debug("Constructing explainer instance")
        # create a new explainer instance because it varies for different action masks
        if explainer_type == "kernel":
            raise NotImplementedError("Kernel SHAP is not implemented yet")
            explainer = KernelExplainerWrapper(
                model=policy_network, ground_dataset=self._ground_dataset
            )
        elif explainer_type == "deep":
            explainer = DeepExplainerWrapper(
                model=policy_network, ground_dataset=self._ground_dataset
            )
        else:
            raise ValueError(f"Unknown explainer: {explainer}")

        if self._logger is not None:
            self._logger.debug("Converting observation to tensor")
        # basically these steps are taken out from the sb3 implementation
        # in the case of the current implementation these two steps result in just:
        #   env_obs_tensor = env_obs.unsqueeze(0)
        # but they are left for the sake of future changes
        # obsolete: change numpy arrays in dict to torch tensors
        env_obs_tensor, _ = self._model.policy.obs_to_tensor(env_obs)
        # obsolete: change dict to torch tensor
        env_obs_tensor = self._model.policy.extract_features(env_obs_tensor)
        # sanity check that the input is not a tuple of tensors
        assert isinstance(env_obs_tensor, torch.Tensor)

        if env_obs_tensor.shape[0] != 1:
            raise ValueError("Only one observation can be explained at a time")

        if self._logger is not None:
            self._logger.debug("Calculating probabilities")
        # probabilities for visualization
        unmasked_probs = (
            policy_network.get_unmasked_probs(env_obs_tensor).cpu().detach().numpy()
        )
        masked_probs = policy_network(env_obs_tensor).cpu().detach().numpy()

        if self._logger is not None:
            self._logger.debug("Calculating SHAP values")
        shap_values = explainer.shap_values(env_obs_tensor)

        explanation = Explanation(
            shapley_values=shap_values.squeeze().tolist(),
            shapley_expected_values=explainer.expected_values.tolist(),
            unmasked_probs=unmasked_probs.squeeze().tolist(),
            masked_probs=masked_probs.squeeze().tolist(),
            action_mask=action_mask.tolist(),
            feature_names=self._feature_names,
            target_names=self._target_names,
        )
        explanation.assert_correctness()

        ## Added by Joe
        # Random Shap values are generated for plot, this needs to be swapped out with real values
        shap_values = np.random.randn(*shap_values.shape)
        shap_values = np.transpose(shap_values, (2, 0, 1))
        shap_values = shap_values[action_mask == 1]
        shap_values = list(shap_values)

        # filter target names to active ones
        target_names = list(np.array(self._target_names)[action_mask == 1])

        shap.summary_plot(
            shap_values=shap_values,
            feature_names=self._feature_names,
            class_names=target_names,
            plot_size=(25, 12),
            max_display=25,
            show=False,
        )
        ## values above need to be changed to real values

        ## Added by Joe
        #Save Figure which is then encoded and added to dashboard and posted to grafana
        plt.savefig("shap.svg")
        with open("shap.svg", "rb") as image_file:
            image_base64 = str(base64.b64encode(image_file.read()),'utf-8')
            #To import from json file
            with open('/home/app/function/dashboard.json') as dashboard_file:
                dashboard = json.load(dashboard_file)
            newHtml = '<img src="data:image/svg+xml;base64,{}" />'.format(image_base64)
            dashboard['panels'][0]['options']['html'] = newHtml
            dashboard['panels'][0]['title'] = "HLO-Explainer"
            dashboard['title'] = "HLO-Dashboard"
            
            reqBody = {
            "dashboard": dashboard,
            "message": "Updating Dashboard from File",
            "overwrite": True
            }
            reqUrl = 'http://eat-grafana.embedded-analytics-tool.cluster.local:80/api/dashboards/db'
            reqHeaders = {'Accept': 'application/json', 'Content-Type': 'application/json'}
            r = requests.post(reqUrl, json=reqBody, headers=reqHeaders, auth=(os.environ['GRAFANA_USER'], os.environ['GRAFANA_PASS']))
        ## Above code is complete

        return explanation

    def test(self) -> None:
        explanation = self.explain(
            env_obs=torch.randn(20604),
            action_mask=np.ones(100),
            explainer_type="deep",
        )
        explanation_json = explanation.to_json()
        explanation_dict = explanation.to_dict()
        if self._logger is not None:
            self._logger.info("Test passed")
            return json.dumps({"Test" : "Passed"})

def handle(req):
    # the logger will be used internally in the explainer
    logger = logging.getLogger("explainer")
    logging.basicConfig(level=logging.DEBUG)

    # data path should be a directory with the following files:
    #   allocation_model_trained.zip (hlo model)
    #   ground_dataset.pkl (2d numpy array)
    #   feature_names.pkl (list of strings)
    #   target_names.pkl (list of strings)
    data_path = pathlib.Path(__file__).parent.parent / "function/data"

    # creates an instance of the explainer
    explainer = PolicyNetworkExplainer.load(data_path, logger)

    # verifies that the explainer works correctly
    output = explainer.test()

    # saves the explainer instance
    # however, this step became obsolete
    # explainer.export_as_dill(data_path)
    
    return output

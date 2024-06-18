# TODO
# re-inspect statistical model for the priors - compare with pymc-marketing
# scaling of channels and sales?

from typing import Any
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pymc_marketing.mmm.delayed_saturated_mmm import DelayedSaturatedMMM
import arviz as az
import seaborn as sns

import pymc as pm
from pymc_marketing.mmm.utils import (
    transform_1d_array,
)
from pymc_marketing.mmm.tvp import create_time_varying_intercept
from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation

class MelkConfounder():
    def __init__(self, data: pd.DataFrame,  true_values: dict, datevarname: str) -> None:
        self.data = data
        self.datevarname = datevarname
        self.dates = data[datevarname]
        self.true_values = true_values
        self.model_structure: pm.Model = None
        # self.model_simulate: pm.Model = None
        self.model: pm.Model = None
        self.sampler_config: dict | None = None
        self.idata: az.InferenceData | None = None
        self.define_model_structure()
        self.observedvars = ["ch_fb", "ch_sem", "y"]

    def sampler_config_default(self) -> dict:
        return {
            "target_accept": 0.90,
            "chains": 4,
            "cores": 4
        }

    def define_model_structure(self) -> None: 
        coords_mutable = { self.datevarname: self.dates }

        # define the model
        with pm.Model(coords_mutable=coords_mutable) as self.model_structure:
            beta_y0 = pm.Normal("beta_y0")                

            beta_fb_0 = pm.Normal("beta_fb_0")            
            beta_fb_y = pm.Normal("beta_fb_y")            
            beta_fb_sem = pm.Normal("beta_fb_sem")        

            beta_sem_0 = pm.Normal("beta_sem_0")          
            beta_sem_y = pm.Normal("beta_sem_y")          
            # observation noise on Y
            sigma_y = pm.HalfNormal("sigma_y")
            
            # core nodes and causal relationships
            ch_fb = pm.Normal("ch_fb", mu=beta_fb_0, sigma=1, dims=self.datevarname)
            ch_sem = pm.Normal("ch_sem", mu=beta_sem_0 + beta_fb_sem * ch_fb, sigma=1, dims=self.datevarname)
            y_mu = pm.Deterministic("y_mu", beta_y0 + (beta_fb_y * ch_fb) + (beta_sem_y * ch_sem), dims=self.datevarname)
            y = pm.Normal("y", mu=y_mu, sigma=sigma_y)

    # def simulate_model_from_true_values(self) -> None:
    #     self.model_simulate = pm.do(self.model_structure, self.true_values)
    #     with self.model_simulate:
    #         sim = pm.sample_prior_predictive()

    #     simulated_observed = { var : sim.prior[var].values.flatten() for var in self.observedvars }
    #     df = pd.DataFrame(simulated_observed)
    #     return df


    def define_model(self) -> None: 
        # define the model
        with pm.Model() as self.model:
            beta_y0 = pm.Normal("beta_y0")                

            beta_fb_0 = pm.Normal("beta_fb_0")            
            beta_fb_y = pm.Normal("beta_fb_y")            
            beta_fb_sem = pm.Normal("beta_fb_sem")        

            beta_sem_0 = pm.Normal("beta_sem_0")          
            beta_sem_y = pm.Normal("beta_sem_y")          
            # observation noise on Y
            sigma_y = pm.HalfNormal("sigma_y")
            
            # core nodes and causal relationships
            ch_fb = pm.Normal("ch_fb", mu=beta_fb_0, sigma=1, observed=self.data["ch_fb"])
            ch_sem = pm.Normal("ch_sem", mu=beta_sem_0 + beta_fb_sem * ch_fb, sigma=1, observed=self.data["ch_sem"])
            y_mu = pm.Deterministic("y_mu", beta_y0 + (beta_fb_y * ch_fb) + (beta_sem_y * ch_sem))
            y = pm.Normal("y", mu=y_mu, sigma=sigma_y, observed=self.data["y"])

    def fit(self) -> None:
        sampler_config_used = self.sampler_config_default().copy()

        self.model = pm.observe(self.model_structure, self.data[self.observedvars])

        # sample the model
        with self.model:
            self.idata = pm.sample(**sampler_config_used)

        # add posterior predictive samples
        with self.model:
            pm.sample_posterior_predictive(self.idata, extend_inferencedata=True)
    

        
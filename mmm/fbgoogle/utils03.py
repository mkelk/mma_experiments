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
    def __init__(self, data_incoming: pd.DataFrame,  true_values: dict, datevarname: str) -> None:
        self.data_incoming = data_incoming
        self.data: pd.DataFrame = None
        self.datevarname = datevarname
        self.dates = data_incoming[datevarname]
        self.true_values = true_values
        # self.model_structure: pm.Model = None
        # self.model_simulate: pm.Model = None
        self.model: pm.Model = None
        self.sampler_config: dict | None = None
        self.idata: az.InferenceData | None = None
        # self.define_model_structure()
        self.observedvars = ["ch_fb", "ch_sem", "y"]

    def sampler_config_default(self) -> dict:
        return {
            "target_accept": 0.90,
            "chains": 4,
            "cores": 4
        }

    def prepare_data(self) -> None:
        self.data = self.data_incoming.copy()
        self.data["dayofweek"] = self.data.assign(dayofweek=lambda x: x["date"].dt.dayofweek)["dayofweek"]
        daysofweek = range(6)
        for dayofweek in daysofweek:
            self.data[f"dayofweek_{dayofweek}"] = np.where(self.data["dayofweek"] == dayofweek, 1, 0)
        # remove day of week
        self.data = self.data.drop(columns=["dayofweek"])

    def define_model(self) -> None: 
        if self.data is None:
            self.prepare_data()

        coords_mutable = { self.datevarname: self.dates }
        coords = { "dayofweek" : [f"dayofweek_{d}" for d in range(6)]}
        dayofweek_cols = [f"dayofweek_{d}" for d in range(6)]

        # define the model
        with pm.Model(coords_mutable=coords_mutable, coords=coords) as self.model:
            beta_y0 = pm.Normal("beta_y0")                

            beta_fb_0 = pm.Normal("beta_fb_0")            
            beta_fb_y = pm.Normal("beta_fb_y")            
            beta_fb_sem = pm.Normal("beta_fb_sem")        

            beta_sem_0 = pm.Normal("beta_sem_0")          
            beta_sem_y = pm.Normal("beta_sem_y")          
            # observation noise on Y
            sigma_y = pm.HalfNormal("sigma_y")

            # day of week
            dayofweek_indicator = pm.MutableData(
                name="dayofweek_indicator",
                value=self.data[dayofweek_cols],
                dims=("date", "dayofweek"),
            )

            gamma_dayofweek = pm.Normal("gamma_dayofweek", dims=("dayofweek"))            
            dayofweek_effect = pm.math.dot(dayofweek_indicator, gamma_dayofweek)

            # core nodes and causal relationships
            ch_fb = pm.Normal("ch_fb", mu=beta_fb_0, sigma=1, dims=self.datevarname, observed=self.data["ch_fb"])
            ch_sem = pm.Normal("ch_sem", mu=beta_sem_0 + beta_fb_sem * ch_fb, sigma=1, dims=self.datevarname, observed=self.data["ch_sem"])
            y_mu = pm.Deterministic("y_mu", beta_y0 + (beta_fb_y * ch_fb) + (beta_sem_y * ch_sem + dayofweek_effect), dims=self.datevarname)
            y = pm.Normal("y", mu=y_mu, sigma=sigma_y, observed=self.data["y"])

    def fit(self) -> None:
        if self.model is None:
            self.define_model()

        sampler_config_used = self.sampler_config_default().copy()

        # self.model = pm.observe(self.model_structure, self.data[self.observedvars])

        # sample the model
        with self.model:
            self.idata = pm.sample(**sampler_config_used)

        # add posterior predictive samples
        with self.model:
            pm.sample_posterior_predictive(self.idata, extend_inferencedata=True)
    

        
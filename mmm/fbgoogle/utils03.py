# TODO
# re-inspect statistical model for the priors - compare with pymc-marketing
# scaling of channels and sales?
# get rid of y_obs_dim_2

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
from xarray import DataArray, Dataset

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
        self.y_org = self.data_incoming["y"]

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

            ch_fb = pm.Data("ch_fb", self.data["ch_fb"], dims=self.datevarname)
            # ch_sem = pm.Data("ch_sem", self.data["ch_sem"], dims=self.datevarname)

            # beta_fb_0 = pm.Normal("beta_fb_0")            
            beta_fb_y = pm.Normal("beta_fb_y")            
            # beta_fb_sem = pm.Normal("beta_fb_sem")        

            # beta_sem_0 = pm.Normal("beta_sem_0")          
            # beta_sem_y = pm.Normal("beta_sem_y")          
            
            # Natural, level of sales, without any fb or sem
            beta_y0 = pm.Normal("beta_y0")                
            sigma_y = pm.HalfNormal("sigma_y")
            # intercept = pm.Normal("intercept", mu=beta_y0, sigma=sigma_y, dims=self.datevarname)

            # # day of week
            # dayofweek_indicator = pm.MutableData(
            #     name="dayofweek_indicator",
            #     value=self.data[dayofweek_cols],
            #     dims=("date", "dayofweek"),
            # )

            # gamma_dayofweek = pm.Normal("gamma_dayofweek", dims=("dayofweek"))            
            # dayofweek_effect = pm.math.dot(dayofweek_indicator, gamma_dayofweek)

            # core nodes and causal relationships
            # ch_fb = pm.Normal("ch_fb", mu=beta_fb_0, sigma=1, dims=self.datevarname, observed=self.data["ch_fb"])
            # ch_sem = pm.Normal("ch_sem", mu=beta_sem_0 + beta_fb_sem * ch_fb, sigma=1, dims=self.datevarname, observed=self.data["ch_sem"])
            # ch_sem = pm.Normal("ch_sem", mu=beta_sem_0 + beta_fb_sem * ch_fb, sigma=1, dims=self.datevarname, observed=self.data["ch_sem"])
            
            y_mu_fb = pm.Deterministic("y_mu_fb", beta_fb_y * ch_fb, dims=self.datevarname)
            # y_mu_sem = pm.Deterministic("y_mu_sem", beta_y0 + (beta_sem_y * ch_sem), dims=self.datevarname)
            # y_mu_sem = pm.Deterministic("y_mu_sem", beta_sem_y * ch_sem, dims=self.datevarname)
            y_mu = pm.Deterministic("y_mu",  beta_y0 + y_mu_fb + sigma_y, dims=self.datevarname)
            # y_mu = pm.Deterministic("y_mu",  beta_y0 + y_mu_fb + y_mu_sem, dims=self.datevarname)
            # y_mu = pm.Deterministic("y_mu", beta_y0 + (beta_fb_y * ch_fb) + (beta_sem_y * ch_sem), dims=self.datevarname)
            # y_mu = pm.Deterministic("y_mu", beta_y0 + (beta_fb_y * ch_fb) + (beta_sem_y * ch_sem) + dayofweek_effect, dims=self.datevarname)
            y_obs = pm.Normal("y_obs", mu=y_mu, observed=self.data["y"])

    def fit(self) -> None:
        if self.model is None:
            self.define_model()

        sampler_config_used = self.sampler_config_default().copy()

        # sample the model
        with self.model:
            self.idata = pm.sample(**sampler_config_used)

        # add posterior predictive samples
        with self.model:
            self.posterior_predictive = pm.sample_posterior_predictive(self.idata).posterior_predictive
    
    def plot_posterior_predictive(self) -> pd.DataFrame:
        with self.model:
            pp = pm.sample_posterior_predictive(self.idata, var_names=["y_mu"])
        
        plotdata = { 
            "date" : self.dates }

        extract_vars = ["y_mu"]
        for plotvar in extract_vars:
            plotdata[plotvar] = pp.posterior_predictive[plotvar].mean(dim=["chain", "draw"])

        plotdata_df = pd.DataFrame(data=plotdata)
        fig, ax = plt.subplots()
        sns.lineplot(x="date", y="y", color="black", data=self.data_incoming, ax=ax)
        ax.set(title="Sales (Target Variable)", xlabel="date", ylabel="y (thousands)");
        sns.lineplot(x="date", y="y_mu", color="red", data=plotdata_df, ax=ax)  

        for hdi_prob, alpha in zip((0.94, 0.50), (0.2, 0.4), strict=True):
            likelihood_hdi: DataArray = az.hdi(
                ary=pp.posterior_predictive, hdi_prob=hdi_prob
            )["y_mu"]

            ax.fill_between(
                x=self.dates,
                y1=likelihood_hdi[:, 0],
                y2=likelihood_hdi[:, 1],
                color="C0",
                alpha=alpha,
                label=f"${100 * hdi_prob}\%$ HDI",  # noqa: W605
            )


    def get_errors(self) -> DataArray:
        """Get model errors posterior distribution.

        errors = true values - predicted

        Returns
        -------
        DataArray
        """
        try:
            posterior_predictive_data: Dataset = self.posterior_predictive

        except Exception as e:
            raise RuntimeError(
                "Make sure the model has bin fitted and the posterior predictive has been sampled!"
            ) from e

        target_array = self.y_org

        if len(target_array) != len(posterior_predictive_data.y_obs_dim_2):
            raise ValueError(
                "The length of the target variable doesn't match the length of the date column. "
                "If you are computing out-of-sample errors, please overwrite `self.y` with the "
                "corresponding (non-transformed) target variable."
            )

        target = (
            pd.Series(target_array, index=self.posterior_predictive.y_obs_dim_2)
            .rename_axis("y_obs_dim_2")
            .to_xarray()
        )

        errors = (
            (target - posterior_predictive_data)["y"]
            .rename("errors")
            .transpose(..., "y_obs_dim_2")
        )

        return errors

            
    def plot_errors(
        self, ax: plt.Axes = None, **plt_kwargs: Any
    ) -> plt.Figure:
        """Plot model errors by taking the difference between true values and predicted.

        errors = true values - predicted

        Parameters
        ----------
        ax : plt.Axes, optional
            Matplotlib axis object.
        **plt_kwargs
            Keyword arguments passed to `plt.subplots`.

        Returns
        -------
        plt.Figure
        """
        errors = self.get_errors()

        if ax is None:
            fig, ax = plt.subplots(**plt_kwargs)
        else:
            fig = ax.figure

        for hdi_prob, alpha in zip((0.94, 0.50), (0.2, 0.4), strict=True):
            errors_hdi = az.hdi(ary=errors, hdi_prob=hdi_prob)

            ax.fill_between(
                x=self.posterior_predictive.date,
                y1=errors_hdi["errors"].sel(hdi="lower"),
                y2=errors_hdi["errors"].sel(hdi="higher"),
                color="C3",
                alpha=alpha,
                label=f"${100 * hdi_prob}\%$ HDI",  # noqa: W605
            )

        ax.plot(
            self.posterior_predictive.date,
            errors.mean(dim=("chain", "draw")).to_numpy(),
            color="C3",
            label="Errors Mean",
        )

        ax.axhline(y=0.0, linestyle="--", color="black", label="zero")
        ax.legend()
        ax.set(
            title="Errors Posterior Distribution",
            xlabel="date",
            ylabel="true - predictions",
        )
        return fig

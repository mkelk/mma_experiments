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

class MelkDelayedSaturatedMMM(DelayedSaturatedMMM):

    def plot_control_parameter(self, original_scale: bool = False, **plt_kwargs: Any) -> plt.Figure:
        """
        Plot the posterior distribution of the control parameter - optionally scaled back to the original scale.
        """
        param_name = "gamma_control"
        scale = self.get_observation_scale() if original_scale else 1.0
        param_samples_df = pd.DataFrame(
            data=az.extract(data=self.fit_result, var_names=[param_name]).T / scale,
            columns=self.control_columns,
        )

        fig, ax = plt.subplots(**plt_kwargs)
        sns.violinplot(data=param_samples_df, orient="h", ax=ax)
        ax.set(
            title=f"Posterior Distribution: {param_name} Parameter",
            xlabel=param_name,
            ylabel="index",
        )
        return fig
    
    def get_observation_scale(self) -> float:
        """
        Returns the scale factor that is applied to the original y values before they are modeled.
        """
        scale_array = np.array([0.0, 1.0])
        tmp = transform_1d_array(self.get_target_transformer().transform, scale_array)
        # an original y diff of 1.0 is scaled with this factor inside the model
        scale = tmp[1] - tmp[0]
        return scale
    


    def build_model(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        **kwargs,
    ) -> None:
        """
        Builds a probabilistic model using PyMC for marketing mix modeling.

        The model incorporates channels, control variables, and Fourier components, applying
        adstock and saturation transformations to the channel data. The final model is
        constructed with multiple factors contributing to the response variable.

        Parameters
        ----------
        X : pd.DataFrame
            The input data for the model, which should include columns for channels,
            control variables (if applicable), and Fourier components (if applicable).

        y : Union[pd.Series, np.ndarray]
            The target/response variable for the modeling.

        **kwargs : dict
            Additional keyword arguments that might be required by underlying methods or utilities.

        Attributes Set
        ---------------
        model : pm.Model
            The PyMC model object containing all the defined stochastic and deterministic variables.

        Examples
        --------
        custom_config = {
            'intercept': {'dist': 'Normal', 'kwargs': {'mu': 0, 'sigma': 2}},
            'beta_channel': {'dist': 'LogNormal', 'kwargs': {'mu': 1, 'sigma': 3}},
            'alpha': {'dist': 'Beta', 'kwargs': {'alpha': 1, 'beta': 3}},
            'lam': {'dist': 'Gamma', 'kwargs': {'alpha': 3, 'beta': 1}},
            'likelihood': {'dist': 'Normal',
                'kwargs': {'sigma': {'dist': 'HalfNormal', 'kwargs': {'sigma': 2}}}
            },
            'gamma_control': {'dist': 'Normal', 'kwargs': {'mu': 0, 'sigma': 2}},
            'gamma_fourier': {'dist': 'Laplace', 'kwargs': {'mu': 0, 'b': 1}}
        }

        model = DelayedSaturatedMMM(
                    date_column="date_week",
                    channel_columns=["x1", "x2"],
                    control_columns=[
                        "event_1",
                        "event_2",
                        "t",
                    ],
                    adstock_max_lag=8,
                    yearly_seasonality=2,
                    model_config=custom_config,
                )
        """

        self.intercept_dist = self._get_distribution(
            dist=self.model_config["intercept"]
        )
        self.beta_channel_dist = self._get_distribution(
            dist=self.model_config["beta_channel"]
        )
        self.lam_dist = self._get_distribution(dist=self.model_config["lam"])
        self.alpha_dist = self._get_distribution(dist=self.model_config["alpha"])
        self.gamma_control_dist = self._get_distribution(
            dist=self.model_config["gamma_control"]
        )
        self.gamma_fourier_dist = self._get_distribution(
            dist=self.model_config["gamma_fourier"]
        )

        self._generate_and_preprocess_model_data(X, y)
        with pm.Model(
            coords=self.model_coords,
            coords_mutable=self.coords_mutable,
        ) as self.model:
            channel_data_ = pm.MutableData(
                name="channel_data",
                value=self.preprocessed_data["X"][self.channel_columns],
                dims=("date", "channel"),
            )

            target_ = pm.MutableData(
                name="target",
                value=self.preprocessed_data["y"],
                dims="date",
            )

            if self.time_varying_intercept:
                time_index = pm.Data(
                    "time_index",
                    self._time_index,
                    dims="date",
                )
                intercept = create_time_varying_intercept(
                    time_index,
                    self._time_index_mid,
                    self._time_resolution,
                    self.intercept_dist,
                    self.model_config,
                )
            else:
                intercept = self.intercept_dist(
                    name="intercept", **self.model_config["intercept"]["kwargs"]
                )

            beta_channel = self.beta_channel_dist(
                name="beta_channel",
                **self.model_config["beta_channel"]["kwargs"],
                dims=("channel",),
            )
            alpha = self.alpha_dist(
                name="alpha",
                dims="channel",
                **self.model_config["alpha"]["kwargs"],
            )
            lam = self.lam_dist(
                name="lam",
                dims="channel",
                **self.model_config["lam"]["kwargs"],
            )

            channel_adstock = pm.Deterministic(
                name="channel_adstock",
                var=geometric_adstock(
                    x=channel_data_,
                    alpha=alpha,
                    l_max=self.adstock_max_lag,
                    normalize=True,
                    axis=0,
                ),
                dims=("date", "channel"),
            )
            channel_adstock_saturated = pm.Deterministic(
                name="channel_adstock_saturated",
                var=logistic_saturation(x=channel_adstock, lam=lam),
                dims=("date", "channel"),
            )

            channel_contributions_var = channel_adstock_saturated * beta_channel
            channel_contributions = pm.Deterministic(
                name="channel_contributions",
                var=channel_contributions_var,
                dims=("date", "channel"),
            )

            mu_var = intercept + channel_contributions.sum(axis=-1)

            if (
                self.control_columns is not None
                and len(self.control_columns) > 0
                and all(
                    column in self.preprocessed_data["X"].columns
                    for column in self.control_columns
                )
            ):
                gamma_control = self.gamma_control_dist(
                    name="gamma_control",
                    dims="control",
                    **self.model_config["gamma_control"]["kwargs"],
                )

                control_data_ = pm.MutableData(
                    name="control_data",
                    value=self.preprocessed_data["X"][self.control_columns],
                    dims=("date", "control"),
                )

                control_contributions = pm.Deterministic(
                    name="control_contributions",
                    var=control_data_ * gamma_control,
                    dims=("date", "control"),
                )

                mu_var += control_contributions.sum(axis=-1)

            if (
                hasattr(self, "fourier_columns")
                and self.fourier_columns is not None
                and len(self.fourier_columns) > 0
                and all(
                    column in self.preprocessed_data["X"].columns
                    for column in self.fourier_columns
                )
            ):
                fourier_data_ = pm.MutableData(
                    name="fourier_data",
                    value=self.preprocessed_data["X"][self.fourier_columns],
                    dims=("date", "fourier_mode"),
                )

                gamma_fourier = self.gamma_fourier_dist(
                    name="gamma_fourier",
                    dims="fourier_mode",
                    **self.model_config["gamma_fourier"]["kwargs"],
                )

                fourier_contribution = pm.Deterministic(
                    name="fourier_contributions",
                    var=fourier_data_ * gamma_fourier,
                    dims=("date", "fourier_mode"),
                )

                mu_var += fourier_contribution.sum(axis=-1)

            mu = pm.Deterministic(name="mu", var=mu_var, dims="date")

            self._create_likelihood_distribution(
                dist=self.model_config["likelihood"],
                mu=mu,
                observed=target_,
                dims="date",
            )

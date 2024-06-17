from typing import Any
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pymc_marketing.mmm.delayed_saturated_mmm import DelayedSaturatedMMM
import arviz as az
import seaborn as sns

from pymc_marketing.mmm.utils import (
    transform_1d_array,
)

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
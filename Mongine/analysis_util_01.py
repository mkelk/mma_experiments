from abc import ABC, abstractmethod
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az


class MMM():
    def __init__(self, data):
        self.data_raw = data
        self.modelname: str # must be set in subclass
        self.channelnames: str # must be set in subclass   
        self.salesname = 'sales'
        self.datename = 'date'
        self.dates = self.data_raw[self.datename]
        self.model: pm.Model = None

    def set_scaling(self):
        """
        Set the scaling of the channels and sales to get more normalized data
        """
        self.data_scaled = self.data_raw.copy()
        self.channelscale = {}
        for channel in self.channelnames:
            self.channelscale[channel] = self.data_scaled[channel].median()
            self.data_scaled[channel] = self.data_scaled[channel] / self.channelscale[channel]
        
        self.salesscale = self.data_scaled[self.salesname].median()
        self.data_scaled[self.salesname] = self.data_scaled[self.salesname] / self.salesscale

    def define_model(self):
        """
        Define the model based on the modelname
        """
        raise NotImplementedError("Subclass must implement abstract method 'define_model()")


    def fit(self):
        """
        Fit the model
        """
        if self.model is None:
            self.define_model()        
        with self.model:
            self.idata = pm.sample()


    def plot_posterior_predictive(self, plot_kwargs=None):
        """
        Plot the posterior predictive of sales
        """
        with self.model:
            pp = pm.sample_posterior_predictive(self.idata, var_names=["mu_y"])
        
        plotdata = { 
            "date" : self.dates }

        extract_vars = ["mu_y"]
        for plotvar in extract_vars:
            plotdata[plotvar] = pp.posterior_predictive[plotvar].mean(dim=["chain", "draw"])

        plotdata_df = pd.DataFrame(data=plotdata)
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 6)
        sns.lineplot(x="date", y="sales", color="black", data=self.data_scaled, ax=ax)
        ax.set(title="Sales (Target Variable)", xlabel="date", ylabel="y (scaled)");
        sns.lineplot(x="date", y="mu_y", color="red", data=plotdata_df, ax=ax)  

        for hdi_prob, alpha in zip((0.94, 0.50), (0.2, 0.4), strict=True):
            likelihood_hdi: DataArray = az.hdi(
                ary=pp.posterior_predictive, hdi_prob=hdi_prob
            )["mu_y"]

            ax.fill_between(
                x=self.dates,
                y1=likelihood_hdi[:, 0],
                y2=likelihood_hdi[:, 1],
                color="C0",
                alpha=alpha,
                label=f"${100 * hdi_prob}\%$ HDI",  
            )

    def plot_parm_dist(self):
        az.plot_posterior(
            self.idata,
            var_names=self.fittedparmnames,
            figsize=(12, 6),
        )
        plt.tight_layout();



# class MMMGoogleStraight(MMM):
#     def __init__(self, data):
#         super().__init__(data)
#         self.modelname = 'google_straight'
#         self.channelnames = ['spend_fb']
#         self.set_scaling()
  
#     def define_model(self):
#         """
#         Define the model
#         """
#         coords = { self.datename: self.dates }
#         with pm.Model(coords=coords) as self.model:
#             # variables
#             spend_google = pm.Data('spend_google', self.data_scaled['spend_google'].values, dims=(self.datename))

#             # Priors
#             beta_google = pm.Normal('beta_google', mu=1, sigma=1)

#             sigma = pm.HalfNormal('sigma')

#             # Expected value
#             mu_y = pm.Deterministic('mu_y', beta_google * spend_google, dims=(self.datename))

#             # Likelihood
#             y = pm.Normal('y', mu=mu_y, sigma=sigma, observed=self.data_scaled[self.salesname].values, dims=(self.datename))


class MMMChannelsStraight(MMM):
    def __init__(self, data, channelnames, allowIntercept: bool = True):
        super().__init__(data)
        self.modelname = 'google_fb_straight'
        self.channelnames = channelnames
        self.fittedparmnames = ['beta', 'sigma']
        self.set_scaling()
        self.allowIntercept = allowIntercept
  
    def define_model(self):
        """
        Define the model
        """
        coords = { self.datename: self.dates, "channels": self.channelnames }
        
        with pm.Model(coords=coords) as self.model:
            # variables
            spend = pm.Data('spend', self.data_scaled[self.channelnames].values, dims=(self.datename, "channels"))

            # Priors
            if self.allowIntercept:
                intercept = pm.Normal('intercept', mu=1, sigma=1)
                if 'intercept' not in self.fittedparmnames:
                    self.fittedparmnames = self.fittedparmnames + ['intercept']

            beta = pm.Normal('beta', mu=1, sigma=1, dims=("channels"))

            sigma = pm.HalfNormal('sigma')

            # Expected value
            mu_y = pm.Deterministic('mu_y', intercept + pm.math.dot(spend, beta), dims=(self.datename))

            # Likelihood
            y = pm.Normal('y', mu=mu_y, sigma=sigma, observed=self.data_scaled[self.salesname].values, dims=(self.datename))


import pandas as pd
from sklearn.preprocessing import MinMaxScaler #mypy: ignore
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns #mypy: ignore
import pymc as pm
import arviz as az
from IPython.display import display
from xarray import DataArray, Dataset



def get_distict_color(i, map_name='tab10'):
    """
    Get a distinct color for a given index
    Args:
        i (int): index
        map_name (str): colormap name
        
    Returns:
        str: color
    """
    cmap = plt.get_cmap(map_name, 10)         # Get the colormap and 10 colors from it
    rgba_color = cmap(i)                     # Get the ith color from the colormap
    hex_color = mcolors.rgb2hex(rgba_color)  # Convert the RGBA color to hexadecimal format
    return hex_color           

def line_plot(dataframe: pd.DataFrame, features: list, title: str, normalize: bool = True) -> None:
    """
    Plots multiple time series
    Args:
        dataframe (pd.DataFrame): data
        features (list): features to plot
        title (str): title for chart
    """

    colors = {0: '#070620', 1: '#dd4fe4'}
    
    plt.rcParams["figure.figsize"] = (20,3)
    for i in range(len(features)):
        if normalize:
            dataframe[[features[i]]] = MinMaxScaler().fit_transform(dataframe[[features[i]]])
        sns.lineplot(data=dataframe, x='ds', y=features[i], label=features[i], color=get_distict_color(i))
    plt.legend()
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.show()




class MyModel():
    def __init__(self, data, obsvar: str, timevar: str, features: list = []):
        self.data = data
        self.obsvar: str = obsvar
        self.timevar: str = timevar
        self.features: list[str] = features
        self.idata = None # The InferenceData object
        self.idata_inf: Dataset | None = None # The InferenceData object with prior and posterior predictive samples

        self.obsdata = self.data[self.obsvar]
        self.timedata = self.data[self.timevar]

        self.model: pm.Model = None


    def fit(self):
        with pm.Model() as model:
            self.model = model

            # control variables
            # tv = self.data["tv"].values

             # Define tv as a shared variable
            tv = pm.Data('tv', self.data["tv"], mutable=True)
            radio = pm.Data('radio', self.data["radio"], mutable=True)
            newspaper = pm.Data('newspaper', self.data["newspaper"], mutable=True)

            # PRIORS for unknown model parameters
            # intercept gets a prior here
            beta_i = pm.Normal("beta_i", mu=4, sigma=1, initval=1)

            # betas gets priors here
            beta_tv = pm.Normal("beta_tv", mu=0.1, sigma=0.1)
            beta_radio = pm.Normal("beta_radio", mu=0.1, sigma=0.1)
            beta_newspaper = pm.Normal("beta_newspaper", mu=0.1, sigma=0.1)

            # sigma for the model gets a prior here
            sigma = pm.HalfNormal("sigma", sigma=1)

            y_obs = pm.Normal("y_obs", mu=beta_i + beta_tv * tv + beta_radio * radio + beta_newspaper * newspaper, sigma=sigma, observed=self.obsdata)

            # Sample observed data points (time series) from the prior distribution 
            # idata_inf collects both the prior and posterior predictive samples and the posterior samples of parameters
            self.idata_inf = pm.sample_prior_predictive()
            # Sample the posterior distribution of parameters (not data observations) using Markov chains
            self.idata = pm.sample()
            # Add to the inference data object
            self.idata_inf.extend(self.idata)
            # Sample observed data points (time series) using the posterior distribution of parameters and the inference data object
            # will set self.idata_inf.posterior_predictive["y_obs"] to the posterior predictive samples
            pm.sample_posterior_predictive(self.idata_inf, extend_inferencedata=True)


    def line_plot_general(self, features: list, title: str, normalize: bool = True) -> None:
        line_plot(self.data.copy(), features, title, normalize)


    def plot_channel_spends(self, normalize: bool = False) -> None:
        """Plot all channel spends on same chart"""
        self.line_plot_general(self.features, 'Channel Spends', normalize)


    def plot_sales(self, normalize: bool = False) -> None:
        """Plot sales in original data"""
        self.line_plot_general([self.obsvar], 'Sales', normalize)


    def plot_channel_spends_vs_sales(self) -> None:
        """Plot channel spends vs sales for each channel and shown normalized"""
        for feature in self.features:
            self.line_plot_general([feature] + [self.obsvar], f'{feature} spend vs Sales')


    def plot_correlation_matrix(self) -> None:
        """Plot correlation matrix between channel spends and sales"""
        corr_matrix = self.data[self.features + [self.obsvar]].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='Blues')
        plt.show()


    def overall_numbers(self) -> dict:
        """Print overall numbers"""
        overall = {}
        overall['Sales Overall Mean'] = self.obsdata.mean()
        overall['Sales Overall Variance'] = self.obsdata.var()
        for feature in self.features:
            overall[f'{feature}: Overall Mean Spend '] = self.data[feature].mean()
            overall[f'{feature}: Overall total sales divided by total spend'] = self.obsdata.sum() / self.data[feature].sum()
        return overall
    

    def show_model_graphviz(self):
        """Show model graphviz representation"""
        if self.model is not None:
            graphviz = pm.model_to_graphviz(self.model)
            display(graphviz)
        else:
            raise ValueError("Model not fitted yet")


    def show_chain_traces(self):
        """Show the chain traces and the distributions of the parameters per chain"""
        if self.idata is not None:
            az.plot_trace(self.idata_inf)
        else:
            raise ValueError("Model not fitted yet")
        

    def show_prior_and_posterior_distribution(self):
        """Show the prior and posterior distribution of the parameters"""
        if self.idata is not None:
            az.plot_dist_comparison(self.idata_inf)
        else:
            raise ValueError("Model not fitted yet")
        

    def get_posterior_distribution_overview(self):
        """Gets an overview table of the properties of the posterior distribution of the parameters"""
        if self.idata is not None:
            return az.summary(self.idata, round_to=3)
        else:
            raise ValueError("Model not fitted yet")
        

    def plot_sales_vs_predicted(self):
        """Plots the observed sales vs the posterior predicted sales"""
        if self.idata is not None:
            obsdata_chains = self.idata_inf.posterior_predictive["y_obs"]
            obsdata_chains_stacked = obsdata_chains.stack(index=("chain", "draw"))
            # calculate the mean for each of the 200 timestamps
            obsdata_mean = obsdata_chains_stacked.mean(dim="index")

            fig, ax = plt.subplots()
            ax.plot(
                self.timedata,
                obsdata_mean,
                label="posterior predictive sales sampled",
                color="red")
            ax.plot(
                self.timedata,
                self.obsdata,
                label="original sales",
                color="blue")            
        else:
            raise ValueError("Model not fitted yet")
        

    def contribution_full_from_feature(self, feature: str):
        sales_org = self.data[self.obsvar].mean()
        spend = self.data[feature].mean()

        # original spends
        spend_vars = {feature: self.data[feature].copy() for feature in self.features}
        # remove the spend for the feature
        spend_vars[feature] = spend_vars[feature] * 0

        with self.model:
            pm.set_data(spend_vars)
            new_pp = pm.sample_posterior_predictive(self.idata_inf, progressbar=False)
            sales_new = new_pp.posterior_predictive["y_obs"].mean().item()
        
        contribution = sales_org - sales_new
        return {"spend": spend, "sales_org": sales_org, "sales_new": sales_new, "contribution": contribution, "full_roas": contribution / spend}

    def contribution_full(self):
        contributions = { feature: self.contribution_full_from_feature(feature) for feature in self.features}
        return pd.DataFrame.from_dict(contributions, orient='index')


    def test_contribution_full_plot(self):

        # get sales contribution over time from each feature and collect them to a dataframe
        contribs = {}
        contribs["sales_org"] = self.data[self.obsvar].copy()
        for feature in self.features:
            # set spend_var to 0 for the feature
            spend_vars = {feature: self.data[feature].copy() for feature in self.features}
            spend_vars[feature] = spend_vars[feature] * 0

            with self.model:
                pm.set_data(spend_vars)
                new_pp = pm.sample_posterior_predictive(self.idata_inf, progressbar=False)
                sales_contrib = new_pp.posterior_predictive["y_obs"]
                sales_contrib_mean_time = sales_contrib.mean(dim=["chain", "draw"])
                # add the sales without feature
                contribs[feature] = sales_contrib_mean_time

        # create a dataframe from the dict
        contribs_df = pd.DataFrame(contribs)

        # find actual contribution per feature
        for feature in self.features:
            contribs_df[feature] = contribs_df["sales_org"] - contribs_df[feature]
        
        # add base contribution
        contribs_df["base"] = contribs_df["sales_org"] - contribs_df[self.features].sum(axis=1)

        # find actual contribution per feature
        for feature in self.features:
            contribs_df[feature] = contribs_df["sales_org"] - contribs_df[feature]
        
        # make base the first column
        cols = contribs_df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        contribs_df = contribs_df[cols]


        # drop sales_org
        contribs_df.drop(columns=["sales_org"], inplace=True)

        # replace negative values with 0
        # TODO: This is ugly, find a better way
        contribs_df[contribs_df < 0] = 0

        # make a stacked area plot of the feature contributions
        contribs_df.plot.area(stacked=True, colormap="tab10", figsize=(20, 5))

        return contribs_df
    
    def test_contribution_full_plot_2(self):
        # get sales contribution over time from each feature and collect them to a dataframe

        contribs = {}
        contribs["sales_org"] = self.data[self.obsvar].copy()


        # Assuming `trace` is your posterior samples from the fitted model
        # and `time` is your time variable in a numpy array
        # `data` is a dictionary or DataFrame containing your marketing channels

        trace = self.idata
        data = self.data

        # Extracting posterior samples of coefficients (betas)
        beta_name_dict = {feature: f"beta_{feature}" for feature in self.features}
        beta_traces = {feature: trace.posterior[beta_name_dict[feature]] for feature in self.features}
        # beta_tv_samples = trace.posterior['beta_tv']  # Shape: (num_samples,)
        # beta_newspaper_samples = trace.posterior['beta_newspaper']  # Shape: (num_samples,)
        # beta_radio_samples = trace.posterior['beta_radio']  # Shape: (num_samples,)
        # channel_traces = { "feature"
        contribs_dict = { feature: (beta_traces[feature] * data[feature].to_xarray()).mean(dim=["chain", "draw"]) for feature in self.features}
        contribs_dict["sales_org"] = self.data[self.obsvar].copy()

        contribs_df = pd.DataFrame(contribs_dict)

        # add base contribution
        contribs_df["base"] = contribs_df["sales_org"] - contribs_df[self.features].sum(axis=1)

        # make base the first column
        cols = contribs_df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        contribs_df = contribs_df[cols]


        # drop sales_org
        contribs_df.drop(columns=["sales_org"], inplace=True)

        # replace negative values with 0
        # TODO: This is ugly, find a better way
        contribs_df[contribs_df < 0] = 0

        # make a stacked area plot of the feature contributions
        contribs_df.plot.area(stacked=True, colormap="tab10", figsize=(20, 5))

        return contribs_df        

        # Marketing channel spend data
        # tv = data['tv'].to_xarray()  # Assuming this is a numpy array or pandas Series
        # newspaper = data['newspaper']
        # radio = data['radio']

        # Compute contributions for each sample
        # tv_contributions = beta_tv_samples * tv  # Shape: (num_samples, num_time_points)
        # contribs["tv"] = tv_contributions

        # newspaper_contributions = beta_newspaper_samples[:, None] * newspaper
        # radio_contributions = beta_radio_samples[:, None] * radio

        

        # Summarize contributions at each time point
        # mean_tv_contribution = np.mean(tv_contributions, axis=0)
        # mean_newspaper_contribution = np.mean(newspaper_contributions, axis=0)
        # mean_radio_contribution = np.mean(radio_contributions, axis=0)

        # lower_tv_contribution = np.percentile(tv_contributions, 5, axis=0)
        # upper_tv_contribution = np.percentile(tv_contributions, 95, axis=0)

        # lower_newspaper_contribution = np.percentile(newspaper_contributions, 5, axis=0)
        # upper_newspaper_contribution = np.percentile(newspaper_contributions, 95, axis=0)

        # lower_radio_contribution = np.percentile(radio_contributions, 5, axis=0)
        # upper_radio_contribution = np.percentile(radio_contributions, 95, axis=0)

        # # Create a summary DataFrame
        # time_points = np.arange(len(tv))  # Assuming `time` is a sequence of time points
        # contribution_summary = pd.DataFrame({
        #     'Time': time_points,
        #     'Mean TV Contribution': mean_tv_contribution,
        #     'Lower TV Contribution (5%)': lower_tv_contribution,
        #     'Upper TV Contribution (95%)': upper_tv_contribution,
        #     'Mean Newspaper Contribution': mean_newspaper_contribution,
        #     'Lower Newspaper Contribution (5%)': lower_newspaper_contribution,
        #     'Upper Newspaper Contribution (95%)': upper_newspaper_contribution,
        #     'Mean Radio Contribution': mean_radio_contribution,
        #     'Lower Radio Contribution (5%)': lower_radio_contribution,
        #     'Upper Radio Contribution (95%)': upper_radio_contribution
        # })

        # import ace_tools as tools; tools.display_dataframe_to_user(name="Time-based Channel Contribution Summary", dataframe=contribution_summary)

        # contribution_summary
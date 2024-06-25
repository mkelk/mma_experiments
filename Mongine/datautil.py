#####################################################################
# Description: Utility functions for data manipulation
#####################################################################
import pandas as pd
import os

# Export from BQ via 
# 1. Do query (all rows)
# 1. Export to Drive
# 1. Download from Drive
# 1. Rename to `{date}-{BQtablename}.csv`
# 1. Import to VS Code

# Files for fermliving
# - `{date}_fermliving_googleads_campaign_stats.csv`
# - `{date}_fermliving_facebook_ad_stats.csv`
# - `{date}_fermliving_shopify_order_analytics.csv`

DATADIR = "data"

FILES = {
    'facebook': '2024-06-19_fermliving_facebook_ad_stats.csv',
    'google': '2024-06-19_fermliving_googleads_campaign_stats.csv',
    'shopify': '2024-06-19_fermliving_shopify_order_analytics.csv'
}

def read_csv(file):
    """
    Read a csv file and return a pandas dataframe
    """
    # get script filepath
    script_dir = os.path.dirname(__file__)

    filepath = os.path.join(script_dir, DATADIR, file)
    df = pd.read_csv(filepath)
    return df

def preprocess_fb_data(df):
    """
    Process the facebook data very simply
    """
    df['date'] = pd.to_datetime(df['date'])
    df['spend'] = pd.to_numeric(df['spend'])
    return df

def preprocess_google_data(df):
    """
    Process the google data very simply
    """
    df['date'] = pd.to_datetime(df['date'])
    df['cost'] = pd.to_numeric(df['cost'])
    return df

def preprocess_shopify_data(df):
    """
    Process the google data very simply
    """
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['order_total_value_DKK'] = pd.to_numeric(df['order_total_value_DKK'])
    return df

PREPROCESSORS = {
    'facebook': preprocess_fb_data,
    'google': preprocess_google_data,
    'shopify': preprocess_shopify_data
}

def get_loaded_file(filekey):
    """
    Read and preprocess a file by key, e.g. 'facebook'
    """
    filename = FILES[filekey]
    preprocessor = PREPROCESSORS[filekey]
    df = read_csv(filename)
    df = preprocessor(df)
    return df

def get_engineered_file(filekey):
    """
    Read, preprocess and enginee a file by key, e.g. 'facebook'
    """
    df = get_loaded_file(filekey)
    engineer = ENGINEERS[filekey]
    df = engineer(df)
    return df

def engineer_facebook_data(df_in):
    """
    Engineer the facebook data
    """
    df = df_in.copy()
    # now group by ad.id find the first date this ad.id was seen
    first_date = df.groupby('ad_id').agg({'date': 'min'}).reset_index()
    first_date.rename(columns={'date': 'ad_id_date_first'}, inplace=True)
    df = df.merge(first_date, on='ad_id', how='left')
    # add a column for days since first seen
    df['ad_id_days_since_first_seen'] = (df['date'] - df['ad_id_date_first']).dt.days



    return df

def engineer_google_data(df):
    """
    Engineer the google data
    """
    df["spend"] = df["cost"]
    return df   

def engineer_shopify_data(df):
    """
    Engineer the shopify data
    """
    return df

ENGINEERS = {
    'facebook': engineer_facebook_data,
    'google': engineer_google_data,
    'shopify': engineer_shopify_data
}

def get_collected_dataframe():
    """
    Collect all the dataframes into a single dataframe, using aggregated values per date
    """
    facebook = get_engineered_file('facebook')
    google = get_engineered_file('google')
    shopify = get_engineered_file('shopify')

    # facebook
    fb_df = facebook.groupby('date').agg({
            'spend': 'sum', 
            'impressions' : 'sum',
            'clicks' : 'sum',
            'purchases' : 'sum',
        }).reset_index()
    fb_df.rename(columns={'clicks': 'clicks_fb'}, inplace=True)
    fb_df.rename(columns={'impressions': 'impressions_fb'}, inplace=True)
    fb_df.rename(columns={'purchases': 'purchases_fb'}, inplace=True)
    fb_df.rename(columns={'spend': 'spend_fb'}, inplace=True)
    df_all = fb_df.copy()

    # google
    google_df = google.groupby('date').agg({
        'spend': 'sum',
        'impressions': 'sum',
        'clicks': 'sum',
        }).reset_index()
    google_df.rename(columns={'clicks': 'clicks_google'}, inplace=True)
    google_df.rename(columns={'impressions': 'impressions_google'}, inplace=True)
    google_df.rename(columns={'spend': 'spend_google'}, inplace=True)
    df_all = df_all.merge(google_df, on='date', how='outer')

    # shopify
    shopify_df = shopify.groupby('order_date').agg({'order_total_value_DKK': 'sum'}).reset_index()
    shopify_df["date"] = shopify_df["order_date"]
    shopify_df.drop(columns=['order_date'], inplace=True)
    shopify_df["sales"] = shopify_df["order_total_value_DKK"]
    shopify_df.drop(columns=['order_total_value_DKK'], inplace=True)

    df_all = df_all.merge(shopify_df, on='date', how='outer')

    # drop cols with google spend NaN
    df_all = df_all.dropna(subset=['spend_google'])

    return df_all

# make a main
if __name__ == "__main__":
    facebook = get_engineered_file('facebook')
    
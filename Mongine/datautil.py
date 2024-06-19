#####################################################################
# Description: Utility functions for data manipulation
#####################################################################
import pandas as pd
import os

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
    filepath = os.path.join(DATADIR, file)
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

def engineer_facebook_data(df):
    """
    Engineer the facebook data
    """
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
    Collect all the dataframes into a single dataframe
    """
    facebook = get_engineered_file('facebook')
    google = get_engineered_file('google')
    shopify = get_engineered_file('shopify')

    # facebook
    fb_df = facebook.groupby('date').agg({'spend': 'sum'}).reset_index()
    df_all = fb_df.copy()
    df_all["spend_fb"] = df_all["spend"]
    df_all.drop(columns=['spend'], inplace=True)

    # google
    google_df = google.groupby('date').agg({'spend': 'sum'}).reset_index()
    google_df["spend_google"] = google_df["spend"]
    google_df.drop(columns=['spend'], inplace=True)
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

    # Merge the dataframes
    df = facebook.merge(google, on='date', how='outer')
    df = df.merge(shopify, left_on='date', right_on='order_date', how='outer')

    return df
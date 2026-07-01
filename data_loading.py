import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import torch

def create_lagged_dataset(df, feature_cols, target_col='covid_cases', window_size=30):
    """
    Converts a time-series dataframe into samples with lag features for supervised learning.
    """
    X, y = [], []
    for i in range(len(df) - window_size):
        window = df.iloc[i:i+window_size]
        X.append(window[feature_cols].values)
        y.append(df.iloc[i + window_size][target_col])
    
    X = np.array(X)  # shape: (samples, 30, num_features)
    y = np.array(y)  # shape: (samples,)
    return X, y



# ----------------------------------------- to create the dataframe ---------------------------------------- #
def load_GCMR():    
    # Download file first from https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv
    gcmr_path = '/home/barbpt/Desktop/ML_course/Final_project/' # downloaded

    # Create an empty list to hold filtered chunks
    ## we are using chunks to only load relevant data, since the total data is too large
    chunks = []

    # Read in chunks and filter
    for chunk in pd.read_csv(f'{gcmr_path}Global_Mobility_Report.csv', chunksize=100_000,low_memory=False):
        # Keep rows where country_region == 'United States' and sub_region_1 == 'New York'
        ny_chunk = chunk[(chunk['country_region'] == 'United States') & 
                         (chunk['sub_region_1'] == 'New York')]
        if not ny_chunk.empty:
            chunks.append(ny_chunk)

    # Concatenate the filtered chunks
    df_ny = pd.concat(chunks, ignore_index=True)

    ## create daily mobility index: Mobility index as the average (mean) of percentage change from baseline
    df_ny['date'] = pd.to_datetime(df_ny['date'])
    mobility_cols = ['workplaces_percent_change_from_baseline',
                     'transit_stations_percent_change_from_baseline',
                     'grocery_and_pharmacy_percent_change_from_baseline',
                     'retail_and_recreation_percent_change_from_baseline']
    df_ny['mobility_index'] = df_ny[mobility_cols].mean(axis=1)
    df_mobility = df_ny[['date', 'mobility_index']].dropna()
    df_mobility['date'] = pd.to_datetime(df_mobility['date'])

    return df_mobility

def load_covid_cases():
    ''' 29.02.2020 - today
    '''
    # Daily NYC COVID case data from GitHub
    url_cases = 'https://raw.githubusercontent.com/nychealth/coronavirus-data/master/trends/data-by-day.csv'
    df_cases = pd.read_csv(url_cases)
    # print(df_cases.columns)

    df_cases['date'] = pd.to_datetime(df_cases['date_of_interest']) #.dt.date
    df_cases = df_cases[['date', 'CASE_COUNT']].rename(columns={'CASE_COUNT': 'covid_cases'})
    return df_cases

def load_air_quality():
    # Download datasets before from: https://aqs.epa.gov/aqsweb/airdata/download_files.html
    df_ozone_2020 = pd.read_csv('./daily_44201_2020.csv',low_memory=False)
    df_ozone_2021 = pd.read_csv('./daily_44201_2021.csv',low_memory=False)
    df_ozone_2022 = pd.read_csv('./daily_44201_2022.csv',low_memory=False)

    df_pm_2020 = pd.read_csv('./daily_88101_2020.csv')
    df_pm_2021 = pd.read_csv('./daily_88101_2021.csv')
    df_pm_2022 = pd.read_csv('./daily_88101_2022.csv')
    
    df_ozone_all = pd.concat([df_ozone_2020, df_ozone_2021, df_ozone_2022])
    df_pm_all = pd.concat([df_pm_2020, df_pm_2021, df_pm_2022])

    date_range = pd.date_range(start='2020-02-29', end='2022-10-15')
    df_empty = pd.DataFrame({'date': date_range})

    df_aqi = merge_aqi_data(df_empty, df_pm_all, pollutant='PM2.5')
    df_aqi = merge_aqi_data(df_aqi, df_ozone_all, pollutant='Ozone')
    
    return df_aqi

def merge_aqi_data(df_main, df_aqi, pollutant='PM2.5'):
    """
    Merges EPA air quality data into your main dataset by date.
    
    Parameters:
    - df_main: DataFrame with existing features, must have a 'date' column (datetime.date)
    - df_aqi: EPA air quality DataFrame with 'Date Local', 'City Name', and either 'AQI' or 'Arithmetic Mean'
    - pollutant: Optional label for naming the column
    
    Returns:
    - Merged DataFrame with new AQI column
    """
    # 1. Convert EPA date column to datetime.date
    df_aqi = df_aqi.copy()
    df_aqi['date'] = pd.to_datetime(df_aqi['Date Local']) #.dt.date

    # 2. Optional: Filter to just NYC if needed
    df_aqi = df_aqi[df_aqi['City Name'].str.lower() == 'new york']

    # 3. Take daily average (if multiple sites)
    if 'AQI' in df_aqi.columns:
        df_aqi_clean = df_aqi.groupby('date')['AQI'].mean().reset_index()
        df_aqi_clean = df_aqi_clean.rename(columns={'AQI': f'{pollutant}_AQI'})
    elif 'Arithmetic Mean' in df_aqi.columns:
        df_aqi_clean = df_aqi.groupby('date')['Arithmetic Mean'].mean().reset_index()
        df_aqi_clean = df_aqi_clean.rename(columns={'Arithmetic Mean': f'{pollutant}_mean'})
    else:
        raise ValueError("Expected 'AQI' or 'Arithmetic Mean' column not found.")

    # 4. Merge into main dataset
    df_merged = pd.merge(df_main, df_aqi_clean, on='date', how='left')

    return df_merged

def load_weather():
    ## New York coordinates
    latitude = 40.7128
    longitude = -74.0060

    # Date range
    start_date = '2020-02-29'
    end_date = '2022-10-15'


    # Request weather data
    url = 'https://archive-api.open-meteo.com/v1/archive'
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_date,
        'end_date': end_date,
        'daily': ['temperature_2m_max', 'temperature_2m_min', 'relative_humidity_2m_mean'],
        'timezone': 'auto'
    }
    weather = requests.get(url, params=params).json()
    df_weather = pd.DataFrame({
        'date': weather['daily']['time'],
        'temp_max': weather['daily']['temperature_2m_max'],
        'temp_min': weather['daily']['temperature_2m_min'],
        'humidity': weather['daily']['relative_humidity_2m_mean']
    })
    df_weather['date'] = pd.to_datetime(df_weather['date']) #.dt.date
    df_weather.head()

    return df_weather
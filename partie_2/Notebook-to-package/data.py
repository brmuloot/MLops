import pandas as pd

# implement get_data() function
def get_data(nrows=10000):
    '''returns a DataFrame with nrows from s3 bucket'''
    df = pd.read_csv("../Data/train.csv", nrows = nrows)
    return df

# implement clean_data() function
def clean_data(df, test=False):
    '''returns a DataFrame without outliers and missing values'''
    df = df[df.fare_amount >= 0]
    df = df[df.distance <= 100]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count > 0]
    return df
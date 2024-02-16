import pandas as pd
import matplotlib.pyplot as plt

def clean_data(df, fillna_method):
    # Filling gaps in the data
    if fillna_method in ['time', 'linear', 'quadratic', 'cubic']:
        df.interpolate(method=fillna_method, inplace=True)
    else:
        df.fillna(method=fillna_method, inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Remove outliers
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~( (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)) ).any(axis=1)]

    # Reset indexes
    df.reset_index(drop=False, inplace=True)


file_path = './bitstampUSD.csv'
df = pd.read_csv(file_path)

print(df.head())
print(df.tail())
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')  
df = df[df['Timestamp'].dt.year >= 2021]

df = df.set_index('Timestamp')

# Uncomment to check the data


# df['Low'].plot()
# plt.show()

# print(f"dataframe contains {df['Low'].isnull().sum()} missing values")
# print(f"dataframe contains {df.duplicated().sum()} duplicate values")


clean_data(df, 'time')
print(df.head())

# print(df.head())
# print(f"dataframe contains {df['Low'].isnull().sum()} missing values")
# print(f"dataframe contains {df.duplicated().sum()} duplicate values")

# df['Low'].plot()
# plt.show()


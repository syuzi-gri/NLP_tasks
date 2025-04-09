import pandas as pd
import numpy as np

df = pd.read_csv("oil.csv")

# df['Weekday'] = df['Date'].str.split().str[0] 
weekly_analysis = df.groupby("Weekday").agg({
    "Open": np.mean,
    "High": np.mean,
    "Low": np.mean,
    "Close": np.mean,
    "Vol": np.sum
})
# print("Weekly Analysis:\n", weekly_analysis)

yearly_analysis = df.groupby("Year").agg({
    "Open": np.mean,
    "High": np.mean,
    "Low": np.mean,
    "Close": np.mean,
    "Vol": np.sum
})
# print("Yearly Analysis:\n", yearly_analysis)


monthly_analysis = df.groupby("Month").agg({
    "Open": np.mean,
    "High": np.mean,
    "Low": np.mean,
    "Close": np.mean,
    "Vol": np.sum
}).reset_index()
# print("Monthlyly Analysis:\n", monthly_analysis)


numeric_columns = ['Open', 'High', 'Low', 'Close', 'Vol']

statistics = df[numeric_columns].agg([np.mean, np.median])
# df[numeric_columns] = df[numeric_columns].fillna(0)
# print(df[numeric_columns].dtypes)

print(statistics)



# new features

# add a new column
df['Price_Difference'] = df['High'] - df['Low']
print(df.head())


#price change
df['Price_Change'] = df['Close'] - df['Open']
# file-i mej add anel
# df.to_csv('oil.csv', index=False)
max_price_change = df['Price_Change'].max()
# ete toxy gtnenq
max_price_change_row = df.loc[df['Price_Change'].idxmax()]
print("Date ", max_price_change_row['Date'], "price ", max_price_change_row['Price_Change'])
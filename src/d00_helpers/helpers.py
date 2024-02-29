# Function to replace missing values with median date
import pandas as pd

def replace_with_median(df, column_name):
    # Convert datetime to numeric to find median
    not_null_series = df[column_name].dropna().apply(lambda x: x.value)
    median = not_null_series.median()

    # Convert numeric median back to datetime
    median_date = pd.to_datetime(median, unit='ns')

    # Replace missing values with median date
    df[column_name].fillna(median_date, inplace=True)

    return df


def main():
    pass


if __name__ == '__main__':
    main()
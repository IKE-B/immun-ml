# Function to replace missing values with median date
import datetime
import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None


def replace_with_median(df, column_name):
    # Convert datetime to numeric to find median
    not_null_series = df[column_name].dropna().apply(lambda x: x.value)
    median = not_null_series.median()

    # Convert numeric median back to datetime
    median_date = pd.to_datetime(median, unit='ns')

    # Replace missing values with median date
    df[column_name].fillna(median_date, inplace=True)

    return df

def find_future_measurement_date(group):
    future_dates = []  # List to store the result for each row
    for i, row in group.iterrows():
        # Find dates between 350 to 370 days ahead
        min_date = row['date'] + pd.Timedelta(days=350)
        max_date = row['date'] + pd.Timedelta(days=370)
        future_measurements = group[(group['date'] >= min_date) & (group['date'] <= max_date)]

        # Check if there's any such measurement and act accordingly
        if not future_measurements.empty:
            future_dates.append(future_measurements.iloc[0]['date'].strftime('%Y-%m-%d'))
        else:
            future_dates.append(pd.NA)

    return pd.Series(future_dates, index=group.index)


def create_T_date_mapping(abs_path_to_codebook):
    cb = pd.read_excel(abs_path_to_codebook, sheet_name='TimePoints').iloc[:25]

    res_dic = {}
    for vacc_number in cb['relation_to_number_of_vaccination'].unique():
        cb_filt = cb[cb['relation_to_number_of_vaccination'] == vacc_number]
        res_dic[vacc_number] = dict(zip(cb_filt['coding'],
                                        cb_filt['relative_timediff'].astype(int)
                                        )
                                    )
    return res_dic


def calculate_measuring_date_and_create_long_format(df, t_date_mapping, t_date_vaccs, patient_id, measurement_column):
    """
    Enhances a DataFrame by calculating measurement dates for a specified patient and measurement column,
    incorporating vaccination and infection dates, then creates a long-format DataFrame with this enriched information.

    This function:
    - Filters the original DataFrame for the specified patient.
    - Identifies measurement columns containing the specified measurement string.
    - Calculates measurement dates using vaccination dates, infection dates, and a mapping of time differences.
    - Compiles the enriched data into a new DataFrame with standardized columns, including information about vaccinations and infections.

    Parameters:
    - df (pd.DataFrame): The original DataFrame containing patient data.
    - t_date_mapping (dict): A mapping from vaccination numbers to dictionaries mapping timepoints to days after vaccination.
    - t_date_vaccs (dict): Maps timepoint strings to vaccination numbers.
    - patient_id (str/int): The patient's unique identifier.
    - measurement_column (str): The string identifying relevant measurement columns in the DataFrame.

    Returns:
    - pd.DataFrame: A DataFrame with columns ['patient_id', 'timepoint', measurement_column, 'date', 'vaccination', 'infection'],
                    containing measurements, corresponding timepoints, calculated dates, and flags for vaccination and infection events.

    The function attempts to calculate a measurement date for each relevant column for the patient, marking dates as 'NA' and excluding measurements if a date cannot be calculated or if the measurement is NaN. It includes additional columns to flag vaccination and infection dates, providing a comprehensive view of the patient's medical timeline.

    Note:
    - 'vaccination' column indicates the presence of a vaccination event on the date with a 1, and 0 otherwise.
    - 'infection' column indicates the presence of an infection event on the date with a 1, and 0 otherwise.
    - The function sorts the resulting DataFrame by date and ensures there are no duplicate dates, keeping the last record for any given date.
    """

    cols = ['patient_id', 'timepoint', measurement_column, 'date', 'vaccination', 'infection']
    res = pd.DataFrame(columns=cols)

    measurement_columns = [col for col in df.columns if measurement_column in col]

    df_filt = df[(df['ID'] == patient_id)]

    infection_dates = df_filt[['1._Infektion', '2._Infektion']].iloc[0].dropna().tolist()
    if infection_dates:
        infection_dates = [pd.to_datetime(d).date().strftime('%Y-%m-%d') for d in infection_dates]
        for d in infection_dates:
            res_dic = {'patient_id': patient_id, 'date': d, 'infection': 1}
            res = pd.concat([res, pd.DataFrame([res_dic])], ignore_index=True)

    vaccination_dates = sorted(df_filt[[f'{i}._Impfung_DATUM' for i in range(1, 6, 1)]].iloc[0, :].dropna().tolist())
    vaccination_dates = [pd.to_datetime(d).date().strftime('%Y-%m-%d') for d in vaccination_dates]

    for d in vaccination_dates:
        res_dic = {'patient_id': patient_id, 'date': d, 'vaccination': 1}
        res = pd.concat([res, pd.DataFrame([res_dic])], ignore_index=True)

    for m_col in measurement_columns:
        t = m_col.split('_')[-1]
        v = int(t_date_vaccs[t])
        diff = t_date_mapping[v][t]
        vacc_date = df_filt[f'{v}._Impfung_DATUM']
        try:
            m_date = vacc_date + datetime.timedelta(days=diff)
            m_date = m_date.iloc[0].date().strftime('%Y-%m-%d')
        except:
            m_date = np.datetime64("NaT")

        if not math.isnan(df_filt[m_col].values[0]):
            res_dic = {'patient_id': patient_id,
                       'timepoint': t,
                       measurement_column: round(df_filt[m_col].values[0], 2),
                       'date': m_date}
            res = pd.concat([res.dropna(axis='columns', how='all'),
                             pd.DataFrame([res_dic]).dropna(axis='columns', how='all')],
                            ignore_index=True)

    try:
        res = res.sort_values(by='date').groupby('date').agg('first').reset_index()

    except:
        print('Patient without any date: ', patient_id)





    return res


def plot_sars_igg_with_events(df, results_path, bl_s, show_plot=False, save_fig=False):
    """
    Plots SARS-IgG values over time with markers for vaccinations and infections.

    Parameters:
    - df: A pandas DataFrame with columns 'date', 'vaccination', 'infection', and 'SARS-IgG'.
    """

    # Convert 'date' column to datetime if it's not already
    df.loc[:, 'date'] = pd.to_datetime(df['date'])

    # Setting the plot size
    plt.figure(figsize=(14, 7))

    # Plotting SARS-IgG values over time
    ax = sns.lineplot(data=df, x='date', y='SARS-IgG', marker='o', label='SARS-IgG', linestyle='-', color='blue')

    # just for plotting reasons so the markers know their "height" on y-axis
    df.loc[:, 'SARS-IgG'] = df.loc[:, 'SARS-IgG'].fillna(
        (df['SARS-IgG'].fillna(method='ffill') + df['SARS-IgG'].fillna(method='bfill')) / 2)

    # Highlighting vaccination dates
    vaccination_dates = df[df['vaccination'] == 1]
    plt.scatter(vaccination_dates['date'], vaccination_dates['SARS-IgG'], color='green', marker='^', s=100,
                label='Vaccination')

    # Highlighting infection dates
    infection_dates = df[df['infection'] == 1]
    plt.scatter(infection_dates['date'], infection_dates['SARS-IgG'], color='red', marker='x', s=100, label='Infection')

    # Enhancing the plot
    plt.xlabel('Date')
    plt.ylabel('SARS-IgG Value')
    plt.title(
        f"SARS-IgG Values Over Time With Vaccination and Infection Events \n Age = {bl_s.Alter.values[0]}, Sex = {'male' if bl_s.Geschlecht.values[0] == 0 else 'female'}, ID={bl_s.ID.values[0]}")
    plt.legend()
    plt.grid(True)

    # Custom x-ticks for vaccination and infection dates

    df.loc[:, 'kind'] = pd.Series(dtype='object')
    df.loc[df.index[df['vaccination'] == 1].tolist(), 'kind'] = 'vaccination'
    df.loc[df.index[df['infection'] == 1].tolist(), 'kind'] = 'infection'
    filt = (df['infection'] == 0) & (df['vaccination'] == 0)
    df.loc[filt, 'kind'] = 'measurement'

    event_dates = df['date']
    ax.set_xticks(event_dates)
    ax.set_xticklabels(event_dates.dt.strftime('%Y-%m-%d'))

    # Color x-tick labels based on the event type
    for tick_label, kind in zip(ax.get_xticklabels(), df['kind']):
        if kind == 'infection':
            tick_label.set_color('red')
        elif kind == 'vaccination':
            tick_label.set_color('green')
        else:
            pass

    # Improving the x-axis date formatting
    plt.gcf().autofmt_xdate(rotation=90, ha='center')

    id = df.ID.unique().tolist()[0]
    if save_fig:
        plt.savefig(f'{results_path}/{id}.svg')

    if show_plot:
        plt.show()

    plt.close()

    return None


def calculate_days_since_last_event(df, date_col='date', event_col='vaccination'):
    """
    Calculates the days since the last event for each date in the DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - date_col: The name of the column containing the date information.
    - vaccinated_col: The name of the binary column indicating whether a event happened (1 for yes, 0 for no).

    Returns:
    - DataFrame with an additional column 'days_since_last_event' showing the days since the last event.
    """
    # Ensure the DataFrame is sorted by date
    df_sorted = df.sort_values(by=date_col).copy()

    # Initialize the last event date
    df_sorted[f'last_{event_col}_date'] = pd.NaT  # Use NaT for consistency with datetime types

    # Update 'last_event_date' for vaccinated entries
    last_event_date = pd.NaT
    for index, row in df_sorted.iterrows():
        # Check if the row indicates event happened, explicitly handling NA values
        if not pd.isna(row[event_col]) and row[event_col] == 1:
            last_event_date = row[date_col]
        df_sorted.at[index, f'last_{event_col}_date'] = last_event_date

    # Calculate the days since the last event
    df_sorted[f'days_since_last_{event_col}'] = (df_sorted[date_col] - df_sorted[f'last_{event_col}_date']).dt.days

    # Handle dates before the first event
    df_sorted[f'days_since_last_{event_col}'] = df_sorted[f'days_since_last_{event_col}'].fillna(-1).astype(int)
    df_sorted.loc[df_sorted[f'days_since_last_{event_col}'] < 0, f'days_since_last_{event_col}'] = -1

    # Drop the temporary column
    df_sorted.drop(columns=[f'last_{event_col}_date'], inplace=True)

    return df_sorted

def calc_average_sarsigg_increase(sub_df, infection_date):
    # Ensure 'date' column is in datetime format
    sub_df['date'] = pd.to_datetime(sub_df['date'])

    # Find the index for the infection date
    infection_idx = sub_df[sub_df['date'] == pd.to_datetime(infection_date)].index
    infection_idx = infection_idx[0]

    if sub_df.index.get_loc(infection_idx)==0 or infection_idx == sub_df.index[-1]:
        return None  # Infectiondate is last or first entry of sub_dataframe


    # Initialize variables
    prev_sars_igg_value = None
    next_sars_igg_value = None
    prev_date = None
    next_date = None

    # Look backward for the last SARS-IgG measurement before the infection
    for i in range(infection_idx, -1, -1):
        if not pd.isna(sub_df.loc[i, 'SARS-IgG']):
            prev_sars_igg_value = sub_df.loc[i, 'SARS-IgG']
            prev_date = sub_df.loc[i, 'date']
            break

    # Look forward for the first SARS-IgG measurement after the infection without vaccination in between
    idx = infection_idx
    while not np.isnan(sub_df.loc[idx+1, 'SARS-IgG']):
        if sub_df.loc[idx+1, 'vaccination'] == 1:
            return None  # Vaccination occurred after infection before next SARS-IgG measurement

        if not pd.isna(sub_df.loc[idx+1, 'SARS-IgG']):
            next_sars_igg_value = sub_df.loc[idx+1, 'SARS-IgG']
            next_date = sub_df.loc[idx+1, 'date']
            break
        idx += 1

    # Ensure we have all needed data
    if prev_sars_igg_value is None or next_sars_igg_value is None:
        return None  # Required measurements not found

    # Calculate the differences
    days_diff = (next_date - prev_date).days
    sars_diff = next_sars_igg_value - prev_sars_igg_value
    slope = sars_diff / days_diff if days_diff > 0 else None

    return {'sars_diff': sars_diff, 'slope': slope}


def main():
    global cb

    """ Test calculate_measuring_date_and_create_long_format
    abs_path_to_codebook = 'G://My Drive//Forschung//Mitarbeiter/Allgaier//23-12-06_Immun-ML//01_Codebook//Codebook.xlsx'
    abs_path_to_df = 'G://My Drive//Forschung//Mitarbeiter/Allgaier//23-12-06_Immun-ML//04_Data//00_raw//2024.03.21_Mastertabelle.xlsx'

    cb = pd.read_excel(abs_path_to_codebook, sheet_name='TimePoints').iloc[:25]

    t_date_vaccs = dict(zip(cb['coding'], cb['relation_to_number_of_vaccination']))
    t_date_mapping = create_T_date_mapping(abs_path_to_codebook)

    df = pd.read_excel(abs_path_to_df, na_values=[' ', ''])
    patient_id = 'HD19'
    measurement_column = 'SARS-IgG'  # without _T1, _T2, ...

    res = calculate_measuring_date_and_create_long_format(df, t_date_mapping, t_date_vaccs, patient_id,
                                                          measurement_column)
                                                          
    print(res.shape)
    """

    # test plot_sars_igg_with_events
    df = pd.read_excel(
        'G://My Drive//Forschung//Mitarbeiter//Allgaier//23-12-06_Immun-ML//04_Data//01_processed//2024.03.21_Mastertabelle_long.xlsx',
        parse_dates=['date'], index_col='Unnamed: 0', dtype={'vaccination': "Int64", 'infection': 'Int64'})
    df.rename(columns={'patient_id': 'ID'}, inplace=True)

    bl = pd.read_excel(
        'G://My Drive//Forschung//Mitarbeiter//Allgaier//23-12-06_Immun-ML//04_Data//00_raw//2024.03.21_Mastertabelle.xlsx')
    bl_sub = bl[['ID', 'Alter', 'Dialyse', 'Geschlecht']]

    """
    # loop over all patients
    results_path = '../../results/rq1/sars-igg-plots-by-patient'
    for id in df.ID.unique()[:2]:
        bl_s = bl_sub[bl_sub['ID'] == id]
        df_s = df[df['ID'] == id]
        plot_sars_igg_with_events(df_s, results_path, bl_s)
    """
    sub_df = df[df['ID']=='HD75']
    # timepoint where infection happened
    infection_dates = list(sub_df[sub_df['infection'] == 1]['date'])
    infection_date = infection_dates[0]

    res_dic = calc_average_sarsigg_increase(sub_df, infection_date)

if __name__ == '__main__':
    main()

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

    if infection_dates:
        res['infection'] = res['infection'].fillna(0)
    else:
        res['infection'] = None
        res['infection'] = res['infection'].fillna(0)

    if vaccination_dates:
        res['vaccination'] = res['vaccination'].fillna(0)

    try:
        res = res.sort_values(by='date').groupby('date').agg('last').reset_index()
    except:
        print('Patient without any date: ', patient_id)
    return res


def plot_sars_igg_with_events(df, results_path, bl_s, show_plot=False):
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
    plt.savefig(f'{results_path}/{id}.svg')

    if show_plot:
        plt.show()

    plt.close()

    return None


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

    # loop over all patients
    results_path = '../../results/rq1/sars-igg-plots-by-patient'
    for id in df.ID.unique()[:2]:
        bl_s = bl_sub[bl_sub['ID'] == id]
        df_s = df[df['ID'] == id]
        plot_sars_igg_with_events(df_s, results_path, bl_s)


if __name__ == '__main__':
    main()

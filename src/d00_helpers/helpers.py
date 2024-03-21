# Function to replace missing values with median date
import pandas as pd
import datetime
import math

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
        cb_filt = cb[cb['relation_to_number_of_vaccination']==vacc_number]
        res_dic[vacc_number] = dict(zip(cb_filt['coding'],
                                        cb_filt['relative_timediff'].astype(int)
                                        )
                                    )
    return res_dic


def calculate_measuring_date_and_create_long_format(df, t_date_mapping, t_date_vaccs, patient_id, measurement_column):
    """
    Calculates the measurement dates for a given patient and measurement column,
    then creates a long-format DataFrame containing the measurement information.

    The function filters the original DataFrame for the specified patient,
    iterates through the measurement columns that contain the specified measurement column string,
    calculates the measurement dates based on vaccination dates and a mapping of time differences,
    and finally compiles the results into a new DataFrame with standardized columns.

    Parameters:
    - df (pd.DataFrame): The original DataFrame containing patient data.
    - t_date_mapping (dict): A dictionary mapping vaccination numbers to another dictionary,
                             which maps timepoints to days after vaccination.
    - t_date_vaccs (dict): A dictionary mapping timepoint strings to vaccination numbers.
    - patient_id (str/int): The unique identifier for the patient.
    - measurement_column (str): The string to identify relevant measurement columns in the DataFrame.

    Returns:
    - pd.DataFrame: A DataFrame with columns ['patient_id', 'timepoint', measurement_column, 'date'],
                    containing the patient's measurements, the corresponding timepoints,
                    and the calculated measurement dates.

    The function attempts to calculate a measurement date for each measurement column associated
    with the specified measurement type for the patient. If a date cannot be calculated or if the
    measurement is NaN, the function will mark the date as 'NA' and exclude the measurement from the results.
    """
    res = pd.DataFrame(columns=['patient_id', 'timepoint', measurement_column, 'date'])

    measurement_columns = [col for col in df.columns if measurement_column in col]

    df_filt = df[(df['ID'] == patient_id)]

    for m_col in measurement_columns:
        t = m_col.split('_')[-1]
        v = int(t_date_vaccs[t])
        diff = t_date_mapping[v][t]
        try:
            m_date = df_filt[f'{v}._Impfung_DATUM'] + datetime.timedelta(days=diff)
            m_date = m_date.loc[0].date().strftime('%Y-%m-%d')
        except:
            m_date = 'NA'
        if not math.isnan(df_filt[m_col].values[0]):
            res_dic = {'patient_id': patient_id,
                       'timepoint': t,
                       measurement_column: round(df_filt[m_col].values[0], 2),
                       'date': m_date}
            res = pd.concat([res, pd.DataFrame([res_dic])], ignore_index=True)

    return res


def main():
    global cb

    abs_path_to_codebook = 'G://My Drive//Forschung//Mitarbeiter/Allgaier//23-12-06_Immun-ML//01_Codebook//Codebook.xlsx'
    abs_path_to_df = 'G://My Drive//Forschung//Mitarbeiter/Allgaier//23-12-06_Immun-ML//04_Data//00_raw//2024.03.21_Mastertabelle.xlsx'

    cb = pd.read_excel(abs_path_to_codebook, sheet_name='TimePoints').iloc[:25]


    t_date_vaccs = dict(zip(cb['coding'], cb['relation_to_number_of_vaccination']))
    t_date_mapping = create_T_date_mapping(abs_path_to_codebook)

    df = pd.read_excel(abs_path_to_df)
    patient_id = 'HD1'
    measurement_column = 'SARS-IgG' # without _T1, _T2, ...

    res = calculate_measuring_date_and_create_long_format(df, t_date_mapping, t_date_vaccs, patient_id, measurement_column)

    print(res.shape)

if __name__ == '__main__':
    main()
import pandas as pd
import os
import numpy as np


class LoadData:
    def __init__(self, business_pattern_path, naics_path, county_information_path, occupation_path, occupation_prioritization_path, indicator_path, output_dir='data/processed_data/pkl/'):
        """
        Initializes the LoadData class with paths to various data files.

        Parameters:
        - business_pattern_path (str): Path to the Business Pattern data file.
        - naics_path (str): Path to the NAICS code file.
        - county_information_path (str): Path to the County information pickle file.
        - occupation_path (str): Path to the Occupation Master data file.
        - indicator_path (str): Path to the Indicator Master data file.
        - output_dir (str): Directory to save the processed Pickle files.
        """
        self.business_pattern_path = business_pattern_path
        self.naics_path = naics_path
        self.county_information_path = county_information_path
        self.occupation_path = occupation_path
        self.occupation_prioritization_path = occupation_prioritization_path
        self.indicator_path = indicator_path
        self.output_dir = output_dir

    def load_business_pattern_data(self):
        """
        Loads the Business Pattern data and NAICS codes, merges them, and calculates GEOIDs.

        Returns:
        - tuple:
            - pd.DataFrame: The processed DataFrame with GEOID calculations.
            - pd.DataFrame: The County information DataFrame.
        """
        df = pd.read_csv(self.business_pattern_path, sep=',', encoding='latin-1')
        df_naics = pd.read_csv(self.naics_path, sep=',').rename(columns={'NAICS': 'naics'})
        df_county = pd.read_pickle(self.county_information_path)

        # Merge and GEOID Calculations
        df_pattern = (df.merge(df_naics, on='naics', how='left')
                      .assign(
                          State_GEOID=lambda x: x['fipstate'].astype(str).str.zfill(2),
                          County_GEOID=lambda x: x['fipscty'].astype(str).str.zfill(3),
                          GEOID=lambda x: x['State_GEOID'].str.cat(x['County_GEOID']),
                          naics_2=lambda x: x['naics'].str[:2]
                      )
                      .rename(columns={'GEOID': 'FIPS'})
                      .astype({'FIPS': int}))

        # Save the processed data to Pickle
        df_pattern.to_pickle(f'{self.output_dir}df_pattern.pickle')
        df_county.to_pickle(f'{self.output_dir}df_county.pickle')

        return df_pattern, df_county

    def load_and_prepare_occupation_data(self):
        """
        Loads the Occupation data and Indicator Master files, prepares and filters them by priority codes.

        Parameters:
        - file_path (str): Path to the Occupation data file (Excel).
        - occupation_path (str): Path to the Occupation Master file (Excel).
        - indicator_path (str): Path to the Indicator Master file (Excel).
        
        Returns:
        - tuple:
          - pd.DataFrame: The filtered and prepared DataFrame with priority data.
          - pd.Series: The filtered Occupation codes.
        """
        df = pd.read_excel(self.occupation_path)

        # Data Preparation
        df['AREA'] = df['AREA'].astype(str)
        df['TOT_EMP'] = pd.to_numeric(df['TOT_EMP'].replace('++', pd.NA), errors='coerce').astype('Int64')

        # Load Master Data
        occupation_master = pd.read_excel(self.occupation_prioritization_path)
        indicator_master = pd.read_excel(self.indicator_path)

        # Filter for Priority Codes and Indicators
        filtered_occ_codes = occupation_master.loc[occupation_master['prio'] == 'x', 'OCC_CODE']
        filtered_indicators = indicator_master.loc[indicator_master['prio'] == 'x', 'Field']

        # Filter and prepare the final DataFrame
        df['NAICS 2 digit'] = df['NAICS'].str[:2]
        #prio_df = df[df['OCC_CODE'].isin(filtered_occ_codes)][filtered_indicators]
        df_occupation = df[filtered_indicators]

        # Save the DataFrames as Pickle files
        #prio_df.to_pickle(f'{self.output_dir}prio_df.pickle')
        df_occupation.to_pickle(f'{self.output_dir}df_occupation.pickle')
        filtered_occ_codes.to_pickle(f'{self.output_dir}filtered_occ_codes.pickle')

        return df_occupation, filtered_occ_codes

class DataProcessing:
    @staticmethod
    def convert_naics_codes(df_pattern, df_occupation):
        """
        Convert 6-digit NAICS codes to 4-digit by standardizing and aggregating specific codes,
        and create overview DataFrames for NAICS and occupations.
        
        Parameters:
        df_pattern (DataFrame): DataFrame with NAICS codes and descriptions.
        df_occupation (DataFrame): DataFrame with occupation codes and titles.

        Returns:
        tuple: DataFrames with updated 4-digit NAICS codes and occupation overviews.
        """
        # Convert 6-digit NAICS codes to 4-digit
        df_pattern_4d = df_pattern[df_pattern['naics'].str.endswith('//') & (df_pattern['naics'].str.count('/') == 2)].copy()
        df_pattern_4d.loc[:, 'naics'] = df_pattern_4d['naics'].str.replace('/', '', regex=False)
        
        # Create a dictionary for the old NAICS codes and their descriptions
        naics_description_dict = df_pattern_4d.set_index('naics')['DESCRIPTION'].to_dict()

        # Mappings for NAICS codes
        naics_to_replace_mappings = {
            '3320A1': ['3321', '3322', '3325', '3326', '3329'],
            '3320A2': ['3323', '3324'],
            '3250A1': ['3251', '3252', '3253', '3259'],
            '3250A2': ['3255', '3256'],
            '3330A1': ['3331', '3332', '3334', '3339'],
            '3370A1': ['3371', '3372'],
            '4230A1': ['4232', '4233', '4235', '4236', '4237', '4239'],
            '4240A1': ['4244', '4248'],
            '4240A2': ['4242', '4246'],
            '4240A3': ['4241', '4247', '4249'],
            '4450A1': ['4451', '4452'],
            '5320A1': ['5322', '5323', '5324']
        }

        for new_naics, old_naics_list in naics_to_replace_mappings.items():
            # Assign the new `naics` code
            df_pattern_4d.loc[df_pattern_4d['naics'].isin(old_naics_list), 'naics'] = new_naics
            
            # Create a description for all old codes in the format "code (description)"
            old_descriptions = [
                f"{code} ({naics_description_dict[code]})"
                for code in old_naics_list
                if code in naics_description_dict
            ]
            
            # Create the description as a concatenated string
            description = ', '.join(old_descriptions)
            
            # Set the description for all rows with the new `naics` code
            df_pattern_4d.loc[df_pattern_4d['naics'] == new_naics, 'DESCRIPTION'] = description
        
        # Reset index and save the processed DataFrame
        df_pattern_4d.reset_index(drop=True, inplace=True)
        df_pattern_4d.to_pickle('data/processed_data/pkl/df_pattern_4d.pickle')
        
        # Create and save naics_overview
        naics_overview = df_pattern_4d[['naics', 'DESCRIPTION']].drop_duplicates()
        naics_overview.to_pickle('data/processed_data/pkl/naics_overview.pickle')
        
        # Create and save occ_overview
        occ_overview = df_occupation[['OCC_CODE', 'OCC_TITLE']].drop_duplicates()
        occ_overview.to_pickle('data/processed_data/pkl/occ_overview.pickle')
        
        return df_pattern_4d, naics_overview, occ_overview


    @staticmethod
    def analyze_occupation_data(df, filtered_occ_codes):
        """
        Analyzes occupation data based on pre-filtered occupation codes.
        """
        # Ensure filtered_occ_codes is a list
        if isinstance(filtered_occ_codes, pd.Series):
            filtered_occ_codes = filtered_occ_codes.tolist()
    
        # Remove the first element, if it exists
        if filtered_occ_codes and len(filtered_occ_codes) > 0:
            filtered_occ_codes.pop(0)
    
        # Filter the DataFrame
        ratio_df = df.query("AREA_TYPE == 1 and O_GROUP == 'detailed' and I_GROUP == '4-digit'")
        occu_prio = ratio_df.query("OCC_CODE.isin(@filtered_occ_codes)")
    
        # NAICS codes for prioritized occupations
        naics_prio_list = occu_prio["NAICS"].unique()
        naics_prio = ratio_df.query("NAICS.isin(@naics_prio_list)")
    
        # Display statistics
        print('Occupation_df')
        print(f'Total number of industries: {len(ratio_df["NAICS"].unique())}')
        print(f'Number of industries for prioritized occupations are working in: {len(occu_prio["NAICS"].unique())}')
        print("")
        print(f'Total number of employees: {ratio_df["TOT_EMP"].sum()}')
        print(f'Number of employees in prioritized industries: {naics_prio["TOT_EMP"].sum()}')
        print(f'Number of employees in prioritized occupations: {occu_prio["TOT_EMP"].sum()}')
    
        # Create a dictionary for the occupations
        OCC_desc = occu_prio[['OCC_CODE', 'OCC_TITLE']].drop_duplicates().set_index('OCC_CODE')['OCC_TITLE'].to_dict()
    
        # Pivot table for employee distribution
        pivot_table = pd.pivot_table(naics_prio, values='TOT_EMP', index='NAICS', columns='OCC_CODE', aggfunc='sum')
    
        # Calculate percentages
        pivot_table_percent = pivot_table.div(pivot_table.sum(axis=1), axis=0)
        pivot_table_percent['Total'] = pivot_table_percent.sum(axis=1).round()
        pivot_table_percent.reset_index(inplace=True)
    
        # Calculate percentages for prioritized occupations
        pivot_table_percent_prio = pivot_table.div(pivot_table.sum(axis=1), axis=0)
        pivot_table_percent_prio['OTHERS'] = 1 - pivot_table_percent_prio[filtered_occ_codes].sum(axis=1)
    
        # Keep relevant columns
        columns_to_keep = filtered_occ_codes + ['OTHERS']
        pivot_table_percent_prio = pivot_table_percent_prio[columns_to_keep].reset_index()
        
        pivot_table_percent_prio['NAICS'] = pivot_table_percent_prio['NAICS'].astype(str)
        pivot_table_percent_prio.loc[pivot_table_percent_prio['NAICS'].str.endswith('00'), 'NAICS'] = \
        pivot_table_percent_prio['NAICS'].str[:-2]
        
        pivot_table_percent['NAICS'] = pivot_table_percent['NAICS'].astype(str)
        pivot_table_percent.loc[pivot_table_percent['NAICS'].str.endswith('00'), 'NAICS'] = \
        pivot_table_percent['NAICS'].str[:-2]
    
        # Check if the sum equals 100%
        print(f"Unique values of 'Total' percentage: {pivot_table_percent['Total'].unique().astype(int)}")
        all_occu_list = pivot_table_percent.columns[1:-1].tolist()
    
        # Save the results
        pivot_table_percent_prio.to_pickle('data/processed_data/pkl/pivot_table_percent_prio.pickle')
        pivot_table_percent.to_pickle('data/processed_data/pkl/pivot_table_percent.pickle')
        pd.Series(OCC_desc).to_pickle('data/processed_data/pkl/OCC_desc.pickle')
        pd.Series(all_occu_list).to_pickle('data/processed_data/pkl/all_occu_list.pickle')
    
        return pivot_table_percent_prio, pivot_table_percent, OCC_desc, all_occu_list
        

    @staticmethod
    def filter_by_naics(df_pattern_4d, pivot_table_percent):
        """
        Filter the df_pattern_4d DataFrame based on the intersection of NAICS codes 
        with the unique NAICS codes from the pivot_table_percent DataFrame.
        
        Parameters:
        - df_pattern_4d (DataFrame): The DataFrame containing NAICS codes to filter.
        - pivot_table_percent (DataFrame): The DataFrame containing the NAICS codes to compare against.
    
        Returns:
        - df_pattern_4d_filtered (DataFrame): The filtered DataFrame containing only the matching NAICS codes.
        """
        # Extract unique NAICS codes from both DataFrames
        unique_pattern = df_pattern_4d['naics'].unique()
        unique_occ = pivot_table_percent['NAICS'].unique()
    
        # Calculate missing values and intersection
        misssing_values = pd.Series(list(set(unique_occ) - set(unique_pattern)))
        intersection = pd.Series(list(set(unique_occ).intersection(unique_pattern)))
    
        # Filter the df_pattern_4d DataFrame based on the intersection of NAICS codes
        df_pattern_4d_filtered = df_pattern_4d[df_pattern_4d['naics'].isin(intersection)]
        
        # Print statistics for debugging purposes
        print(f'Count of unique NAICS Codes in Pattern Data: {len(unique_pattern)}')
        print(f'Count of unique NAICS Codes in Occupation Data: {len(unique_occ)}')
        print(f'Count of NAICS Code intersection of both datasets: {len(intersection)}')
        print(f'Missing NAICS Codes:\n{misssing_values}')
    
        # Save the filtered DataFrame to a pickle file
        df_pattern_4d_filtered.to_pickle('data/processed_data/pkl/df_pattern_4d_filtered.pickle')
        
        return df_pattern_4d_filtered
        
    @staticmethod
    def merge_naics_data(df_pattern_4d_filtered, pivot_table_percent):
        """
        Aggregates employment data by NAICS code and county, merges it with occupation percentage data, 
        and calculates the absolute number of employees in each occupation group.
        Saves the results to a pickle file for future use.

        Args:
            df_pattern_4d_filtered: DataFrame containing filtered pattern data with employment information.
            pivot_table_percent: DataFrame containing occupation percentages for each NAICS code.

        Returns:
            merged_df_county: Aggregated DataFrame with employment data and calculated occupation group values.
        """
        
        # Group by 'naics', 'FIPS', and 'fipstate' and sum 'emp' column
        grouped_county = df_pattern_4d_filtered.groupby(['naics', 'FIPS', 'fipstate'])['emp'].sum().reset_index()
        grouped_county = grouped_county.rename(columns={'emp': 'emp_total_county_naics'})

        # Debug outputs
        print(f"Number of unique NAICS codes: {len(grouped_county['naics'].unique())}")
        print(f"Number of rows in the grouped DataFrame: {len(grouped_county)}")
        print(f"Total employment in grouped DataFrame: {grouped_county['emp_total_county_naics'].sum()}")

        # Prepare DataFrames for merging
        df1 = grouped_county
        df2 = pivot_table_percent.iloc[:, :-1]  # Exclude the last column ('Sum')

        # Extract occupation columns (excluding 'NAICS')
        occupation_columns = df2.columns[1:]

        # Merge DataFrames based on 'naics' column
        merged_df = pd.merge(df1, df2, how='left', left_on='naics', right_on='NAICS')

        # Calculate absolute numbers for each occupation group
        for occupation in occupation_columns:
            merged_df[occupation] = round(merged_df['emp_total_county_naics'] * merged_df[occupation], 2)

        # Drop redundant 'NAICS' column
        merged_df.drop(columns=['NAICS'], inplace=True)

        # Calculate sum and delta for verification
        merged_df['Sum'] = merged_df[occupation_columns].sum(axis=1)
        merged_df['Delta'] = merged_df['Sum'] - merged_df['emp_total_county_naics']

        # Final aggregation
        merged_df_county = merged_df.groupby(['FIPS', 'fipstate', 'naics', 'emp_total_county_naics']).sum().reset_index()

        # Save to pickle file
        merged_df_county.to_pickle('data/processed_data/pkl/merged_df_county.pickle')

        # Final debug output
        print(f"Number of rows in the final DataFrame: {len(merged_df_county)}")

        return merged_df_county    


    @staticmethod
    def melt_dataframe_in_chunks(df, output_folder="data/processed_data/melted_data", chunk_size=5000):
        """
        This function melts a DataFrame in chunks and saves the results to CSV files.

        1. **Check for Required Columns**: 
           - The function checks if the DataFrame contains the required columns ('FIPS', 'fipstate', 'naics', 'emp_total_county_naics').
           - If any of these columns are missing, it raises a `ValueError`.

        2. **Chunk Processing**:
           - The DataFrame is divided into chunks based on the specified `chunk_size` (default 5000 rows per chunk).
           - The total number of chunks is calculated based on the length of the DataFrame.
           - Each chunk is processed independently to handle large datasets efficiently.

        3. **Melting the DataFrame**:
           - The function uses `pd.melt` to reshape each chunk, turning selected columns into rows under the 'OCC_CODE' and 'emp_pattern' columns.
           - The `id_vars` for melting are columns that uniquely identify each row (such as 'FIPS', 'fipstate', etc.).
           - The `value_vars` are the columns that will be "melted" (in this case, selected columns excluding the first and last few).

        4. **Saving the Processed Chunks**:
           - After melting, the chunk is saved as a CSV file in the specified output folder.
           - A file name is generated for each chunk (e.g., `melted_data_chunk_0.csv`, `melted_data_chunk_1.csv`, etc.).
           - The `FIPS` column is converted to a string format to ensure consistent storage.

        5. **Progress Output**:
           - The function prints the progress of processing each chunk (e.g., "Processed chunk 1/10").

        Args:
        df: The original DataFrame to be melted.
        output_folder: The folder where the CSV files will be saved (default: "melted_data").
        chunk_size: The size of the chunks (default: 5000).
        """
        
        # Check if the required columns are present
        if not all(col in df.columns for col in ['FIPS', 'fipstate', 'naics', 'emp_total_county_naics']):
            raise ValueError("The DataFrame must contain the columns 'FIPS', 'fipstate', 'naics', and 'emp_total_county_naics'.")
        
        selected_columns = df.columns[4:-2]

        # Calculate the number of chunks
        num_chunks = len(df) // chunk_size + 1

        # Create a folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Iterate over the chunks of the DataFrame
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size
            chunk_df = df.iloc[start_idx:end_idx].copy()  # Create a copy of the chunk
            
            # Apply melt function to the chunk
            df_melted_chunk = pd.melt(chunk_df, 
                                       id_vars=['FIPS', 'fipstate', 'naics', 'emp_total_county_naics'], 
                                       value_vars=selected_columns, 
                                       var_name='OCC_CODE', 
                                       value_name='emp_pattern')
            
            # Convert FIPS to string
            df_melted_chunk['FIPS'] = df_melted_chunk['FIPS'].astype(str)
            
            # Save the results of each chunk to a file in the specified folder
            file_name = os.path.join(output_folder, f"melted_data_chunk_{i}.csv")
            df_melted_chunk.to_csv(file_name, index=False)

            # Output progress
            print(f"Processed chunk {i + 1}/{num_chunks}")


    @staticmethod
    def load_and_combine_csvs(input_folder="data/processed_data/melted_data", chunk_size=50000):
        """
        This function loads all CSV files from the specified folder in chunks and combines them into a single DataFrame.

        1. **Loading CSV Files**:
           - The function first identifies all CSV files in the specified folder (`input_folder`).
           - It then iterates over these files and reads them in chunks (to manage large files efficiently).

        2. **Combining Chunks**:
           - The individual chunks from all CSV files are appended to a list (`dfs`).
           - Once all chunks are read, they are concatenated into a single DataFrame.

        3. **Grouping and Sorting**:
           - The combined DataFrame is grouped by 'fipstate' and 'OCC_CODE', summing the 'emp_pattern' values.
           - The grouped data is then sorted by 'fipstate'.

        4. **Saving the Result**:
           - The final grouped and sorted DataFrame is saved as a Pickle file for efficient storage and retrieval.

        Args:
        input_folder: The folder from which the CSV files are loaded (default: "melted_data").
        chunk_size: The number of rows to load per chunk (default: 50000).
        
        Returns:
        df_combined: A DataFrame containing all the combined data from the CSV files.
        """
        # List to store the loaded DataFrames
        dfs = []

        # Count the total number of CSV files in the folder
        all_csv_files = [file for file in os.listdir(input_folder) if file.endswith(".csv")]
        total_files = len(all_csv_files)

        # Iterate over the files in the folder and load them
        for i, file_name in enumerate(all_csv_files):
            file_path = os.path.join(input_folder, file_name)
            print(f"Loading CSV {i + 1}/{total_files}: {file_name}")

            # Use chunks to read the CSV files
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                dfs.append(chunk)

        # Combine the individual DataFrames into a single DataFrame
        df_combined = pd.concat(dfs, ignore_index=True)
        
        # Group the data and sort
        #result = df_combined.groupby(['fipstate', 'OCC_CODE']).agg({'emp_pattern': 'sum'})
        #result = result.sort_values(by='fipstate', ascending=True).reset_index()
        
        # Save the result as a Pickle file
        os.makedirs('data/processed_data/pkl', exist_ok=True)
        #result.to_pickle('processed_data/pkl/df_combined.pickle')
        df_combined.to_pickle('data/processed_data/pkl/df_combined.pickle')
        return df_combined

    @staticmethod
    def process_occupation_data(file_path: str, naics_list: list) -> pd.DataFrame:
        """
        Process the occupation data from a pickle file and group by area and occupation.

        Parameters:
        - file_path: str - Path to the pickle file containing occupation data.
        - naics_list: list - List of NAICS codes to filter occupations.

        Returns:
        - pd.DataFrame - A DataFrame grouped by area and occupation with total employment.
        """
        # Load the DataFrame from the pickle file
        df = pd.read_pickle(file_path)
        
        # Filter the DataFrame based on AREA_TYPE and O_GROUP
        occu_state = df.query("AREA_TYPE == 2 and O_GROUP == 'detailed'")
        
        # Keep only the occupations that are in the provided NAICS list
        occu_state = occu_state[occu_state['OCC_CODE'].isin(naics_list)]
        
        # Group by area and occupation, summing the total employment
        occu_state_grouped = occu_state.groupby(['AREA', 'AREA_TITLE', 'OCC_CODE', 'OCC_TITLE'])['TOT_EMP'].sum().reset_index()
        
        # Rename columns for clarity
        occu_state_grouped = occu_state_grouped.rename(columns={
            'AREA': 'fipstate',
            'AREA_TITLE': 'state',
            'TOT_EMP': 'emp_occu'
        })
        
        # Convert columns to numeric
        occu_state_grouped['emp_occu'] = pd.to_numeric(occu_state_grouped['emp_occu'], errors='coerce')
        occu_state_grouped['fipstate'] = pd.to_numeric(occu_state_grouped['fipstate'], errors='coerce')
        
        # Print the number of unique OCC_CODE values
        print(f"Unique OCC_CODE count: {len(occu_state_grouped['OCC_CODE'].unique())}")
        
        # Save the result as a Pickle file
        occu_state_grouped.to_pickle('data/processed_data/pkl/occu_state_grouped.pickle')
        
        return occu_state_grouped

    @staticmethod
    def merge_and_calculate(occu_state_grouped: pd.DataFrame, df_combined: pd.DataFrame) -> pd.DataFrame:
        """
        Merge occupation data with additional employment pattern data, and calculate key metrics for comparison.

        Parameters:
        - occu_state_grouped: pd.DataFrame - A DataFrame containing grouped occupation data with total employment (`emp_occu`).
        - df_combined: pd.DataFrame - A DataFrame containing combined data with occupation codes (`OCC_CODE`), 
          state codes (`fipstate`), and employment patterns (`emp_pattern`).

        Returns:
        - pd.DataFrame - A DataFrame with merged data and calculated metrics: Delta, Delta %, Factor %, 
          Weighted Delta %, sorted by Delta.
        """
        
        # Group df_combined by 'fipstate' and 'OCC_CODE', summing the 'emp_pattern' values
        df_combined_grouped = df_combined.groupby(['fipstate', 'OCC_CODE']).agg({
            'emp_pattern': 'sum'
        })
        # Sort grouped DataFrame by 'fipstate'
        df_combined_grouped = df_combined_grouped.sort_values(by='fipstate', ascending=True).reset_index()
        
        # Merge the grouped DataFrame with occu_state_grouped on 'fipstate' and 'OCC_CODE'
        occu_state_merged = pd.merge(
            occu_state_grouped, 
            df_combined_grouped[['fipstate', 'OCC_CODE', 'emp_pattern']], 
            on=['fipstate', 'OCC_CODE'], 
            how='left'
        )
        
        # Calculate new columns based on the merged data
        occu_state_merged['Delta'] = occu_state_merged['emp_pattern'] - occu_state_merged['emp_occu']
        occu_state_merged['Delta %'] = round(abs(occu_state_merged['Delta'] / occu_state_merged['emp_occu'] * 100), 1)
        occu_state_merged['Faktor %'] = occu_state_merged['emp_occu'] / occu_state_merged['emp_pattern']
        occu_state_merged['Gewichtung'] = occu_state_merged['emp_occu'] / occu_state_merged['emp_occu'].sum()
        occu_state_merged['Delta % gewichtet'] = round(occu_state_merged['Gewichtung'] * occu_state_merged['Delta %'], 3)
        
        # Sort the merged DataFrame by 'Delta' in descending order and reset the index
        occu_state_merged = occu_state_merged.sort_values(by='Delta', ascending=False).reset_index(drop=True)
        
        # Save the final DataFrame as a pickle file
        occu_state_all = occu_state_merged
        occu_state_all.to_pickle('data/processed_data/pkl/occu_state_all.pickle')
        
        # Return the processed DataFrame
        return occu_state_all


    @staticmethod
    def merge_and_save_data(df_combined: pd.DataFrame, occu_state_all: pd.DataFrame) -> pd.DataFrame:
        """
        Merge df_combined with occu_state_all on 'fipstate' and 'OCC_CODE', then save the merged DataFrame.

        Parameters:
        - df_combined: pd.DataFrame - A DataFrame containing combined data with 'fipstate' and 'OCC_CODE'.
        - occu_state_all: pd.DataFrame - A DataFrame containing occupation data with 'Faktor %' column.

        Returns:
        - pd.DataFrame - The merged DataFrame saved as a pickle file.
        """
        
        # Merge the dataframes on 'fipstate' and 'OCC_CODE' with a left join
        df_combined_2 = pd.merge(df_combined, occu_state_all[['fipstate', 'OCC_CODE', 'Faktor %']], 
                                 on=['fipstate', 'OCC_CODE'], how='left')
        
        # Save the merged dataframe to a pickle file
        df_combined_2.to_pickle("data/processed_data/pkl/df_combined_2.pickle")
        
        # Print the number of unique values in the 'naics' column
        print(f"Anzahl eindeutiger Werte in der Spalte 'naics': {df_combined_2['naics'].nunique()}")
        
        return df_combined_2


    @staticmethod
    def process_employee_data(df_combined_2: pd.DataFrame, filtered_occ_codes: list) -> pd.DataFrame:
        """
        Process employee data by filtering, adjusting patterns, and creating new metrics.
    
        Parameters:
        - df_combined_2: pd.DataFrame - A merged DataFrame containing employment data.
        - filtered_occ_codes: list - List of OCC_CODEs to filter the data.
    
        Returns:
        - pd.DataFrame - The processed DataFrame saved as a pickle file.
        """
        
        # Filter the dataframe and explicitly create a copy
        df_prio = df_combined_2.query("OCC_CODE.isin(@filtered_occ_codes)").copy()
        
        # Print the number of unique values in the 'naics' column
        print(f"Number of unique values in the 'naics' column: {df_prio['naics'].nunique()}")
        
        # Count Inf and NaN values in 'emp_pattern' and 'Faktor %' columns
        inf_emp_pattern = df_prio['emp_pattern'].isin([np.inf]).sum()
        nan_emp_pattern = df_prio['emp_pattern'].isna().sum()
        print(f"Number of Inf values in 'emp_pattern': {inf_emp_pattern}")
        print(f"Number of NaN values in 'emp_pattern': {nan_emp_pattern}")
        
        inf_factor = df_prio['Faktor %'].isin([np.inf]).sum()
        nan_factor = df_prio['Faktor %'].isna().sum()
        print(f"Number of Inf values in 'Faktor %': {inf_factor}")
        print(f"Number of NaN values in 'Faktor %': {nan_factor}")
        
        # Adjust 'emp_pattern' by multiplying with 'Faktor %'
        df_prio['emp_pattern_adjusted'] = df_prio['emp_pattern'] * df_prio['Faktor %']
        
        # Replace NaN values in 'emp_pattern_adjusted' with 0
        df_prio['emp_pattern_adjusted'] = df_prio['emp_pattern_adjusted'].apply(lambda x: np.nan if pd.isna(x) else x)
        df_prio['emp_pattern_adjusted'] = df_prio['emp_pattern_adjusted'].fillna(0)
        
        # Create a new column 'abs_adjustment' for the absolute difference between adjusted and original values
        df_prio['abs_adjustment'] = df_prio['emp_pattern_adjusted'] - df_prio['emp_pattern']
        
        # Output the sum of 'emp_pattern_adjusted' and 'abs_adjustment'
        print(f"Sum of 'emp_pattern_adjusted': {df_prio['emp_pattern_adjusted'].sum()}")
        print(f"Sum of 'abs_adjustment': {df_prio['abs_adjustment'].sum()}")
        
        # Sort the dataframe by 'emp_pattern_adjusted' in descending order
        df_prio = df_prio.sort_values(by='emp_pattern_adjusted', ascending=False)
        
        # Save the processed DataFrame to a pickle file
        df_prio.to_pickle("data/processed_data/pkl/df_prio.pickle")
        
        return df_prio

    @staticmethod
    def process_and_merge_data(df_prio: pd.DataFrame, occu_state_grouped: pd.DataFrame, filtered_occ_codes: list) -> pd.DataFrame:
        """
        Process and merge data by aggregating adjusted patterns and calculating deltas.

        Parameters:
        - df_prio: pd.DataFrame - A DataFrame with prioritized employee data.
        - occu_state_grouped: pd.DataFrame - A DataFrame with grouped occupation data by state.
        - filtered_occ_codes: list - List of OCC_CODEs to filter the data.

        Returns:
        - pd.DataFrame - The merged and processed DataFrame with calculated deltas.
        """
        
        # Group by 'fipstate' and 'OCC_CODE' and sum the 'emp_pattern_adjusted' column
        df_1 = df_prio.groupby(['fipstate', 'OCC_CODE'])['emp_pattern_adjusted'].sum().reset_index()
        
        # Filter the occupation state grouped data for relevant OCC_CODEs
        df_2 = occu_state_grouped.query("OCC_CODE.isin(@filtered_occ_codes)")
        
        # Merge the two DataFrames on 'fipstate' and 'OCC_CODE'
        df_merged_1 = df_2.merge(df_1, on=['fipstate', 'OCC_CODE'])
        
        # Handle NaN values in 'emp_pattern_adjusted'
        df_merged_1['emp_pattern_adjusted'] = df_merged_1['emp_pattern_adjusted'].apply(lambda x: np.nan if pd.isna(x) else x)
        df_merged_1['emp_pattern_adjusted'] = df_merged_1['emp_pattern_adjusted'].fillna(0)
        
        # Calculate the delta between original and adjusted employment
        df_merged_1['delta'] = df_merged_1['emp_occu'] - df_merged_1['emp_pattern_adjusted']
        
        # Print summary statistics
        nan_count = df_merged_1['emp_pattern_adjusted'].isna().sum()
        print(f"Number of NaN values in 'emp_pattern_adjusted': {nan_count}")
        print(f"Sum of 'emp_occu': {df_merged_1['emp_occu'].sum()}")
        print(f"Sum of 'delta': {df_merged_1['delta'].sum()}")
        ratio = df_merged_1['delta'].sum() / df_merged_1['emp_occu'].sum()
        print(f"Ratio of delta to emp_occu: {ratio * 100:.5f}%")
        df_merged_1.to_pickle('data/processed_data/pkl/df_merged_1.pickle')
        return df_merged_1



    @staticmethod
    def clean_and_summarize(df_prio: pd.DataFrame, naics_overview: pd.DataFrame, occ_overview: pd.DataFrame, df_county: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and summarizes the data by handling zeros, NaNs, and mapping titles.

        Parameters:
        - df_prio: pd.DataFrame - The prioritized employee data.
        - naics_overview: pd.DataFrame - NAICS overview with descriptions.
        - occ_overview: pd.DataFrame - Occupation overview with titles.
        - df_county: pd.DataFrame - County-level data with state names.

        Returns:
        - pd.DataFrame - The cleaned and summarized DataFrame.
        """
        
        # Count the number of zeros and NaN values in 'emp_pattern_adjusted'
        number_of_zeros = df_prio['emp_pattern_adjusted'].astype(float).eq(0).sum()
        number_of_nans = df_prio['emp_pattern_adjusted'].isna().sum()

        # Record the number of rows before cleaning
        rows_before_cleaning = len(df_prio)

        # Remove rows with NaN or zero in 'emp_pattern_adjusted'
        df_cleaned = df_prio.dropna(subset=['emp_pattern_adjusted'])
        df_cleaned = df_cleaned[df_cleaned['emp_pattern'] != 0]

        # Map NAICS and OCC_CODE to their respective titles
        df_cleaned['NAICS_TITLE'] = df_cleaned['naics'].map(naics_overview.set_index('naics')['DESCRIPTION'].to_dict())
        df_cleaned['OCC_TITLE'] = df_cleaned['OCC_CODE'].map(occ_overview.set_index('OCC_CODE')['OCC_TITLE'].to_dict())

        # Record the number of rows after cleaning
        rows_after_cleaning = len(df_cleaned)

        # Output cleaning statistics
        print("Number of zeros:", number_of_zeros)
        print("Number of NaN values:", number_of_nans)
        print('Rows before cleaning:', rows_before_cleaning)
        print('Rows after cleaning:', rows_after_cleaning)
        print('Total Employees:', df_cleaned['emp_pattern_adjusted'].sum())
        print('Length of cleaned DataFrame:', rows_after_cleaning)
        print('Number of Counties:', df_cleaned['FIPS'].nunique())
        print('Number of Occupations:', df_cleaned['OCC_CODE'].nunique())
        print('Number of Industries:', df_cleaned['naics'].nunique())

        # Transform columns for further analysis
        df_cleaned = df_cleaned.rename(columns={
            'fipstate': 'statefp',
            'emp_pattern_adjusted': 'emp_occupation',
            'emp_total_county_naics': 'emp_total_county_naics'
        })
        df_cleaned = df_cleaned[['FIPS', 'statefp', 'naics', 'NAICS_TITLE', 'emp_total_county_naics', 
                                 'OCC_CODE', 'OCC_TITLE', 'emp_occupation']]
        df_cleaned = df_cleaned.reset_index(drop=True)

        # Format 'FIPS' and 'statefp' columns to ensure consistency
        df_cleaned['FIPS'] = df_cleaned['FIPS'].astype(str).apply(lambda x: '0' + x if len(x) == 4 else x)
        df_cleaned['statefp'] = df_cleaned['statefp'].astype(str).apply(lambda x: '0' + x if len(x) == 1 else x)

        # Merge with county data to include 'state_name'
        df_county_unique = df_county[['statefp', 'state_name']].drop_duplicates(subset='statefp')
        df_cleaned = df_cleaned.merge(df_county_unique[['statefp', 'state_name']], on='statefp', how='left')
        df_cleaned.rename(columns={'naics': 'NAICS_CODE'}, inplace=True)
        df_cleaned.to_pickle('data/processed_data/pkl/df_cleaned.pickle')
        # Output the cleaned DataFrame
        print("Final Cleaned DataFrame:")
        return df_cleaned
import pandas as pd




class MasterDataBuilder:
    def __init__(self, occ_path, naics_path, county_information):
        """
        Initializes the class with DataFrames loaded from given Pickle paths.

        :param occ_path: File path to the Pickle file with OCC_CODE and emp_occupation data
        :param naics_path: File path to the Pickle file with NAICS_CODE, emp, and est data
        :param county_information: DataFrame containing FIPS data
        """
        self.original_occupation_df = pd.read_pickle(occ_path)
        self.original_pattern_df = pd.read_pickle(naics_path)
        self.original_pattern_df = self.original_pattern_df.rename(columns={'naics': 'NAICS_CODE'})
        self.original_pattern_df['FIPS'] = self.original_pattern_df['FIPS'].astype(str).str.zfill(5)
        self.county_information = county_information
        self.master_df = pd.DataFrame(county_information['FIPS'].unique(), columns=['FIPS'])

    def _process_occ(self, occ_codes):
            """
            Processes the OCC_CODES and returns a list of DataFrames, one for each OCC_CODE.
            """
            df_list = []
            for code in occ_codes:
                filtered = self.original_occupation_df[self.original_occupation_df['OCC_CODE'] == code]
                occ_code = filtered['OCC_CODE'].unique()[0]
                aggregated = (
                    filtered.groupby(['FIPS', 'OCC_CODE'])
                    .agg(total_emp_occu=('emp_occupation', 'sum'))
                    .reset_index()
                )
                aggregated.columns = ['FIPS', 'OCC_CODE', f'total_emp_occu_{occ_code}']
                df_list.append(aggregated)
            return df_list
    
    def _process_naics(self, naics_codes):
        """
        Processes the NAICS_CODES and returns a list of DataFrames,
        each containing only the employment data for the given NAICS_CODE.
        """
        df_list = []
        for code in naics_codes:
            filtered = self.original_pattern_df[self.original_pattern_df['NAICS_CODE'] == code]
            if filtered.empty:
                continue  # Falls kein Eintrag für den Code vorhanden ist
    
            naics_code = filtered['NAICS_CODE'].unique()[0]
            aggregated = (
                filtered.groupby(['FIPS', 'NAICS_CODE'])
                .agg(total_emp_naics=('emp', 'sum'))
                .reset_index()
            )
            aggregated.columns = [
                'FIPS',
                'NAICS_CODE',
                f'total_emp_naics_{naics_code}',
            ]
            df_list.append(aggregated[['FIPS', f'total_emp_naics_{naics_code}']])
        return df_list
    
    def build_master_df(self, occ_codes, naics_codes, save_path):
        """
        Builds the master DataFrame and saves it directly as a Pickle file.
    
        :param occ_codes: Combined list of relevant OCC_CODES
        :param naics_codes: Combined list of relevant NAICS_CODES
        :param save_path: File path where the Pickle should be saved
        """
        # Process all OCC codes
        df_list_occ = self._process_occ(occ_codes)
        for occ_df in df_list_occ:
            value_column = occ_df.columns[2]
            if value_column not in self.master_df.columns:
                self.master_df = self.master_df.merge(occ_df[['FIPS', value_column]], on='FIPS', how='left')
    
        # Process all NAICS codes (nur emp-Features!)
        df_list_naics = self._process_naics(naics_codes)
        for naics_df in df_list_naics:
            value_column = naics_df.columns[1]  # da nur 'FIPS' und 1 emp-Spalte
            if value_column not in self.master_df.columns:
                self.master_df = self.master_df.merge(naics_df[['FIPS', value_column]], on='FIPS', how='left')
    
        # Fill all NaN values with 0
        master_df = self.master_df.fillna(0)
    
        # Save master_df as a Pickle file
        master_df.to_pickle(save_path)
        print(f"master_df successfully saved as a Pickle file at '{save_path}'")
    
        return master_df


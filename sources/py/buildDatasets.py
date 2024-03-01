import pandas as pd
from os import listdir
import pathManager


class buildDatasets:
    def __init__(self, p_chemical, p_desc, p_dir_target, p_dir_out):
        self.p_chemical = p_chemical
        self.p_desc = p_desc
        self.p_dir_target = p_dir_target
        self.p_dir_out = p_dir_out

        self.p_dir_set = pathManager.create_folder(p_dir_out + "datasets/")

    def build_dataset(self, p_target):
        """
        Build the dataset
        """
        # read chemical
        df_chem = pd.read_csv(self.p_chemical)
        df_chem.drop_duplicates(subset=['INPUT'], keep='first', inplace=True)
        
        # read target
        df_target = pd.read_csv(p_target)

        # merge
        df_target_chem = pd.merge(df_target, df_chem, right_on='INPUT', left_on='casn', how='inner')

        # merge on descriptors
        df_desc = pd.read_csv(self.p_desc)
        df_target_chem_desc = pd.merge(df_target_chem, df_desc, right_on='DTXSID', left_on='DTXSID', how='inner')
        
        # drop extract columns
        df_target_chem_desc.drop(columns=['INPUT', 'casn', "FOUND_BY", "PREFERRED_NAME", "SMILES"], inplace=True)
        df_target_chem_desc.to_csv(self.p_dir_set + p_target.split("/")[-1], index=False)


    def build_all(self):
        
        for p_cell in listdir(self.p_dir_target):
            self.build_dataset(self.p_dir_target + p_cell)
            
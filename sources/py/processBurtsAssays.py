import pandas as pd
from os import path, listdir
import pathManager


class processBurtsAssays:
    def __init__(self, input_file, p_dir_out):
        self.input_file = input_file
        self.p_dir_out = p_dir_out

    def extract(self):
        
        # merge columns
        df_brust = pd.read_csv(self.input_file)
        df_brust["cell_timepoint"] = df_brust["cell_short_name"].astype(str) + "__" + df_brust["timepoint_hr"].astype(str)

        # create folders
        p_dir_extracted = self.p_dir_out + "extracted/"
        pathManager.create_folder(p_dir_extracted)
        

        
        # split using cell_short_name
        df_by_cell = df_brust.groupby('cell_timepoint')
        # write by cell
        for cell_name, df_cell in df_by_cell:
            cell_name = cell_name.replace("/", "_").strip()
            p_out = p_dir_extracted + cell_name + ".csv"
            df_cell.to_csv(p_out, index=False)


        # generate the list of CASRN
        l_casrn = df_brust['casn'].unique()
        # write by casrn
        df_casrn = pd.DataFrame(l_casrn)
        p_out = self.p_dir_out + "casrn.csv"
        df_casrn.to_csv(p_out, index=False)

        self.p_dir_extracted = p_dir_extracted
        


    def process_for_modeling(self, type_merge="median"):
        
        p_dir_formated = self.p_dir_out + "formated/"
        self.p_dir_formated = p_dir_formated
        pathManager.create_folder(p_dir_formated)

        for extracted_dataset in listdir(self.p_dir_extracted):
            df_extracted = pd.read_csv(self.p_dir_extracted + extracted_dataset)
            df_group_chem = df_extracted.groupby('casn')

            df_formated = pd.DataFrame()
            for casn, df_chem in df_group_chem:
                if type_merge == "median":
                    med = df_chem["AC50"].median()
                df_by_chem = pd.DataFrame({"casn": [casn], "AC50": [med]})
                df_formated = pd.concat([df_formated, df_by_chem], axis=0)
            df_formated.to_csv(p_dir_formated + extracted_dataset, index=False)    


    def by_cell_summary(self):

        l_cells = listdir(self.p_dir_extracted)

        l_cells = [x[:-4] for x in l_cells]


        d_out = {}
        for cell in l_cells:
            d_out[cell] = {}

            p_in = self.p_dir_out + "extracted/" + cell + ".csv"
            df_cell = pd.read_csv(p_in)

            nb_chemical = len(df_cell['casn'].unique())
            nb_active = len(df_cell[df_cell['new_hitc'] == 1])
            nb_inactive = len(df_cell[df_cell['new_hitc'] == 0])
            avg_AC50 = df_cell['AC50'].mean()
            nb_different_assays = len(df_cell['assay_component_endpoint_name'].unique())

            d_out[cell]['nb_chemical'] = nb_chemical
            d_out[cell]['nb_active_record'] = nb_active
            d_out[cell]['nb_inactive_recod'] = nb_inactive
            d_out[cell]['avg_AC50'] = avg_AC50
            d_out[cell]['nb_different_assays'] = nb_different_assays

            nb_act_inact = 0
            nb_act = 0
            nb_inact = 0
            df_group_chem = df_cell.groupby('casn')
            for casn, df_chem in df_group_chem:
                l_hitc = df_chem['new_hitc'].tolist()
                l_hitc.sort()
                l_hitc = list(set(l_hitc))
                if l_hitc == [0, 1]:
                    nb_act_inact = nb_act_inact + 1
                elif l_hitc == [1]:
                    nb_act = nb_act + 1
                else:
                    nb_inact = nb_inact + 1
                
            d_out[cell]['nb_chem_act_inact'] = nb_act_inact
            d_out[cell]['nb_chem_act'] = nb_act
            d_out[cell]['nb_chem_inact'] = nb_inact
        
        p_summary = self.p_dir_out + "summary.csv"
        f_summary = open(p_summary, "w")
        f_summary.write("cell,nb_chemical,nb_active_record,nb_inactive_record,avg_ac50,nb_different_assays,nb_chem_act_inact,nb_chem_act,nb_chem_inact\n")
        for cell in l_cells:
            f_summary.write(cell + "," + str(d_out[cell]['nb_chemical']) + "," + str(d_out[cell]['nb_active_record']) + "," + str(d_out[cell]['nb_inactive_recod']) + "," + str(d_out[cell]['avg_AC50']) + "," + str(d_out[cell]['nb_different_assays']) + "," + str(d_out[cell]['nb_chem_act_inact']) + "," + str(d_out[cell]['nb_chem_act']) + "," + str(d_out[cell]['nb_chem_inact']) + "\n")
        f_summary.close()



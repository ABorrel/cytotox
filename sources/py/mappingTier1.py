import pandas as pd
from re import search


class mappingTier1:

    def __init__(self, p_cHTS_assay, p_assay_map, p_assay_burst, p_dir_out) :

        self.p_cHTS_assay = p_cHTS_assay
        self.p_assay_map = p_assay_map
        self.p_assay_burst = p_assay_burst
        self.p_dir_out = p_dir_out

        pass

    def mapping_on_burst(self):

        df_assays_HTS = pd.read_excel(self.p_cHTS_assay, sheet_name="invitrodb34_AssayAnnotation")


        df_burst = pd.read_csv(self.p_assay_burst)
        l_cell = df_burst["cell_short_name"].astype(str)
        l_cell_burst = list(l_cell)

        count = 0
        for index, row in df_assays_HTS.iterrows():
            cell = row["Invitro Assay Format"]
            if search("cell line", str(cell)):
                cell = cell.split("(")[-1]
                cell = cell.replace(")", "")

                for cell_burst in l_cell_burst:
                    if search(cell, cell_burst):
                        count += 1
                        print(cell, cell_burst)
                        break

        print(count)
    


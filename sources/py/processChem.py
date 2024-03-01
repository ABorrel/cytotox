import pathManager
import CompDesc
import pandas as pd
from os import path, listdir

class processChem:
    def __init__(self, p_chem, p_dir_out):
        self.p_chem = p_chem
        self.p_dir_out = p_dir_out
        self.p_dir_desc = pathManager.create_folder(p_dir_out + "desc/")
              
    def computeDesc(self):

        p_desc_out = self.p_dir_out + "desc.csv"
        if path.exists(p_desc_out):
            return
        
        df_chem = pd.read_csv(self.p_chem)
        df_chem = df_chem.sample(frac=1)
        
        for index, row in df_chem.iterrows():
            dtx = row['DTXSID']
            if type(dtx) != str:
                continue
            p_dtx_desc = self.p_dir_desc + dtx + ".csv"
            if not path.exists(p_dtx_desc):
                c_desc = CompDesc.CompDesc(str(row['SMILES']), self.p_dir_desc)
                c_desc.prepChem()
                if c_desc.err == 1:
                    continue
                c_desc.computeAll2D()
                if c_desc.err == 1:
                    continue

                # write Desc
                filin = open(self.p_dir_desc + row['DTXSID'] + ".csv", "w")
                filin.write("%s\n"%("\t".join(c_desc.all2D.keys())))
                filin.write("%s\n"%("\t".join([str(c_desc.all2D[k]) for k in c_desc.all2D.keys()])))
                filin.close()
        
        # merge all of the 
        df_merge = pd.DataFrame()
        l_p_desc = listdir(self.p_dir_desc)
        for p_desc in l_p_desc:
            p_desc = self.p_dir_desc + p_desc
            df_desc = pd.read_csv(p_desc, sep="\t")
            df_desc["DTXSID"] = p_desc.split("/")[-1][:-4]
            df_merge = pd.concat([df_merge, df_desc], axis=0)
        df_merge.to_csv(p_desc_out, index=False)

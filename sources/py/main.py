from os import path, getenv
import processBurtsAssays
import processChem
import pathManager
import buildDatasets
import runRegModeling
import sys


RUN_ENV = getenv('RUN_ENV')

print(RUN_ENV)

PDIR_SCRIPT = path.realpath(path.dirname("")) + "/"
PDIR_ROOT = path.abspath(PDIR_SCRIPT + "../../") + "/"

if RUN_ENV == "biowulf":
    # format the input for modeling
    PDIR_INPUT = "/data/borrela2/CIS-419/input/"
    PDIR_OUTPUT = "/data/borrela2/CIS-419/output/"
else:
    PDIR_INPUT = PDIR_ROOT + "input/"
    PDIR_OUTPUT = PDIR_ROOT + "output/"


p_burst_assays = PDIR_INPUT + "cytotox_assay_burst1.csv"



####
# MAIN
########

p_dir_burst_pred = pathManager.create_folder(PDIR_OUTPUT + "burst_prep/")
c_prep = processBurtsAssays.processBurtsAssays(p_burst_assays, p_dir_burst_pred)
c_prep.extract()
c_prep.process_for_modeling()
c_prep.by_cell_summary()


# prep chemical
p_chem = PDIR_INPUT + "chemicals.csv"
p_dir_chem = pathManager.create_folder(PDIR_OUTPUT + "chem_prep/")
c_chem = processChem.processChem(p_chem, p_dir_chem)
c_chem.computeDesc()

# build dataset
cBuild = buildDatasets.buildDatasets(p_chem, PDIR_OUTPUT + "chem_prep/desc.csv", PDIR_OUTPUT + "burst_prep/formated/", PDIR_OUTPUT)
cBuild.build_all()



#### dataset
p_dataset = PDIR_OUTPUT + "datasets/HepG2__24.csv"
p_dataset = PDIR_OUTPUT + "datasets/HEK293__24.csv"

type_model = "Xboost"
type_model = "RF"


p_dir_modeling = PDIR_OUTPUT + "modeling/"
pathManager.create_folder(p_dir_modeling)

c_modeling = runRegModeling.runRegModeling(p_dataset, type_model, p_dir_modeling)
c_modeling.format_dataset_for_modeling()
c_modeling.run_undersampling(run=10, ratio_inact=0.3)








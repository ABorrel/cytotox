from os import path
import processBurtsAssays
import processChem
import pathManager
import buildDatasets
import runRegModeling


PDIR_SCRIPT = path.realpath(path.dirname("")) + "/"
PDIR_ROOT = path.abspath(PDIR_SCRIPT + "../../") + "/"


PDIR_INPUT = PDIR_ROOT + "input/"
PDIR_OUTPUT = PDIR_ROOT + "output/"


p_burst_assays = PDIR_INPUT + "cytotox_assay_burst1.csv"



####
# MAIN
########

p_dir_burst_pred = pathManager.create_folder(PDIR_OUTPUT + "burst_prep/")
c_prep = processBurtsAssays.processBurtsAssays(p_burst_assays, p_dir_burst_pred)
#c_prep.extract()
#c_prep.process_for_modeling()
#c_prep.by_cell_summary()



# prep chemical
p_chem = PDIR_INPUT + "chemicals.csv"
#p_dir_chem = pathManager.create_folder(PDIR_OUTPUT + "chem_prep/")
#c_chem = processChem.processChem(p_chem, p_dir_chem)
#c_chem.computeDesc()

# build dataset
#cBuild = buildDatasets.buildDatasets(p_chem, PDIR_OUTPUT + "chem_prep/desc.csv", PDIR_OUTPUT + "burst_prep/formated/", PDIR_OUTPUT)
#cBuild.build_all()


p_dataset = "C:/Users/aborrel/work/CIS_419/output/datasets/HepG2__24.csv"
p_dir_modeling = PDIR_OUTPUT + "modeling/"
pathManager.create_folder(p_dir_modeling)

c_modeling = runRegModeling.runRegModeling(p_dataset, p_dir_modeling)
c_modeling.format_dataset_for_modeling()
c_modeling.run_Xboost()

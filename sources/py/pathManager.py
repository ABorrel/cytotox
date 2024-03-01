from os import listdir, remove, makedirs, path
from shutil import rmtree


def clean_folder(prin, l_p_save=[]):
    """
    Clean folder
    - l_p_save: list of files that need to be saved
    """
    lfiles = listdir(prin)
    if len(lfiles) != 0:
        for filin in lfiles:
            # problem with folder
            p_remove = prin + filin
            if p_remove in l_p_save:
                continue
            else:
                try: remove(p_remove)
                except: rmtree(p_remove)
    return prin



def create_folder(prin, clean=0):
    
    if not path.exists(prin):
        makedirs(prin)

    if clean == 1:
        clean_folder(prin)

    return prin
import os, json
import shutil
from pprint import pprint


def lister_paths_file_tif():
    liste_path_files= list()
    dir_base  = "./web/download"
    for path, subdirs, files in os.walk(dir_base):
        for name in files:
            if name.split(".")[-1]=="tif":
                path_file = os.path.join(path,name)
                liste_path_files.append(path_file)
    return liste_path_files

def lister_paths_file_dzi():
    liste_path_files= list()
    dir_base  = "./web/pan"
    for path, subdirs, files in os.walk(dir_base):
        for name in files:
            if name.split(".")[-1]=="dzi":
                path_file = os.path.join(path,name)
                path_file_OK=path_file.replace("\\","/")[1:]
                liste_path_files.append(path_file_OK)
    return liste_path_files

def rename_tif():
    path_files_tif = lister_paths_file_tif()
    for path_file_tif in path_files_tif:
        new_path_file = path_file_tif.replace(".tif","")+".tif"
        if new_path_file != path_file_tif:
            shutil.move(path_file_tif,new_path_file)

def index_dzi():
    files_path = lister_paths_file_dzi()
    files_path.sort()
    pprint(files_path)
    dico = dict()
    for file_path in files_path:
        name_file_tif = file_path.split("/")[-1].replace(".dzi",".tif")
        print(name_file_tif)
        path_file_tif  = "/dowload/"+name_file_tif
        if os.path.isfile("."+path_file_tif):
            dico[file_path]=path_file_tif
        else:
            dico[file_path]=""
    str_js = "dataBase = " + json.dumps(dico,indent=4)
    file_object  = open("./js/data.js", "w")
    file_object.write(str_js)
    file_object.close()
    pprint(dico)

if __name__ == "__main__":
    rename_tif()
    index_dzi()
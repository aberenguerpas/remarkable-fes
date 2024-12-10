import os
for folder_name in os.listdir("dataset/"):
    if folder_name == ".DS_Store" or folder_name == "dataset.csv":
        continue
    for file_name in os.listdir("dataset/"+folder_name):
        if file_name.startswith("ext_"):
            os.remove("dataset/"+folder_name+"/"+file_name)
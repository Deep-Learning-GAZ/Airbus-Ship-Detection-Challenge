import os


def joinFolder(folder, file_names_list):
    return [os.path.join(folder, file_name) for file_name in file_names_list]

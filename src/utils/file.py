
import os
# enssure folder exists


def ensure_folder_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

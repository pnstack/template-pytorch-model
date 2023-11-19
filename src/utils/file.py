
import os
import glob

from multiprocessing import Pool
import timeit
from tqdm import tqdm 

def ensure_folder_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def find_files(pattern):
    files = glob.glob(pattern)
    return files


def save_to_file(file_name, content):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(content)
    # print("Saved: ", file_name)
    
    
def multi_process_list(input_list, function, num_workers=20):
    with Pool(num_workers) as p:
        result = list(tqdm(p.imap_unordered(function, input_list), total=len(input_list)))
    return result

def process_list(input_list, function):
    # process with process bar
    result = []
    for item in tqdm(input_list):
        result.append(function(item))
    return result

def read_txt_content(file):
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
    return text
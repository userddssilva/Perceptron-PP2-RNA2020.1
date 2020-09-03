import numpy as np
import os


def show_shape(file_parameter=''):
    array = np.fromfile(file=file_parameter)
    print(array.shape)

def run_files_data(path_data = ''):
    for file_data in os.listdir(path_data):
        full_path = os.path.join(path_data, file_data)
        show_shape(full_path)

if __name__ == "__main__":
    run_files_data('rna-2020.1-pp2-data')
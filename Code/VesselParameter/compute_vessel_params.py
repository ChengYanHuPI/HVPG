# -*- coding: UTF-8 -*-

from compute_centerlines import main
from compute_params import compute_params

if __name__ == '__main__':
    import sys

    file_path = sys.argv[1]
    file_path_list = main(file_path)
    for fp in file_path_list:
        print(fp[0], fp[1], fp[2])
        compute_params(fp[0], fp[1], fp[2])

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from dotenv import load_dotenv
from scipy.io import loadmat


import cv2
load_dotenv()
DATASET = os.getenv('DATASET_PATH')
from dotenv import load_dotenv
mat = loadmat(DATASET + '/file_list.mat')

for num, img_path in enumerate(mat['file_list']):
    print(num)
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

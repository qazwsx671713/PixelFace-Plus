# Miscellaneous utility functions
import numpy as np
import PIL.Image
import seaborn as sns
from termcolor import colored
from tqdm import tqdm
import math
import os

# Colorful prints
# ----------------------------------------------------------------------------

# string -> bold string
def bold(txt, **kwargs):
    return colored(str(txt),attrs = ["bold"])

# string -> colorful bold string
def bcolored(txt, color):
    return colored(str(txt), color, attrs = ["bold"])

# conditional coloring if num > maxval
# maxval = 0 turns functionality off
def cond_bcolored(num, maxval, color):
    num = num or 0
    txt = f"{num:>6.3f}"
    if maxval > 0 and num > maxval:
        return bcolored(txt, color)
    return txt

def error(txt):
    print(bcolored(f"Error: {txt}", "red"))
    exit()

def log(txt, color = None, log = True):
    if log:
        print(bcolored(txt, color) if color is not None else bold(txt))

# File processing
# ----------------------------------------------------------------------------

# Delete a list of files
def rm(files):
    for f in files:
        os.remove(f)

# Make directory
def mkdir(d):
    os.makedirs(d, exist_ok = True)

# Save a numpy file
def save_npy(mat, filename):
    with open(filename, 'wb') as f:
        np.save(f, mat)

# Saves a list of numpy arrays with ordering and according to a path template
def save_npys(npys, path, verbose = False, offset = 0):
    npys = enumerate(npys)
    if verbose:
        npys = tqdm(list(npys))
    for i, npy in npys:
        save_npy(npy, path % (offset + i))

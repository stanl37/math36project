from lenia import Lenia

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from matplotlib import animation
from IPython.display import HTML
from ipywidgets import interact, widgets
import warnings
import time
import multiprocessing
import concurrent.futures
import random

NUM_LENIAS = 50
DELTA_TYPES = [0, 1, 2, 3]

KERNEL_TYPE = 0
SIZE = 128
GENS = 100

FILENAME = 'data_test.npy'
pairs = np.load(FILENAME)
idxs_to_run = np.where(pairs[:,2] == -1)[0]
np.random.shuffle(idxs_to_run)
my_idxs = idxs_to_run[:NUM_LENIAS]

def create_lenias(DELTA_TYPE):
  lenias = []
  for idx in my_idxs:
    (mu, sigma, factor) = pairs[idx]

    lenia = Lenia(SIZE)
    lenia.clear_world()
    lenia.load_cells(0)
    lenia.add_cells()
    lenia.set_params(kernel_type=KERNEL_TYPE, delta_type=DELTA_TYPE)
    lenia.set_params(mu=mu + random.uniform(-0.02, 0.02), sigma=sigma + random.uniform(-0.005, 0.005))
    lenia.calc_kernel()

    lenias.append(lenia)
  return lenias

list_of_lenia_lists = []
for DELTA_TYPE in DELTA_TYPES:
  list_of_lenia_lists.append(create_lenias(DELTA_TYPE))


for idx, lenias in enumerate(list_of_lenia_lists):
  
  start = time.time()
  for lenia in lenias:
    lenia.run(GENS)
  end = time.time()
  print(f"DeltaFunc {idx}\trun({GENS})\tsize: {lenias[0].SIZE}.\ttime: {round(end-start, 4)}s")

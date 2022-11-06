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

NUM_LENIAS = 50
SIZES = [32, 64, 128, 256]
GENS = 100

FILENAME = 'data_test.npy'
pairs = np.load(FILENAME)
idxs_to_run = np.where(pairs[:,2] == -1)[0]
np.random.shuffle(idxs_to_run)

my_idxs = idxs_to_run[:NUM_LENIAS]

def create_lenias(SIZE):
  lenias = []
  for idx in my_idxs:
    (mu, sigma, factor) = pairs[idx]

    lenia = Lenia(SIZE)
    lenia.clear_world()
    lenia.load_cells(0)
    lenia.add_cells()
    lenia.set_params(kernel_type=0, delta_type=1)
    lenia.set_params(mu=mu, sigma=sigma)
    lenia.calc_kernel()

    lenia.run(5)

    lenias.append(lenia)
  return lenias

list_of_lenia_lists = []
for SIZE in SIZES:
  list_of_lenia_lists.append(create_lenias(SIZE))


for lenias in list_of_lenia_lists:
  
  # singlethread
  start = time.time()
  for lenia in lenias:
    lenia.run(GENS)
  end = time.time()
  print(f"singlethreaded\trun({GENS})\tsize: {lenias[0].SIZE}.\ttime: {round(end-start, 4)}s")

  #multithread
  start = time.time()
  exe = concurrent.futures.ThreadPoolExecutor()
  futures = []
  for lenia in lenias:
    futures.append(exe.submit(lenia.run, GENS))
  for future in concurrent.futures.as_completed(futures):
    result = future.result()
  end = time.time()
  print(f"multithreaded\trun({GENS})\tsize: {lenias[0].SIZE}.\ttime: {round(end-start, 4)}s")

  print()


"""
start = time.time()
for lenia in lenias:
  lenia.calc_once()
end = time.time()
print(f"singlethreaded calc_once: {round(end-start, 4)}s")

start = time.time()
exe = concurrent.futures.ThreadPoolExecutor()
futures = []
for lenia in lenias:
  futures.append(exe.submit(lenia.calc_once))
for future in concurrent.futures.as_completed(futures):
  result = future.result()
end = time.time()
print(f"multithreaded calc_once ({exe._max_workers} workers): {round(end-start, 4)}s")
"""

# fig, ax = plt.subplots()
# lenias[0].draw_world(lenias[0].world, ax=ax)
# plt.show()
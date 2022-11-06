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
import pyfftw
import multiprocessing
import concurrent.futures

lenia = Lenia()
lenia.clear_world()
lenia.load_cells(0)
lenia.multiply_cells(5)
lenia.add_cells()
lenia.set_params(
    kernel_type=0,
    delta_type=1
  )
lenia.calc_kernel()

worlds = []
for i in range(100):
  lenia.calc_once()
  worlds.append(lenia.world)

if __name__ == "__main__":
  start = time.time()
  for world in worlds:
    np.fft.fft2(world)
  end = time.time()
  print(f"numpy: {round(end-start, 4)}s")


if False:
  start = time.time()
  for world in worlds:
    np.fft.fft2(world)
  end = time.time()
  print(f"numpy: {round(end-start, 4)}s")

  start = time.time()
  exe = concurrent.futures.ThreadPoolExecutor()
  futures = []
  for world in worlds:
    exe.submit(np.fft.fft2, world)
  for future in concurrent.futures.as_completed(futures):
    result = future.result()
  end = time.time()
  print(f"numpy multithreaded ({exe._max_workers} workers): {round(end-start, 4)}s")

  start = time.time()
  exe = concurrent.futures.ThreadPoolExecutor()
  futures = []
  for world in worlds:
    pyfftw.interfaces.numpy_fft.fft2(world)
  for future in concurrent.futures.as_completed(futures):
    result = future.result()
  end = time.time()
  print(f"pyfftw multithreaded ({exe._max_workers} workers): {round(end-start, 4)}s")

  start = time.time()
  exe = concurrent.futures.ThreadPoolExecutor()
  futures = []
  for world in worlds:
    pyfftw.interfaces.numpy_fft.fft2(world, threads=multiprocessing.cpu_count())
  for future in concurrent.futures.as_completed(futures):
    result = future.result()
  end = time.time()
  print(f"threaded-pyfftw multithreaded ({exe._max_workers} workers): {round(end-start, 4)}s")


  # throws an error for some reason, but way way slower than numpy multithread
  """
  start = time.time()
  pool = multiprocessing.Pool()
  result = pool.map(np.fft.fft2, worlds)
  end = time.time()
  print(f"numpy multiprocessing ({pool._processes} workers): {round(end-start, 4)}s")
  """

  """
  start = time.time()
  exe = concurrent.futures.ProcessPoolExecutor()
  futures = []
  for world in worlds:
    exe.submit(np.fft.fft2, world)
    
  for future in concurrent.futures.as_completed(futures):
    result = future.result()
  end = time.time()
  print(f"numpy ProcessPoolExe ({exe._max_workers} workers): {round(end-start, 4)}s")
  """



"""
for world in worlds:
  pyfftw.byte_align(world)

start = time.time()
for world in worlds:
  pyfftw.interfaces.numpy_fft.fft2(world)
end = time.time()
print(f"pyfttw: {round(end-start, 4)}s")

start = time.time()
for world in worlds:
  pyfftw.interfaces.numpy_fft.fft2(world, threads=multiprocessing.cpu_count())
end = time.time()
print(f"pyfttw threaded: {round(end-start, 4)}s")
"""



"""
lenia = Lenia()
lenia.clear_world()
lenia.load_cells(0)
lenia.multiply_cells(5)
lenia.add_cells()
lenia.set_params(
    kernel_type=0,
    delta_type=1
  )
lenia.calc_kernel()

start = time.time()
for i in range(100):
  lenia.calc_once(is_update=True)
end = time.time()
print(f"runtime: {end-start}s")

fig, ax = plt.subplots()
lenia.draw_world(lenia.world, title="World", ax=ax)
# plt.show()

import timeit
timeit.Timer(lambda: np.fft.fft2(world)).timeit(number=100)
timeit.Timer(lambda: pyfftw.interfaces.numpy_fft.fft2(world)).timeit(number=100)
"""
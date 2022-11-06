from leniacopy import Lenia

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from matplotlib import animation
from IPython.display import HTML
from ipywidgets import interact, widgets
import warnings

import time



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
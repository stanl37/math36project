from lenia import Lenia
import numpy as np
import time
import datetime

# CONSTANTS
# delta types:
#   0: exponential approx of gaussian (slow, do not use)
#   1: gaussian
#   2: integral rectangular
#   3: gaussian roots rectangular (no reason to use this)
DELTA_TYPE = 1
FILENAME = 'data_test.npy'

GENS = 100
SMALL_WORLD_SIZE = 32
BIG_WORLD_SIZE = 64
WRITE_INTERVAL = 100  # how many sims to run before saving to file

def sim(mu, sigma):

  # run a small world. if the  mass = 0 here then factor = 0, we are done
  # then run a big world
  # factor = big world mass / small world mass

  # normal world
  lenia = Lenia(SMALL_WORLD_SIZE)
  lenia.clear_world()
  lenia.load_cells(0)
  # lenia.multiply_cells(3)
  lenia.add_cells()
  lenia.set_params(kernel_type=0, delta_type=DELTA_TYPE)
  lenia.set_params(mu=mu, sigma=sigma)
  lenia.calc_kernel()

  for i in range(GENS):
      lenia.calc_once()
      if lenia.mass() == 0:
        break
  small_mass = lenia.mass()

  if small_mass == 0:
    factor = 0
    return factor  # assumption that if it dies in the small world, it will die in the big world

  # big world
  lenia = Lenia(BIG_WORLD_SIZE)
  lenia.clear_world()
  lenia.load_cells(0)
  # lenia.multiply_cells(3)
  lenia.add_cells()
  lenia.set_params(kernel_type=0, delta_type=DELTA_TYPE)
  lenia.set_params(mu=mu, sigma=sigma)
  lenia.calc_kernel()

  for i in range(GENS):
      lenia.calc_once()
      if lenia.mass() == 0:
        break
  big_mass = lenia.mass()

  factor = big_mass/small_mass
  return factor


if __name__ == "__main__":
  # SETUP
  pairs = np.load(FILENAME)
  idxs_to_run = np.where(pairs[:,2] == -1)[0]
  np.random.shuffle(idxs_to_run)
  num_idxs_to_run = len(idxs_to_run)
  print("idxs:", idxs_to_run)
  print("num idxs:", num_idxs_to_run)
  input("PRESS ENTER TO BEGIN...")

  # SETUP 2
  tic = time.time()
  counter = 0
  avg_time = 0
  num_times = 0

  # RUN
  for idx in idxs_to_run:
    pair = pairs[idx]
    (mu, sigma, factor) = pair
    factor = sim(mu, sigma)
    pairs[idx][2] = factor
    counter += 1

    # write to file every once in a while
    if counter >= WRITE_INTERVAL:
      toc = time.time()
      np.save(FILENAME, pairs)
      write_time = toc-tic
      num_idxs_to_run -= counter
      component_oldavg = (avg_time) * (num_times / (num_times + 1))
      component_newavg = (write_time) * (1 / (num_times + 1))
      avg_time = component_oldavg + component_newavg
      num_times += 1
      remaining_writes = round(num_idxs_to_run / WRITE_INTERVAL)
      estimated_remaining_runtime = str(datetime.timedelta(seconds=avg_time * remaining_writes))

      print(f"[WRITE {num_times} ({num_times * WRITE_INTERVAL} sims)] -> {FILENAME} \n\t t: {round(write_time, 3)}s, avg: {round(avg_time, 3)}s \n\t remaining writes: {remaining_writes} ({num_idxs_to_run} sims) \n\t est. time to completion: {estimated_remaining_runtime}")
      
      counter = 0
      tic = time.time()
  
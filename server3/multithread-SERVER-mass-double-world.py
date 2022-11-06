from lenia import Lenia
import numpy as np
import time
import datetime
import concurrent.futures

# NOTE: MULTITHREADING IS NOT ALWAYS MORE EFFICIENT.
# if lenia worlds are 128 and 256 (or greater), it is faster, for my system


# CONSTANTS
# delta types:
#   0: exponential approx of gaussian (slow, do not use)
#   1: gaussian
#   2: integral rectangular
#   3: gaussian roots rectangular (no reason to use this)
DELTA_TYPE = 1
FILENAME = 'data_test.npy'

GENS = 100
SMALL_WORLD_SIZE = 128
BIG_WORLD_SIZE = 256
PACKET_SIZE = 50
WRITE_INTERVAL = 100000  # how many sims to run before saving to file

def sim(packet):

  # create simulations dict
  #   key: index (from packet)
  #   value: value_template (below)
  value_template = {
    "small_lenia": None,
    "small_mass": None,
    "big_lenia": None,
    "factor": None
    }
  simulations = dict.fromkeys(packet, value_template)
  
  # add in small world objects
  for idx in simulations:

    (mu, sigma, factor) = pairs[idx]

    lenia = Lenia(SMALL_WORLD_SIZE)
    lenia.clear_world()
    lenia.load_cells(0)
    lenia.add_cells()
    lenia.set_params(kernel_type=0, delta_type=DELTA_TYPE)
    lenia.set_params(mu=mu, sigma=sigma)
    lenia.calc_kernel()

    simulations[idx]["small_lenia"] = lenia

  # singlethreaded run small Lenias
  lenias = []
  for idx in simulations:
    lenia = simulations[idx]["small_lenia"]
    lenias.append(lenia)
  
  start = time.time()
  for lenia in lenias:
    lenia.run(GENS)
  end = time.time()
  print(f"singlethreaded\tsmall\trun\t{len(packet)} sims.\t{round(end-start, 4)}s")

  # multithreaded run small Lenias
  lenias = []
  for idx in simulations:
    lenia = simulations[idx]["small_lenia"]
    lenias.append(lenia)

  start = time.time()
  exe = concurrent.futures.ThreadPoolExecutor()
  futures = []
  for lenia in lenias:
    futures.append(exe.submit(lenia.run, 100))
  for future in concurrent.futures.as_completed(futures):
    result = future.result()
  end = time.time()
  print(f"multithreaded\tsmall\trun\t{len(packet)} sims.\t{round(end-start, 4)}s")

  # updating factors (small worlds with 0)
  for idx in simulations:
    lenia = simulations[idx]["small_lenia"]
    if lenia.mass() == 0:
      simulations[idx]["factor"] = 0
    else:
      simulations[idx]["small_mass"] = lenia.mass()

  # add in big world objects, only for those that need it
  big_lenias = []
  for idx in simulations:
    # avoid those with factors already set
    if simulations[idx]["factor"] == 0:
      continue

    (mu, sigma, factor) = pairs[idx]

    lenia = Lenia(BIG_WORLD_SIZE)
    lenia.clear_world()
    lenia.load_cells(0)
    lenia.add_cells()
    lenia.set_params(kernel_type=0, delta_type=DELTA_TYPE)
    lenia.set_params(mu=mu, sigma=sigma)
    lenia.calc_kernel()

    simulations[idx]["big_lenia"] = lenia

  # singlethreaded run big Lenias
  counter = 0
  start = time.time()
  for idx in simulations:
    # avoid those with factors already set
    if simulations[idx]["factor"] == 0:
      continue
    lenia = simulations[idx]["big_lenia"]
    lenia.run(GENS)
    counter += 1
  end = time.time()
  if counter != 0:
    print(f"singlethreaded\tbig\trun\t{counter} sims.\t{round(end-start, 4)}s")
    counter = 0
  

  # multithreaded run small Lenias
  counter = 0
  lenias = []
  for idx in simulations:
    if simulations[idx]["factor"] == 0:
      continue
    lenia = simulations[idx]["big_lenia"]
    lenias.append(lenia)
  start = time.time()
  exe = concurrent.futures.ThreadPoolExecutor()
  futures = []
  for lenia in lenias:
    futures.append(exe.submit(lenia.run, 100))
  for future in concurrent.futures.as_completed(futures):
    result = future.result()
  end = time.time()
  if counter != 0:
    print(f"multithreaded\tbig\trun\t{len(packet)} sims.\t{round(end-start, 4)}s")
    counter = 0

  # updating factors (small worlds with 0)
  for idx in simulations:

    # avoid those with factors already set
    if simulations[idx]["factor"] == 0:
      continue

    lenia = simulations[idx]["big_lenia"]
    simulations[idx]["factor"] = lenia.mass() / simulations[idx]["small_mass"]

  # make list of (idx, factor) tuples
  idx_factor_tuple_list = []
  for idx in simulations: 
    idx_factor_tuple_list.append((idx, simulations[idx]["factor"]))
  
  return idx_factor_tuple_list

if __name__ == "__main__":

  # SETUP
  pairs = np.load(FILENAME)
  idxs_to_run = np.where(pairs[:,2] == -1)[0]
  np.random.shuffle(idxs_to_run)
  num_idxs_to_run = len(idxs_to_run)
  print("idxs:", idxs_to_run)
  print("num idxs:", num_idxs_to_run)

  packets = np.array_split(idxs_to_run, np.arange(PACKET_SIZE, len(idxs_to_run), PACKET_SIZE))
  print("num packets:", len(packets))
  print("packet size:", PACKET_SIZE)

  input("PRESS ENTER TO BEGIN...")

 

  # SETUP 2
  tic = time.time()
  counter = 0
  avg_time = 0
  num_times = 0

  # RUN
  for packet in packets:

    idxfactor_tuple_list = sim(packet)

    for idxfactor_tuple in idxfactor_tuple_list:
      (idx, factor) = idxfactor_tuple
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
  
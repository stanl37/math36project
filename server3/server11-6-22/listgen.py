import numpy as np

# INPUTS
FILENAME = 'data_rectangular.npy'
mu = 0.156
sigma = 0.0224
mu_range_multipliers = (0.65, 1.3)
sigma_range_multipliers = (0.65, 1.3)
num_mu_variations = 400
num_sigma_variations = 400

# CALCULATED VALS
mu_low = mu * mu_range_multipliers[0]
mu_high = mu * mu_range_multipliers[1]
sigma_low = sigma * sigma_range_multipliers[0]
sigma_high = sigma * sigma_range_multipliers[1]
tot_variations = num_mu_variations * num_sigma_variations

# generate a list of mu-sigma pairs to work through
pairs = []
for mu in np.linspace(mu_low, mu_high, num_mu_variations):
  for sigma in np.linspace(sigma_low, sigma_high, num_sigma_variations):
    pairs.append([mu, sigma, -1])

print(f"created list of {tot_variations} variations:")
print(f"\t{num_mu_variations} mus: {mu_low} to {mu_high}")
print(f"\t{num_sigma_variations} sigmas: {sigma_low} to {sigma_high}")
pairs = np.array(pairs)

# write to file 
import os
if os.path.exists(FILENAME):
  os.remove(FILENAME)
  print(f"removed old {FILENAME}")

np.save(FILENAME, pairs)
print(f"wrote {tot_variations} variations to {FILENAME}")
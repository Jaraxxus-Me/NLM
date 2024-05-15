import numpy as np
import random
import pickle as pkl

data_path = "data/LogiCity/transfer/med_400.pkl"

with open(data_path, 'rb') as f:
    raw_data = pkl.load(f)

np.random.seed(0)
np.random.shuffle(raw_data)

rand_data_path = "data/LogiCity/transfer/med_400_rand0.pkl"
with open(rand_data_path, 'wb') as f:
    pkl.dump(raw_data, f)
    print(f"Randomized data saved to {rand_data_path}")

np.random.seed(1)
np.random.shuffle(raw_data)

rand_data_path = "data/LogiCity/transfer/med_400_rand1.pkl"
with open(rand_data_path, 'wb') as f:
    pkl.dump(raw_data, f)
    print(f"Randomized data saved to {rand_data_path}")

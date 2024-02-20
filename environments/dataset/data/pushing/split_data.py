import os
import pickle
import random
import glob
from agents.utils.sim_path import sim_framework_path

random.seed(42)

train_files = []
eval_files = []

gg_rr = os.listdir("gg_rr")
gr_rg = os.listdir("gr_rg")
rg_gr = os.listdir("rg_gr")
rr_gg = os.listdir("rr_gg")

all_data = [gg_rr, gr_rg, rg_gr, rr_gg]

random.shuffle(gg_rr)
random.shuffle(gr_rg)
random.shuffle(rg_gr)
random.shuffle(rr_gg)

for data in all_data:
    train_files += data[50:]
    eval_files += data[:50]

with open("train_files.pkl", "wb") as f:
    pickle.dump(train_files, f)

with open("eval_files.pkl", "wb") as f:
    pickle.dump(eval_files, f)
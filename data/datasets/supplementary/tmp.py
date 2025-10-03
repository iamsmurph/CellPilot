#%%
import pandas as pd
import matplotlib.pyplot as plt
#%%
pth = "/data/rbg/users/seanmurphy/CellPilot/data/datasets/supplementary/chip_tf_jaccard_top100to400_pairs.csv"
df = pd.read_csv(pth)
#%%
plt.hist(df.jaccard)
# %%

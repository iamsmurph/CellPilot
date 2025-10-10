# %%
import pandas as pd
import matplotlib.pyplot as plt
#%%

pth = "/data/rbg/users/seanmurphy/CellPilot/data/datasets/train/tf_pairs_templates_jaccard.csv"
df = pd.read_csv(pth)

#%%
df.head()
# %%
plt.hist(df.jaccard, bins=100)
plt.axvline(.14, color="r", label="Best so far (step 6)")
plt.xlabel("Jaccard Score")
plt.ylabel("Frequency")
plt.legend()
# %%
plt.hist([len(x) for x in df.template.values])
# %%

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import umap

# ----------------------------------------------------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------------------------------------------------

TRAINING_DATASET = "./data/out.csv"
df = pd.read_csv(TRAINING_DATASET, index_col=False, low_memory=False)
X = np.array(df.drop(['class'], 1))
Y = np.array(df['class'])

# ----------------------------------------------------------------------------------------------------------------------
# UMAP
# ----------------------------------------------------------------------------------------------------------------------

embedding = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=3).fit_transform(X)

# ----------------------------------------------------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], c=Y, cmap="rainbow", s=1)
plt.setp(ax, xticks=[], yticks=[])
plt.title("gesture data embedded into two dimensions by UMAP", fontsize=10)
plt.show()

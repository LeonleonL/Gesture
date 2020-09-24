from numpy import array
from numpy import corrcoef
import pandas as pd
import numpy as np

TRAINING_DATASET = "./data/out.csv"
df = pd.read_csv(TRAINING_DATASET, index_col=False)

# ----------------------------------------------------------------------------------------------------------------------

X = np.array(df.drop(['class'], 1))
Y = np.array(df['class'])

x = array(X[0])
y = array(X[1])

Sigma = corrcoef(x,y)
print(Sigma)
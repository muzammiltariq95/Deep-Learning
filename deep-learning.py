import pandas as pd

data = pd.read_csv('bank_note_data.csv')

data.head()

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Class',data=data)
plt.show()
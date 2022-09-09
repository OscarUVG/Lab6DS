import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dataTweets.csv")

print("--- Vista inicial de los datos ---")
print(df.head())

print("--- Conteo de valores NA ---")
print(df.isnull().sum())

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
from sklearn.datasets import load_digits
import pandas as pd

mices = pd.read_csv("Mices_Proteins.csv",on_bad_lines="skip",sep=";")

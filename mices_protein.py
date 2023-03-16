from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
from sklearn.datasets import load_digits
import pandas as pd

mices = pd.read_csv("Mices_Proteins.csv",on_bad_lines="skip",sep=";")

from sklearn import preprocessing
x = mices.copy()
scaler = preprocessing.MinMaxScaler()
x[["MouseID	DYRK1A_N","ITSN1_N","BDNF_N","NR1_N","NR2A_N","pAKT_N","pBRAF_N","pCAMKII_N","pCREB_N","pELK_N","pERK_N","pJNK_N","PKCA_N","pMEK_N","pNR1_N","pNR2A_N","pNR2B_N","pPKCAB_N","pRSK_N","AKT_N","BRAF_N","CAMKII_N","CREB_N","ELK_N","ERK_N","GSK3B_N","JNK_N","MEK_N","TRKA_N","RSK_N","APP_N","Bcatenin_N","SOD1_N","MTOR_N","P38_N","pMTOR_N","DSCR1_N","AMPKA_N","NR2B_N","pNUMB_N","RAPTOR_N","TIAM1_N","pP70S6_N","NUMB_N"	P70S6_N	pGSK3B_N	pPKCG_N	CDK5_N	S6_N	ADARB1_N	AcetylH3K9_N	RRP1_N	BAX_N	ARC_N	ERBB4_N	nNOS_N	Tau_N	GFAP_N	GluR3_N	GluR4_N	IL1B_N	P3525_N	pCASP9_N	PSD95_N	SNCA_N	Ubiquitin_N	pGSK3B_Tyr216_N	SHH_N	BAD_N	BCL2_N	pS6_N	pCFOS_N	SYP_N	H3AcK18_N	EGR1_N	H3MeK4_N	CaNA_N]] = scaler.fit_transform(x[["Control","Genotype","Treatment","Behavior","class"]])
import numpy as np
import pandas as pd
import sys
import pickle
import plotly.graph_objects as go
import plotly.offline as pyo
import matplotlib as mpl
import matplotlib.pyplot as plt
import SolveLinSys
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CubicSpline
from matplotlib.backends.backend_pdf import PdfPages
import os
import argparse
from support import *
# from matplotlib.ticker import StrMethodFormatter, NullFormatter
# ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
# ax.yaxis.set_minor_formatter(NullFormatter())

mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["figure.figsize"] = (32,30)
mpl.rcParams["font.size"] = 15
mpl.rcParams["legend.frameon"] = False
mpl.style.use('classic')
mpl.rcParams["lines.linewidth"] = 5


import argparse 
sys.stdout.flush()
reporterror = True


parser = argparse.ArgumentParser(description="xi_r values")
parser.add_argument("--rho", type=float)
parser.add_argument("--epsilon", type=float)
parser.add_argument("--fraction", type=float)
parser.add_argument("--maxiter", type=float)
parser.add_argument("--dataname",type=str,default="LinearNDInterpolator")
parser.add_argument("--figname",type=str,default="LinearNDInterpolator")

args = parser.parse_args()


#==============================================================================#
#    PARAMETERS
#==============================================================================#


# (1) Baseline model
alpha_z_hat = 0.0
kappa_hat = 0.014

alpha_c_hat = 0.484       # consumption intercept (estimated) 
beta_hat = 1.0
sigma_c = [0.477, 0.0  ]   # consumption exposure (= exposure of single capital)
sigma_z =  [0.011, 0.025]
rho = args.rho

delta = 0.002
A_cap = 0.05

phi = 28.0

# ell=0.05
ell = 3.82

JJ=201
# rmax =  18.0
# rmin = -rmax       #-25.0 #-rmax
zmax = 0.05
zmin = -zmax

FC_Err = 1
epoch = 0
max_iter = args.maxiter
tol = 1e-6
# fraction = 0.1
# epsilon = 0.01
fraction = args.fraction
epsilon = args.epsilon


Data_Dir = "./data/"+args.dataname+"/"

model_simul_dir_post = Data_Dir + "result_rho_{}_eps_{}_frac_{}".format(rho,epsilon,fraction)


res = pickle.load(open(model_simul_dir_post, "rb"))

W1 = res["W1"]
d_star = res["d_star"]
h1_star = res["h1_star"]
hz_star = res["hz_star"]

V0 = res["V0"]


Fig_Dir = "./figure/"+args.figname+"/"

os.makedirs(Fig_Dir, exist_ok=True)

print("max,min={},{}".format(d_star[:,2,2].max(),d_star[:,2,2].min()))

plt.plot(W1,d_star[:,2,2],label="$d$")
# plt.plot(W1,0.0317914761536931*np.ones(d_star[:,2,2].shape),label=r"$d$: baseline",linestyle='--',color='red')
# print((d_star[-1,2,2]-d_star[0,2,2])/2)
# print(h1_star.max())
# print(hz_star.max())
plt.legend()
plt.xlabel('z')
# plt.ylabel('$\%$ of GDP')
plt.title('Investment-Capital Ratio')  
plt.xlim([-0.05, 0.05])
plt.ylim([0.010,0.030])

plt.savefig(Fig_Dir+"d_rho_{}.png".format(rho))
plt.close()


plt.plot(W1,h1_star[:,2,2],label="$h1$")
plt.plot(W1,hz_star[:,2,2],label="$hz$")
# plt.plot(W1,-0.003048700579899253*np.ones(h1_star[:,2,2].shape),label=r"$h1$: baseline",linestyle='--')
# plt.plot(W1,-0.004090678107421712*np.ones(h1_star[:,2,2].shape),label="$hz$: baseline",linestyle='--')
plt.legend()
plt.xlabel('z')
# plt.ylabel('$\%$ of GDP')
plt.title('Distortion')  
plt.xlim([-0.05, 0.05])
plt.ylim([-0.15, -0.05])
plt.savefig(Fig_Dir+"h_rho_{}.png".format(rho))
plt.close()

print("d0={}".format(d_star[int(len(W1)/2),2,2]))
print("h10={}".format(h1_star[int(len(W1)/2),2,2]))
print("hz0={}".format(hz_star[int(len(W1)/2),2,2]))

plt.plot(W1,V0[:,2,2],label="V")
plt.legend()
plt.xlabel('z')
# plt.ylabel('$\%$ of GDP')
plt.title('Value Function')  
plt.xlim([-0.05, 0.05])

plt.ticklabel_format(style='plain',useMathText=False)    # prevents scientific notation
plt.savefig(Fig_Dir+"VF_rho_{}.png".format(rho))
plt.close()


W1_min = zmin
W1_max = zmax
hW1 = 0.01
W1 = np.arange(W1_min, W1_max+hW1, hW1)
nW1 = len(W1)


dVdW1= finiteDiff_3D(V0, 0, 1, hW1)
print("dVdz0={}".format(dVdW1[int(len(W1)/2),2,2]))
print("dVdzmax,min={},{}".format(dVdW1[:,2,2].max(),dVdW1[:,2,2].min()))

plt.plot(W1,dVdW1[:,2,2],label="dV")
plt.legend()
plt.xlabel('z')
# plt.ylabel('$\%$ of GDP')
plt.title('Derivatives of Value Function')  
plt.xlim([-0.05, 0.05])
plt.ylim([55,65])
plt.ticklabel_format(style='plain')    # prevents scientific notation

# plt.ticklabel_format(style='plain')    # to prevent scientific notation.
# ax = plt.gca()
# ax.get_xaxis().get_major_formatter().set_useOffset(False)
plt.savefig(Fig_Dir+"dVF_rho_{}.png".format(rho))
plt.close()


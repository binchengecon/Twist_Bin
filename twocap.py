# from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.pyplot as plt
# import matplotlib as mpl
import pickle
import time
# import petsclinearsystem
import petsclinearsystemXDiff
from petsc4py import PETSc
import petsc4py
import os
import sys
import numpy as np
from support import *
import argparse 
sys.stdout.flush()
petsc4py.init(sys.argv)
reporterror = True


parser = argparse.ArgumentParser(description="xi_r values")
parser.add_argument("--rho", type=float)
parser.add_argument("--epsilon", type=float)
parser.add_argument("--fraction", type=float)
parser.add_argument("--maxiter", type=float)
parser.add_argument("--dataname",type=str)
parser.add_argument("--figname",type=str)

args = parser.parse_args()


#==============================================================================#
#    PARAMETERS
#==============================================================================#


# (1) Baseline model
# alpha_z_hat = 0.0
# kappa_hat = 0.014
# alpha_k_hat = -0.88       # -1.279
# alpha_c_hat = 0.484      
# beta_hat = 1.0
# sigma_c = np.array([0.477, 0.0 ])   # consumption exposure (= exposure of single capital)
# sigma_z = np.array([0.011, 0.025])
# rho = args.rho
# delta = 0.0025 # 0.0025
# A_cap = 0.0288 # 0.0288
# phi = 28.0
# theta1 = 1/88
# theta2 = 88
# # ell=0.05
# ell = 3.82
# zmax = 0.05
# zmin = -zmax

scale = 1.754 # 1.307
sigma_1 = scale * np.array([.477, 0, 0])
sigma_2 = scale * np.array([ 0, .477, 0])
sigma_z = np.array([ .011*np.sqrt(5), .011*np.sqrt(5) , .025])


delta = 0.0025 
A1_cap = 0.0288 # 0.05 
A2_cap = 0.0288 # 0.05
phi1 = 28.0
phi2 = 28.0

alpha_z_hat = 0.0
beta_hat = 1.0
kappa_hat = 0.014

rho = args.rho
ell = 3.82


lmin = -18
lmax = 18

zmin = -1
zmax = 1




W1_min = lmin
W1_max = lmax
hW1 = 0.1
W1 = np.arange(W1_min, W1_max+hW1, hW1)
nW1 = len(W1)

W2_min = zmin
W2_max = zmax
hW2 = 0.1
W2 = np.arange(W2_min, W2_max+hW2, hW2)
nW2 = len(W2)


W3_min = 0.0
W3_max = 1.0
hW3 = 0.5
W3 = np.arange(W3_min, W3_max+hW3, hW3)
nW3 = len(W3)

(W1_mat, W2_mat, W3_mat) = np.meshgrid(W1, W2, W3, indexing='ij')
stateSpace = np.hstack([W1_mat.reshape(-1, 1, order='F'),
                       W2_mat.reshape(-1, 1, order='F'), W3_mat.reshape(-1, 1, order='F')])

W1_mat_short = W1_mat[1:-1,:,:]

W1_mat_1d = W1_mat.ravel(order='F')
W2_mat_1d = W2_mat.ravel(order='F')
W3_mat_1d = W3_mat.ravel(order='F')

lowerLims = np.array([W1.min(), W2.min(), W3.min()], dtype=np.float64)
upperLims = np.array([W1.max(), W2.max(), W3.max()], dtype=np.float64)



print("Grid dimension: [{}, {}, {}]\n".format(nW1, nW2, nW3))
print("Grid step: [{}, {}, {}]\n".format(hW1, hW2, hW3))

# V0 = -(W1_mat)**2 +2
# V0 = W1_mat**2 +W2_mat**2 +5
V0 = W2_mat**2 + 5
# V0 = -W1_mat**2

r_mat = np.exp(W1_mat)/(1+np.exp(W1_mat))

d1_star = 0.0025*np.ones(W1_mat.shape)
d2_star = 0.0025*np.ones(W1_mat.shape)
h1_star = -0.025*np.zeros(W1_mat.shape)
h2_star = -0.025*np.zeros(W1_mat.shape)
hz_star = -0.025*np.zeros(W1_mat.shape)



dVec = np.array([hW1, hW2, hW3])
increVec = np.array([1, nW1, nW1*nW2], dtype=np.int32)
# increVec = np.array([1, nW1_short, nW1_short*nW2], dtype=np.int32)

petsc_mat = PETSc.Mat().create()
petsc_mat.setType('aij')
# petsc_mat.setSizes([nW1_short * nW2 * nW3, nW1_short * nW2 * nW3])
petsc_mat.setSizes([nW1 * nW2 * nW3, nW1 * nW2 * nW3])
petsc_mat.setPreallocationNNZ(13)
petsc_mat.setUp()
ksp = PETSc.KSP()
ksp.create(PETSc.COMM_WORLD)
ksp.setType('bcgs')
ksp.getPC().setType('ilu')
ksp.setFromOptions()

FC_Err = 1
epoch = 0
max_iter = args.maxiter
tol = 1e-6
# fraction = 0.1
# epsilon = 0.01
fraction = args.fraction
epsilon = args.epsilon

while FC_Err > tol and epoch < max_iter:
    start_eps = time.time()



    # dVdW1= finiteDiff_3D(V0, 0, 1, hW1)
    # ddVddW1= finiteDiff_3D(V0, 0, 2, hW1)
    dVdW1= finiteDiff_3D2(V0, 0, 1, hW1)
    ddVddW1= finiteDiff_3D2(V0, 0, 2, hW1)
    # dZ = dW1
    dVdW2 = finiteDiff_3D(V0, 1, 1, hW2)
    ddVddW2 = finiteDiff_3D(V0, 1, 2, hW2)
    
    ddVdW1dW2 = finiteDiff_3D2(dVdW1, 1, 1, hW2)
    # # # dY = dW2
    # dVdW3 = finiteDiff_3D(V0, 2, 1, hW3)
    # ddVddW3 = finiteDiff_3D(V0, 2, 2, hW3)
    # # dZ = dW2

    # need to change the control optimizatio completely due to corner solution of c

    # if np.any(dVdW1+ddVddW1 * r * sigma**2 >= 0):
    #     print("warning\n")
    
##########################investment-capital ratio#############
    d1_star[d1_star>=A1_cap] = A1_cap-0.001
    d2_star[d2_star>=A2_cap] = A2_cap-0.001
    
    mc = (delta*np.exp(V0)**(rho-1))   /  ( (1-r_mat)*(A1_cap-d1_star) + (r_mat)*(A2_cap-d2_star) )**(rho)
    
    # mc[mc<=1]=1+1e-16
    
    d1_new = mc / (1-r_mat-dVdW1)
    d1_new = d1_new -1
    d1_new = d1_new/(-phi1)

    d2_new = mc / (r_mat+dVdW1)
    d2_new = d2_new -1
    d2_new = d2_new/(-phi2)


    
########################## distortion #############

    h1_new = -(.01*(sigma_1[0]*(1-r_mat)+sigma_2[0]*r_mat)+.01*dVdW1*(sigma_2[0]-sigma_1[0])+sigma_z[0]*dVdW2)/ell
    h2_new = -(.01*(sigma_1[1]*(1-r_mat)+sigma_2[1]*r_mat)+.01*dVdW1*(sigma_2[1]-sigma_1[1])+sigma_z[1]*dVdW2)/ell
    hz_new = -(.01*(sigma_1[2]*(1-r_mat)+sigma_2[2]*r_mat)+.01*dVdW1*(sigma_2[2]-sigma_1[2])+sigma_z[2]*dVdW2)/ell
    # c_new[c_new<=1e-16] = 1e-16

    d1 = d1_new * fraction + d1_star*(1-fraction)
    d2 = d2_new * fraction + d2_star*(1-fraction)
    h1 = h1_new * fraction + h1_star*(1-fraction)
    h2 = h2_new * fraction + h2_star*(1-fraction)
    hz = hz_new * fraction + hz_star*(1-fraction)

    d1[d1>=A1_cap] = A1_cap-0.001
    d2[d2>=A2_cap] = A2_cap-0.001
    
    
    h1[h1>=-1e-16] = -1e-16
    h2[h2>=-1e-16] = -1e-16
    hz[hz>=-1e-16] = -1e-16


    psi1 = d1 - phi1/2*d1**2+.01*(alpha_z_hat+beta_hat*W2_mat)
    psi2 = d2 - phi2/2*d2**2+.01*(alpha_z_hat+beta_hat*W2_mat)
    
    A = np.zeros(W1_mat.shape)
    B_1 = psi2-psi1 - (.01)**2/2 *  (np.sum(sigma_2**2)-np.sum(sigma_1**2))
    B_1 += .01*( ((1-r_mat)*sigma_1[0]+ r_mat*sigma_2[0])*h1  + ((1-r_mat)*sigma_1[1]+ r_mat*sigma_2[1])*h2 + ((1-r_mat)*sigma_1[2]+ r_mat*sigma_2[2])*hz)
    B_2 = -kappa_hat*W2_mat + sigma_z[0]*h1 + sigma_z[1]*h2 + sigma_z[2]*hz
    B_3 = np.zeros(W1_mat.shape)
    C_1 = (.01)**2 * np.sum( (sigma_2-sigma_1)**2 )*np.ones(W1_mat.shape)/2
    C_2 = np.sum( (sigma_z)**2 )*np.ones(W1_mat.shape)/2
    C_3 = np.zeros(W1_mat.shape)
    # C_12 = .01*np.sum( (sigma_2-sigma_1)*sigma_z)*np.ones(W1_mat.shape)
    C_12 = np.zeros(W1_mat.shape)
    C_23 = np.zeros(W1_mat.shape)
    C_31 = np.zeros(W1_mat.shape)
    temp = (1-rho)* ( np.log( (1-r_mat)*(A1_cap - d1) + r_mat *(A2_cap-d2))-V0 )
    D = delta/(1-rho) * ( np.exp(temp) - 1) 
    D += psi1*(1-r_mat) +psi2*r_mat - (.01)**2/2 * (  (sigma_1[0]*(1-r_mat)+sigma_2[0]*r_mat)**2  + (sigma_1[1]*(1-r_mat)+sigma_2[1]*r_mat)**2 + (sigma_1[2]*(1-r_mat)+sigma_2[2]*r_mat)**2     )
    D += .01*( (sigma_1[0]*(1-r_mat)+sigma_2[0]*r_mat)*h1  + (sigma_1[1]*(1-r_mat)+sigma_2[1]*r_mat)*h2 + (sigma_1[2]*(1-r_mat)+sigma_2[2]*r_mat)*hz       )
    D += ell * ( h1**2 + h2**2 +hz**2 )/2
    
    start_ksp = time.time()

    A_1d = A.ravel(order='F')
    B_1_1d = B_1.ravel(order='F')
    B_2_1d = B_2.ravel(order='F')
    B_3_1d = B_3.ravel(order='F')
    C_1_1d = C_1.ravel(order='F')
    C_2_1d = C_2.ravel(order='F')
    C_3_1d = C_3.ravel(order='F')
    C_12_1d = C_12.ravel(order='F')
    C_23_1d = C_23.ravel(order='F')
    C_31_1d = C_31.ravel(order='F')
    D_1d = D.ravel(order='F')
    petsclinearsystemXDiff.formLinearSystem_DirectCrossDiff(W1_mat_1d, W2_mat_1d, W3_mat_1d, A_1d, B_1_1d, B_2_1d,
                                       B_3_1d, C_1_1d, C_2_1d, C_3_1d, C_12_1d, C_23_1d, C_31_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
    V0_1d = V0.ravel(order='F')
    b = V0_1d / epsilon + D_1d 
    # petsclinearsystemXDiff.formLinearSystem(W1_mat_1d, W2_mat_1d, W3_mat_1d, A_1d, B_1_1d, B_2_1d,
    #                                    B_3_1d, C_1_1d, C_2_1d, C_3_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
    # V0_1d = V0.ravel(order='F')
    # b = V0_1d  + D_1d *epsilon
    petsc_rhs = PETSc.Vec().createWithArray(b)
    x = petsc_mat.createVecRight()

    # create linear solver
    start_ksp = time.time()
    ksp.setOperators(petsc_mat)
    ksp.setTolerances(rtol=tol)
    ksp.solve(petsc_rhs, x)
    petsc_rhs.destroy()
    x.destroy()
    out_comp = np.array(ksp.getSolution()).reshape(A.shape, order="F")
    end_ksp = time.time()
    num_iter = ksp.getIterationNumber()

    # PDE_rhs = A * V0 + B_1 * dVdW1 + B_2 * dVdW2 + B_3 * \
    #     dVdW3 + C_1 * ddVddW1 + C_2 * ddVddW2 + C_3 * ddVddW3 + D
    PDE_rhs = A * V0 + B_1 * dVdW1 + B_2 * dVdW2 + C_1 * ddVddW1 + C_2 * ddVddW2 + C_12 * ddVdW1dW2 + D
    PDE_Err = np.max(abs(PDE_rhs))
    FC_Err = np.max(abs((out_comp - V0) / epsilon))

    V0 = out_comp

    d1_star = d1
    d2_star = d2
    h1_star = h1
    h2_star = h2
    hz_star = hz
    epoch += 1
    
    # if FC_Err<=1e-5 or epoch >=2990000:
        
    #     if epoch %1==0:
    #         print("V0_max,min={},{}".format(V0.max() , V0.min()))
    #         print("D_max,min={},{}".format(D.max() , D.min()))
    #         print("d_max,min={},{}".format(d.max() , d.min()))
    #         print("h1_max,min={},{}".format(h1.max() , h1.min()))
    #         print("hz_max,min={},{}".format(hz.max() , hz.min()))
    #         print("petsc total: {:.3f}s".format(end_ksp - start_ksp))
    #         print("PETSc preconditioned residual norm is {:g} iterations: {}".format(
    #             ksp.getResidualNorm(), ksp.getIterationNumber()))
    #         print("Epoch {:d} (PETSc): PDE Error: {:.10f} False Transient Error: {:.10f}" .format(
    #             epoch, PDE_Err, FC_Err))
    #         print("Epoch time: {:.4f}".format(time.time() - start_eps))
    # else:
    #     if epoch %100==0:
    #         print("V0_max,min={},{}".format(V0.max() , V0.min()))
    #         print("D_max,min={},{}".format(D.max() , D.min()))
    #         print("d_max,min={},{}".format(d.max() , d.min()))
    #         print("h1_max,min={},{}".format(h1.max() , h1.min()))
    #         print("hz_max,min={},{}".format(hz.max() , hz.min()))
    #         print("petsc total: {:.3f}s".format(end_ksp - start_ksp))
    #         print("PETSc preconditioned residual norm is {:g} iterations: {}".format(
    #             ksp.getResidualNorm(), ksp.getIterationNumber()))
    #         print("Epoch {:d} (PETSc): PDE Error: {:.10f} False Transient Error: {:.10f}" .format(
    #             epoch, PDE_Err, FC_Err))
    #         print("Epoch time: {:.4f}".format(time.time() - start_eps))
    
    print("V0_max,min={},{}".format(V0.max() , V0.min()))
    print("D_max,min={},{}".format(D.max() , D.min()))
    print("d1_max,min={},{}".format(d1.max() , d1.min()))
    print("d2_max,min={},{}".format(d2.max() , d2.min()))
    print("h1_max,min={},{}".format(h1.max() , h1.min()))
    print("h2_max,min={},{}".format(h2.max() , h2.min()))
    print("hz_max,min={},{}".format(hz.max() , hz.min()))
    print("petsc total: {:.3f}s".format(end_ksp - start_ksp))
    print("PETSc preconditioned residual norm is {:g} iterations: {}".format(
        ksp.getResidualNorm(), ksp.getIterationNumber()))
    print("Epoch {:d} (PETSc): PDE Error: {:.10f} False Transient Error: {:.10f}" .format(
        epoch, PDE_Err, FC_Err))
    print("Epoch time: {:.4f}".format(time.time() - start_eps))

res = {
    "V0": V0,
    "d_star": d_star,
    "h1_star": h1_star,
    "hz_star": hz_star,
    "FC_Err": FC_Err,
    "W1": W1,
    "W2": W2,
    "W3": W3,
}

Data_Dir = "./data/"+args.dataname+"/"
os.makedirs(Data_Dir, exist_ok=True)

with open(Data_Dir + "result_rho_{}_eps_{}_frac_{}".format(rho,epsilon,fraction), "wb") as f:
    pickle.dump(res, f)


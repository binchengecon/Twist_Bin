import numpy as np


def finiteDiff_3D(data, dim, order, dlt, cap = None):  
    # compute the central difference derivatives for given input and dimensions
    res = np.zeros(data.shape)
    l = len(data.shape)
    if l == 3:
        if order == 1:                    # first order derivatives
            
            if dim == 0:                  # to first dimension

                res[1:-1,:,:] = (1 / (2 * dlt)) * (data[2:,:,:] - data[:-2,:,:])
                res[-1,:,:] = (1 / dlt) * (data[-1,:,:] - data[-2,:,:])
                res[0,:,:] = (1 / dlt) * (data[1,:,:] - data[0,:,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1,:] = (1 / (2 * dlt)) * (data[:,2:,:] - data[:,:-2,:])
                res[:,-1,:] = (1 / dlt) * (data[:,-1,:] - data[:,-2,:])
                res[:,0,:] = (1 / dlt) * (data[:,1,:] - data[:,0,:])

            elif dim == 2:                # to third dimension

                res[:,:,1:-1] = (1 / (2 * dlt)) * (data[:,:,2:] - data[:,:,:-2])
                res[:,:,-1] = (1 / dlt) * (data[:,:,-1] - data[:,:,-2])
                res[:,:,0] = (1 / dlt) * (data[:,:,1] - data[:,:,0])

            else:
                raise ValueError('wrong dim')
                
        elif order == 2:
            
            if dim == 0:                  # to first dimension

                res[1:-1,:,:] = (1 / dlt ** 2) * (data[2:,:,:] + data[:-2,:,:] - 2 * data[1:-1,:,:])
                res[-1,:,:] = (1 / dlt ** 2) * (data[-1,:,:] + data[-3,:,:] - 2 * data[-2,:,:])
                res[0,:,:] = (1 / dlt ** 2) * (data[2,:,:] + data[0,:,:] - 2 * data[1,:,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1,:] = (1 / dlt ** 2) * (data[:,2:,:] + data[:,:-2,:] - 2 * data[:,1:-1,:])
                res[:,-1,:] = (1 / dlt ** 2) * (data[:,-1,:] + data[:,-3,:] - 2 * data[:,-2,:])
                res[:,0,:] = (1 / dlt ** 2) * (data[:,2,:] + data[:,0,:] - 2 * data[:,1,:])

            elif dim == 2:                # to third dimension

                res[:,:,1:-1] = (1 / dlt ** 2) * (data[:,:,2:] + data[:,:,:-2] - 2 * data[:,:,1:-1])
                res[:,:,-1] = (1 / dlt ** 2) * (data[:,:,-1] + data[:,:,-3] - 2 * data[:,:,-2])
                res[:,:,0] = (1 / dlt ** 2) * (data[:,:,2] + data[:,:,0] - 2 * data[:,:,1])

            else:
                raise ValueError('wrong dim')
            
        else:
            raise ValueError('wrong order')
    elif l == 2:
        if order == 1:                    # first order derivatives
            
            if dim == 0:                  # to first dimension

                res[1:-1,:] = (1 / (2 * dlt)) * (data[2:,:] - data[:-2,:])
                res[-1,:] = (1 / dlt) * (data[-1,:] - data[-2,:])
                res[0,:] = (1 / dlt) * (data[1,:] - data[0,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1] = (1 / (2 * dlt)) * (data[:,2:] - data[:,:-2])
                res[:,-1] = (1 / dlt) * (data[:,-1] - data[:,-2])
                res[:,0] = (1 / dlt) * (data[:,1] - data[:,0])

            else:
                raise ValueError('wrong dim')
                
        elif order == 2:
            
            if dim == 0:                  # to first dimension

                res[1:-1,:] = (1 / dlt ** 2) * (data[2:,:] + data[:-2,:] - 2 * data[1:-1,:])
                res[-1,:] = (1 / dlt ** 2) * (data[-1,:] + data[-3,:] - 2 * data[-2,:])
                res[0,:] = (1 / dlt ** 2) * (data[2,:] + data[0,:] - 2 * data[1,:])

            elif dim == 1:                # to second dimension

                res[:,1:-1] = (1 / dlt ** 2) * (data[:,2:] + data[:,:-2] - 2 * data[:,1:-1])
                res[:,-1] = (1 / dlt ** 2) * (data[:,-1] + data[:,-3] - 2 * data[:,-2])
                res[:,0] = (1 / dlt ** 2) * (data[:,2] + data[:,0] - 2 * data[:,1])

            else:
                raise ValueError('wrong dim')
            
        else:
            raise ValueError('wrong order')

            
    else:
        raise ValueError("Dimension NOT supported")
        
    if cap is not None:
        res[res < cap] = cap
    return res


# def tilting_function(sigma_k, sigma_z, alpha_k_tilde, alpha_k_hat, alpha_z_tilde, alpha_z_hat, beta_tilde, beta_hat, kappa_hat, kappa_tilde):

#     if sigma_k[0] == 0:
#         ind_sk = 1
#     else:
#         ind_sk = 0
        
#     eta0_1 = (alpha_k_tilde - alpha_k_hat)/sigma_k[ind_sk]
#     eta0_2 = (alpha_z_tilde - alpha_z_hat - sigma_z[ind_sk]*eta0_1)/sigma_z[-1]

#     eta1_1 = (beta_tilde - beta_hat)/sigma_k[ind_sk]
#     eta1_2 = (kappa_hat - kappa_tilde - sigma_z[ind_sk]*eta1_1)/sigma_z[-1]

#     eta0 = np.array([eta0_1, eta0_2])
#     eta1 = np.array([eta1_1, eta1_2])

#     xi0 = np.dot(eta0, eta0)
#     xi1 = np.dot(eta0, eta1)
#     xi2 = np.dot(eta1, eta1)

#     return xi0, xi1, xi2

def tilting_function(W1_mat, sigma_k, sigma_z, alpha_k_tilde, alpha_k_hat, alpha_z_tilde, alpha_z_hat, beta_tilde, beta_hat, kappa_hat, kappa_tilde):

    if sigma_k[0] == 0:
        ind_sk = 1
    else:
        ind_sk = 0
        
    eta0_1 = (alpha_k_tilde - alpha_k_hat)/sigma_k[ind_sk]
    eta0_2 = (alpha_z_tilde - alpha_z_hat - sigma_z[ind_sk]*eta0_1)/sigma_z[-1]

    eta1_1 = (beta_tilde - beta_hat)/sigma_k[ind_sk]
    eta1_2 = (kappa_hat - kappa_tilde - sigma_z[ind_sk]*eta1_1)/sigma_z[-1]

    eta0 = np.array([eta0_1, eta0_2])
    eta1 = np.array([eta1_1, eta1_2])

    xi0 = np.dot(eta0, eta0)
    xi1 = np.dot(eta0, eta1)
    xi2 = np.dot(eta1, eta1)

    return xi0+ xi1*(W1_mat-zmax)+ xi2*(W1_mat-zmax)**2

# if symmetric_returns == 1:

#     # beta2_hat = 0.5
#     # beta1_hat = 0.5

#     # (2) Technology
#     # phi2 = phi_1cap
#     # phi1 = phi_1cap
#     # A2 = A_1cap
#     # A1 = A_1cap

#     if state_dependent_xi == 0:
#         # Constant tilting function
#         scale = 1.754
#         alpha_k2_hat = alpha_k_hat
#         alpha_k1_hat = alpha_k_hat

#         # Worrisome model
#         alpha_z_tilde  = -0.005
#         kappa_tilde    = kappa_hat
#         alpha_k1_tilde = alpha_k1_hat
#         beta1_tilde    = beta1_hat
#         alpha_k2_tilde = alpha_k2_hat
#         beta2_tilde    = beta2_hat

#         ell_star = 0.055594409575544096

#     elif state_dependent_xi == 1:
#         # State-dependent tilting function (fixed kappa, alpha targets q)
#         scale = 1.62
#         alpha_k2_hat = alpha_k_hat
#         alpha_k1_hat = alpha_k_hat

#         alpha_z_tilde  = -0.00155
#         kappa_tilde    =  0.005
#         alpha_k1_tilde = alpha_k1_hat
#         beta1_tilde    = beta1_hat
#         alpha_k2_tilde = alpha_k2_hat
#         beta2_tilde    = beta2_hat

#         ell_star = 0.13852940062708508

#     elif state_dependent_xi == 2:
#         # State-dependent tilting function
#         scale = 1.568
#         alpha_k2_hat = alpha_k_hat
#         alpha_k1_hat = alpha_k_hat

#         alpha_z_tilde  = -0.00155
#         kappa_tilde    = kappa_hat
#         alpha_k1_tilde = alpha_k1_hat
#         beta1_tilde    = beta1_hat + .1941
#         alpha_k2_tilde = alpha_k2_hat
#         beta2_tilde    = beta2_hat + .1941

#         ell_star = 0.18756641482672026

    


# elif symmetric_returns == 0:

#     beta1_hat = 0.0
#     beta2_hat = 0.5

#     # (2) Technology
#     phi2 = phi1 = phi_1cap
#     A2 = A1 = A_1cap

#     if state_dependent_xi == 0:
#         # Constant tilting function
#         scale = 1.307
#         alpha_k2_hat = alpha_k_hat
#         alpha_k1_hat = alpha_k_hat

#         # Worrisome model
#         alpha_z_tilde  = -0.00534
#         kappa_tilde    = kappa_hat
#         alpha_k1_tilde = alpha_k1_hat
#         beta1_tilde    = beta1_hat
#         alpha_k2_tilde = alpha_k2_hat
#         beta2_tilde    = beta2_hat

#         ell_star = 0.026320287107624605

#     elif state_dependent_xi == 1:
#         # State-dependent tilting function (fixed kappa, alpha targets q)
#         scale = 1.14
#         alpha_k2_hat = alpha_k_hat + 0.035 #.034
#         alpha_k1_hat = alpha_k_hat + 0.035 #.034

#         alpha_z_tilde  = -0.002325
#         kappa_tilde    = 0.005
#         alpha_k1_tilde = alpha_k1_hat
#         beta1_tilde    = beta1_hat
#         alpha_k2_tilde = alpha_k2_hat
#         beta2_tilde    = beta2_hat

#         ell_star = 0.04226404306515605

#     elif state_dependent_xi == 2:
#         # State-dependent tilting function (fixed beta1, alpha targets q)
#         scale = 1.27
#         alpha_k2_hat = alpha_k_hat
#         alpha_k1_hat = alpha_k_hat

#         alpha_z_tilde  = -0.002325
#         kappa_tilde    = kappa_hat
#         alpha_k1_tilde = alpha_k1_hat
#         beta1_tilde    = beta1_hat + 0.194 #.195
#         alpha_k2_tilde = alpha_k2_hat
#         beta2_tilde    = beta2_hat + 0.194 #.195

#         ell_star = 0.06678494013273199


# sigma_k1 = [0.477*np.sqrt(scale),               0.0,   0.0]
# sigma_k2 = [0.0              , 0.477*np.sqrt(scale),   0.0]

#==============================================================================#
#    ENVIRONMENT
#==============================================================================#

# symmetric_returns    = 1     # 0 for symmetric return case, 1 for asymmetric returns
# state_dependent_xi   = 0     # 0 for constant twisting function, 1 for twisting beta_tilde, 2 for twsting kappa_tilde 1 and 2 are state dependent cases
# optimize_over_ell    = 0     # whether estimating lagrange multiplier 
# compute_irfs         = 0     # 1 if one wants to compute irfs, it's fine grid and wold be much more slower


# # (3) GRID
# # For analysis
# if compute_irfs == 1:
#     II, JJ = 7001, 501     # number of r points, number of z points
#     rmax = 4.0
#     rmin = -rmax
#     zmax = 0.7
#     zmin = -zmax
# elif compute_irfs == 0:


# # For the optimization (over ell)
# II_opt, JJ_opt = 501, 201     # number of r points, number of z points
# rmax_opt = 18.0
# rmin_opt = -rmax_opt
# zmax_opt = 1.2
# zmin_opt = -zmax_opt

# def tilting_function(W1,params_hat,params_tilde):
    
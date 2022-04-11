import random
import numpy as np
import time
import scipy.sparse as sparse
from itertools import compress
import copy
import pickle
import matplotlib.pyplot as plt
from scipy.special import loggamma
from matplotlib import cm
import multiprocessing
import os
import sys


def cov_idx_to_data_idx(row_idx, col_idx, row_link, col_link):
    '''
    takes two lists row_idx, col_idx, return all non-missing results associated with the corresponding peptides
    and treatments, stacked as a |row_idx|*|col_idx| \times 2 array
    :param row_idx: list of row idx of the bi clustering
    :param col_idx: list of column idx of the bi clustering
    :param row_link, col_link: list of lists, r_link[i] is the list of row idx that corresponds to the i th peptide
    :return: stacked block_data that goes into the func marginal_lkd
    '''
    data_rows, data_columns = [], []
    for r in row_idx:
        data_rows += row_link[r]
    for c in col_idx:
        data_columns += col_link[c]
    return data_rows, data_columns



def normal_approximation(data, x, y, sigma, n_iter=2000, alpha=1., beta=50., mu=-2., nu=1., n_normal=700):
    para_lst = np.zeros((n_iter, 2))
    old_para = [x, y]
    old_post = joint_log_lkd_repar(data, sigma, old_para[0], old_para[1], alpha=alpha, beta=beta, mu=mu, nu=nu)
    post_lst = np.zeros(n_iter)
    mode_est = [x, y]
    for n in range(n_iter):
        proposed_para_a = old_para[0] + 0.07*np.random.randn()
        proposed_post = joint_log_lkd_repar(data, sigma, proposed_para_a, old_para[1], alpha=alpha, beta=beta, mu=mu, nu=nu)
        if (proposed_post-old_post)>np.log(np.random.uniform()):
            old_para[0] = proposed_para_a
            old_post = proposed_post

        proposed_para_b = old_para[1] + 0.07*np.random.randn()
        proposed_post = joint_log_lkd_repar(data, sigma, old_para[0], proposed_para_b, alpha=alpha, beta=beta, mu=mu, nu=nu)
        if (proposed_post-old_post)>np.log(np.random.uniform()):
            old_para[1] = proposed_para_b
            old_post = proposed_post
        para_lst[n] = old_para
        post_lst[n] = old_post
        if n!=0 and old_post>post_lst[n-1]:
            mode_est = old_para

    mean = para_lst[500:].mean(axis=0)
    cov = 1.2*np.cov(para_lst[500:].T)
    # print(mean, mode_est, joint_log_lkd_repar(data, sigma, mean[0], mean[1]), joint_log_lkd_repar(data, sigma, mode_est[0], mode_est[1]))
    original_normal_samples = np.random.randn(n_normal, 2)
    transformed_normal_samples = original_normal_samples @ np.linalg.cholesky(cov).T + mode_est

    log_ratio = joint_log_lkd_repar(data, sigma, transformed_normal_samples[:,0], transformed_normal_samples[:,1], alpha=alpha, beta=beta, mu=mu, nu=nu) + \
                (original_normal_samples[:,0]**2+original_normal_samples[:,1]**2)/2 + \
                np.log(2*np.pi) + 0.5*np.linalg.slogdet(cov)[1]
    max_log_ratio = np.max(log_ratio)

    log_marginal_lkd = np.log(np.mean(np.exp(log_ratio - max_log_ratio))) + \
                       max_log_ratio
    return log_marginal_lkd

# ############ Functions for marginal log likelihood  ################################################################
# there is no d in this log lkd, we also remove the first entry of the time series
# which is 0, not much info there I guess
def joint_log_lkd(block_data, sigma, a, b, alpha=1., beta=50., mu=-2., nu=1.):
    if block_data.size == 0:
        return 0.0  # given empty data set
    # then check if there is any NaNs
    nan_idx = np.isnan(block_data).any(axis=1)
    if np.alltrue(nan_idx):
        return 0.0
    else:
        block_data = block_data[~nan_idx]
    if np.any(np.isnan(block_data)):
        exit('problem in removing nans')

    N_k = len(block_data)
    cond_means = a*(1-np.exp(-b*np.array([30, 300])))[np.newaxis, :].repeat(N_k, axis=0)  # size N_k*2
    residuals = block_data - cond_means  # size N_k*2
    log_lkd = -np.sum(residuals**2)/(2*sigma**2) - N_k*np.log(2*np.pi*sigma**2)
    log_prior = alpha*np.log(beta) - loggamma(alpha) + (alpha-1)*np.log(b) - beta*b - \
                np.log(a*np.sqrt(2*np.pi*nu**2)) - (np.log(a)-mu)**2/(2*nu**2)
    return log_lkd + log_prior


# transform the integrand so it is defined on R^2
def joint_log_lkd_repar(block_data, sigma, x, y, alpha=1., beta=50., mu=-2., nu=1.):
    beta_prime = beta/60  # due to rescaling the time from seconds to minutes
    # warning: only support alpha=1, please do not change
    if block_data.size == 0:
        return 0.0  # given empty data set

    # then check if there is any NaNs
    nan_idx = np.isnan(block_data).any(axis=1)
    if np.alltrue(nan_idx):
        return 0.0
    else:
        block_data = block_data[~nan_idx]

    if np.any(np.isnan(block_data)):
        exit('problem in removing nans')

    N_k = len(block_data)
    if not isinstance(x, np.ndarray):
        x = np.array([x])
        y = np.array([y])
    cond_means = np.exp(x[:,np.newaxis])*(1-np.exp(-np.exp(y[:,np.newaxis])*np.array([0.5, 5.])))
    # P*2 array, p is length of x,y
    residuals = block_data[:,np.newaxis,:] - cond_means[np.newaxis,:,:]
    # need N*1*2 - 1*P*2 to generate N*P*3, then sum over first and last dim
    log_lkd = -np.sum(residuals**2, axis=(0, 2))/(2*sigma**2) - 0.5*(x-mu)**2/nu**2 + (alpha-1)*y - beta_prime*np.exp(y) + alpha*y
    return log_lkd + 0.5*np.log(1/(2*np.pi*nu**2)) + alpha*np.log(beta_prime) - loggamma(alpha) - N_k*np.log(2*np.pi*sigma**2)




# def marginal_lkd_MC(block_data, sigma, max_iter=900, tol=1e-6, alpha=1, beta=50, mu=-2, nu=1, n_iter=2000, n_normal=700):
#     return normal_approximation(block_data, 0, 0, sigma, n_iter=n_iter, alpha=alpha, beta=beta, mu=mu, nu=nu, n_normal=n_normal)
#
# # compute MC estimator of the marginal lkd
# def marginal_lkd_MC(block_data, sigma, max_iter=900, tol=1e-6, alpha=1., beta=50., mu=-2., nu=1., n_iter=1500, n_normal=700, gibbs=False):
#     """after all, one can interpret it as a big pseudo marginal MCMC sampler, still valid!"""
#     if gibbs:
#         return normal_approximation(block_data, 0, -1.6, sigma, n_iter=n_iter, n_normal=n_normal, alpha=alpha, beta=beta, mu=mu, nu=nu)
#     beta_prime = beta/60    # due to rescaling the time from seconds to minutes
#     if block_data.size == 0:
#         return 0.0  # given empty data set
#
#     # then check if there is any NaNs
#     nan_idx = np.isnan(block_data).any(axis=1)
#     if np.alltrue(nan_idx):
#         return 0.0
#     else:
#         block_data = block_data[~nan_idx]
#
#     if np.any(np.isnan(block_data)):
#         exit('problem in removing nans')
#
#     # step1: find optimal transformed parameter for the transformed log integrand
#     N_k = len(block_data)
#     curr_x, curr_y = 0., 0.
#     curr_integrand = joint_log_lkd_repar(block_data, sigma, curr_x, curr_y, alpha=alpha, beta=beta, mu=mu, nu=nu)[0]
#     # first run a few round of adadelta optimizer
#     # a crude way to push it to a region where the objective is ``well behaved"
#     RMSg = np.zeros(2)
#     RMSDelta = np.zeros(2)
#     rho=0.99
#     eps=1e-5
#     lkd=np.zeros(max_iter)
#     for i in range(max_iter):
#         curr_cond_mean = np.exp(curr_x)*(1-np.exp(-np.exp(curr_y)*np.array([0.5, 5.])))  # 2, array
#         curr_residual = block_data - curr_cond_mean  # N_k*2 array
#
#         curr_del_x = -1*curr_cond_mean  # 2, array
#         curr_del_y = -np.array([0.5, 5.])*np.exp(curr_x + curr_y - np.exp(curr_y)*np.array([0.5, 5.]))  # 2, array
#
#         grad_x = -(1/sigma**2)*(curr_residual*curr_del_x).sum() - (curr_x-mu)/nu**2
#         grad_y = -(1/sigma**2)*(curr_residual*curr_del_y).sum() - beta_prime*np.exp(curr_y) + alpha
#
#         # update the RMSEg and RMSEDelta
#         RMSg = rho*RMSg + (1-rho)*np.array([grad_x**2, grad_y**2])
#         Delta = np.sqrt(RMSDelta+eps)/np.sqrt(RMSg+eps)*(np.array([grad_x, grad_y]))
#         RMSDelta = rho*RMSDelta + (1-rho)*(Delta**2)
#         curr_x += Delta[0]
#         curr_y += Delta[1]
#         old_integrand = curr_integrand
#         curr_integrand = joint_log_lkd_repar(block_data, sigma, curr_x, curr_y, alpha=alpha, beta=beta, mu=mu, nu=nu)[0]
#         lkd[i] = curr_integrand
#         if np.abs((curr_integrand-old_integrand)) < tol:
#             break
#     # then a few round of newton raphson, hopefully gradient ascent has found a a region that looks convex so
#     # newton raphson will work properly
#
#     t = 0
#     while t < 50:
#         curr_cond_mean = np.exp(curr_x)*(1-np.exp(-np.exp(curr_y)*np.array([0.5, 5.])))  # 2, array
#         curr_residual = block_data - curr_cond_mean  # N_k*2 array
#         curr_del_x = -1*curr_cond_mean  # 2, array
#         curr_del_y = -np.array([0.5, 5.])*np.exp(curr_x + curr_y - np.exp(curr_y)*np.array([0.5, 5.]))  # 2, array
#
#         grad_x = -(1/sigma**2)*(curr_residual*curr_del_x).sum() - (curr_x-mu)/nu**2
#         grad_y = -(1/sigma**2)*(curr_residual*curr_del_y).sum() - beta_prime*np.exp(curr_y) + alpha
#
#         dxx = -(1/sigma**2)*((N_k*np.sum(curr_del_x**2)) + np.sum(curr_residual*curr_del_x)) - 1/nu**2
#         dyy = -(1/sigma**2)*((N_k*np.sum(curr_del_y**2)) + np.sum(curr_residual*(curr_del_y*(1-np.exp(curr_y)*np.array([0.5, 5.]))))) - beta_prime*np.exp(curr_y)
#         dxy = -(1/sigma**2)*((N_k*np.sum(curr_del_y*curr_del_x)) + np.sum(curr_residual*curr_del_y))
#
#         if (-dxx < 0) or (dxx*dyy-dxy**2) < 0:  # MCMC way of doing it, slightly more expensive
#             return normal_approximation(block_data, curr_x, curr_y, sigma, n_iter=n_iter, n_normal=n_normal, alpha=alpha, beta=beta, mu=mu, nu=nu)
#
#         inv_neg_hess = np.linalg.inv(-np.array([[dxx, dxy], [dxy, dyy]]))
#         step = inv_neg_hess @ np.array([grad_x, grad_y]).T
#         test_integrand = joint_log_lkd_repar(block_data, sigma, curr_x+step[0], curr_y+step[1], alpha=alpha, beta=beta, mu=mu, nu=nu)[0]
#         if (test_integrand < curr_integrand) or np.abs((curr_integrand-test_integrand)) < tol:
#             break
#         else:
#             curr_x += step[0]
#             curr_y += step[1]
#             curr_integrand = test_integrand
#         t += 1
#
#     max_integrand = curr_integrand
#     MAP_x = curr_x
#     MAP_y = curr_y
#     # just in case if the newton raphson step is skipped
#     MAP_cond_mean = np.exp(MAP_x)*(1-np.exp(-np.exp(MAP_y)*np.array([0.5, 5.])))  # 2, array
#     MAP_residual = block_data - MAP_cond_mean  # N_k*3 array
#     MAP_del_x = -1*MAP_cond_mean  # 3, array
#     MAP_del_y = -np.array([0.5, 5.])*np.exp(MAP_x + MAP_y - np.exp(MAP_y)*np.array([0.5, 5.]))  # 2, array
#
#     dxx = -(1/sigma**2)*((N_k*np.sum(MAP_del_x**2)) + np.sum(MAP_residual*MAP_del_x)) - 1/nu**2
#     dyy = -(1/sigma**2)*((N_k*np.sum(MAP_del_y**2)) + np.sum(MAP_residual*(MAP_del_y*(1-np.exp(MAP_y)*np.array([0.5, 5.]))))) - beta_prime*np.exp(MAP_y)
#     dxy = -(1/sigma**2)*((N_k*np.sum(MAP_del_y*MAP_del_x)) + np.sum(MAP_residual*MAP_del_y))
#
#     if (-dxx < 0) or (dxx*dyy-dxy**2) < 0:  # MCMC way of doing it, slightly more expensive
#         return normal_approximation(block_data, curr_x, curr_y, sigma, n_iter=n_iter, n_normal=n_normal, alpha=alpha, beta=beta, mu=mu, nu=nu)
#     else:
#         inv_neg_hess = np.linalg.inv(-np.array([[dxx, dxy], [dxy, dyy]]))
#         inv_scale_mtrx = np.linalg.cholesky(inv_neg_hess)
#
#     # scaled, shifted integrand on log scale
#     def alt_log_integrand_scaled(x, y):
#         input = np.array([x,y])
#         new_input = inv_scale_mtrx @ input
#         new_x = new_input[0]+MAP_x
#         new_y = new_input[1]+MAP_y
#         return joint_log_lkd_repar(block_data, sigma, new_x, new_y, alpha=alpha, beta=beta, mu=mu, nu=nu) - max_integrand
#
#     # simple IS estimator of the marginal lkd. Note that the integrand is centered at 0, and has hessian=I at 0.
#     # In addition, exp(-exp()) should decay faster than the normal pdf exp(-x^2/2), so should be ok
#     # proposal = N(0,3I)
#     test_points_x, test_points_y = 1.5*np.random.randn(400), 1.5*np.random.randn(400)
#     test_points_x = np.r_[test_points_x, -1*test_points_x]
#     test_points_y = np.r_[test_points_y, -1*test_points_y]  # also try antithetic variable
#
#     log_ratio = alt_log_integrand_scaled(test_points_x, test_points_y) + (test_points_x**2+test_points_y**2)/(2*(1.5)**2) + np.log(2*np.pi*(1.5)**2)
#     max_log_ratio = np.max(log_ratio)
#     # Gaussian proposal with 3I or 2I covariance seems sufficient for this dataset
#
#     log_marginal_lkd = np.log(np.mean(np.exp(log_ratio - max_log_ratio))) + \
#                        max_log_ratio + max_integrand + np.log(np.abs(inv_scale_mtrx[0, 0]*inv_scale_mtrx[1, 1]))
#
#     # resulting log integral should be int_alt + logdet(inv_scale_mtrx)+ max_integrand
#     return log_marginal_lkd




# transform the integrand so it is defined on R^2
def joint_log_lkd_repar_scaled(block_data, sigma, x, y, alpha=1., beta=50., mu=-2., nu=1., c=30., ct=100.):
    if block_data.size == 0:
        return 0.0  # given empty data set
    # then check if there is any NaNs
    nan_idx = np.isnan(block_data).any(axis=1)
    if np.alltrue(nan_idx):
        return 0.0
    else:
        block_data = block_data[~nan_idx]

    if np.any(np.isnan(block_data)):
        exit('problem in removing nans')

    # scale the dataset and adjust the parameters
    block_data = c*block_data
    beta_prime = beta/ct  # due to rescaling the time from seconds to minutes, use 5 min as standard unit
    mu_prime = mu + np.log(c)
    sigma_prime = c*sigma

    N_k = len(block_data)
    if not isinstance(x, np.ndarray):
        x = np.array([x])
        y = np.array([y])
    cond_means = np.exp(x[:,np.newaxis])*(1-np.exp(-np.exp(y[:,np.newaxis])*np.array([30./ct, 300./ct])))
    # P*2 array, p is length of x,y
    residuals = block_data[:,np.newaxis,:] - cond_means[np.newaxis,:,:]
    # need N*1*2 - 1*P*2 to generate N*P*3, then sum over first and last dim
    log_lkd = -np.sum(residuals**2, axis=(0, 2))/(2*sigma_prime**2) - 0.5*(x-mu_prime)**2/nu**2 - beta_prime*np.exp(y) + alpha*y
    return log_lkd + 0.5*np.log(1/(2*np.pi*nu**2)) + alpha*np.log(beta_prime) - loggamma(alpha) - N_k*np.log(2*np.pi*(sigma_prime/c)**2)


# compute MC estimator of the marginal lkd
def marginal_lkd_MC(block_data, sigma, max_iter=900, tol=1e-6, alpha=1., beta=50., mu=-2., nu=1., n_iter=1500, n_normal=700, gibbs=False, c=30., ct=100.):
    """after all, one can interpret it as a big pseudo marginal MCMC sampler, still valid!"""
    if gibbs:
        return normal_approximation(block_data, 0, -1.6, sigma, n_iter=n_iter, n_normal=n_normal, alpha=alpha, beta=beta, mu=mu, nu=nu)
    if block_data.size == 0:
        return 0.0  # given empty data set

    # then check if there is any NaNs
    nan_idx = np.isnan(block_data).any(axis=1)
    if np.alltrue(nan_idx):
        return 0.0
    else:
        block_data = block_data[~nan_idx]

    if np.any(np.isnan(block_data)):
        exit('problem in removing nans')

    # scale the data, adjust parameters accordingly
    block_data_prime = c*block_data
    beta_prime = beta/ct  # due to rescaling the time from seconds to minutes
    mu_prime = mu + np.log(c)
    sigma_prime = c*sigma

    # step1: find optimal transformed parameter for the transformed, scaled log integrand
    N_k = len(block_data)
    curr_x, curr_y = 0., 0.
    curr_integrand = joint_log_lkd_repar_scaled(block_data, sigma, curr_x, curr_y, alpha=alpha, beta=beta, mu=mu, nu=nu, c=c, ct=ct)[0]
    # first run a few round of adadelta optimizer
    # a crude way to push it to a region where the objective is ``well behaved"
    RMSg = np.zeros(2)
    RMSDelta = np.zeros(2)
    rho=0.99
    eps=1e-5
    lkd=np.zeros(max_iter)
    for i in range(max_iter):
        curr_cond_mean = np.exp(curr_x)*(1-np.exp(-np.exp(curr_y)*np.array([30./ct, 300./ct])))  # 2, array
        curr_residual = block_data_prime - curr_cond_mean  # N_k*2 array

        curr_del_x = -1*curr_cond_mean  # 2, array
        curr_del_y = -np.array([30./ct, 300./ct])*np.exp(curr_x + curr_y - np.exp(curr_y)*np.array([30./ct, 300./ct]))  # 2, array

        grad_x = -(1/sigma_prime**2)*(curr_residual*curr_del_x).sum() - (curr_x-mu_prime)/nu**2
        grad_y = -(1/sigma_prime**2)*(curr_residual*curr_del_y).sum() - beta_prime*np.exp(curr_y) + alpha

        # update the RMSEg and RMSEDelta
        RMSg = rho*RMSg + (1-rho)*np.array([grad_x**2, grad_y**2])
        Delta = np.sqrt(RMSDelta+eps)/np.sqrt(RMSg+eps)*(np.array([grad_x, grad_y]))
        RMSDelta = rho*RMSDelta + (1-rho)*(Delta**2)
        curr_x += Delta[0]
        curr_y += Delta[1]
        old_integrand = curr_integrand
        curr_integrand = joint_log_lkd_repar_scaled(block_data, sigma, curr_x, curr_y, alpha=alpha, beta=beta, mu=mu, nu=nu, c=c, ct=ct)[0]
        lkd[i] = curr_integrand
        if np.abs((curr_integrand-old_integrand)) < tol:
            break
    # then a few round of newton raphson, hopefully gradient ascent has found a a region that looks convex so
    # newton raphson will work properly

    t = 0
    while t < 50:
        curr_cond_mean = np.exp(curr_x)*(1-np.exp(-np.exp(curr_y)*np.array([30./ct, 300./ct])))  # 2, array
        curr_residual = block_data_prime - curr_cond_mean  # N_k*2 array
        curr_del_x = -1*curr_cond_mean  # 2, array
        curr_del_y = -np.array([30./ct, 300./ct])*np.exp(curr_x + curr_y - np.exp(curr_y)*np.array([30./ct, 300./ct]))  # 2, array

        grad_x = -(1/sigma_prime**2)*(curr_residual*curr_del_x).sum() - (curr_x-mu_prime)/nu**2
        grad_y = -(1/sigma_prime**2)*(curr_residual*curr_del_y).sum() - beta_prime*np.exp(curr_y) + alpha

        dxx = -(1/sigma_prime**2)*((N_k*np.sum(curr_del_x**2)) + np.sum(curr_residual*curr_del_x)) - 1/nu**2
        dyy = -(1/sigma_prime**2)*((N_k*np.sum(curr_del_y**2)) + np.sum(curr_residual*(curr_del_y*(1-np.exp(curr_y)*np.array([30./ct, 300./ct]))))) - beta_prime*np.exp(curr_y)
        dxy = -(1/sigma_prime**2)*((N_k*np.sum(curr_del_y*curr_del_x)) + np.sum(curr_residual*curr_del_y))

        if (-dxx < 0) or (dxx*dyy-dxy**2) < 0:  # MCMC way of doing it, slightly more expensive
            print('oops')
            return normal_approximation(block_data, curr_x, curr_y, sigma, n_iter=n_iter, n_normal=n_normal, alpha=alpha, beta=beta, mu=mu, nu=nu)

        inv_neg_hess = np.linalg.inv(-np.array([[dxx, dxy], [dxy, dyy]]))
        step = inv_neg_hess @ np.array([grad_x, grad_y]).T
        test_integrand = joint_log_lkd_repar_scaled(block_data, sigma, curr_x+step[0], curr_y+step[1], alpha=alpha, beta=beta, mu=mu, nu=nu, c=c, ct=ct)[0]
        if (test_integrand < curr_integrand) or np.abs((curr_integrand-test_integrand)) < tol:
            break
        else:
            curr_x += step[0]
            curr_y += step[1]
            curr_integrand = test_integrand
        t += 1

    max_integrand = curr_integrand
    MAP_x = curr_x
    MAP_y = curr_y
    # just in case if the newton raphson step is skipped
    MAP_cond_mean = np.exp(MAP_x)*(1-np.exp(-np.exp(MAP_y)*np.array([30./ct, 300./ct])))  # 2, array
    MAP_residual = block_data_prime - MAP_cond_mean  # N_k*3 array
    MAP_del_x = -1*MAP_cond_mean  # 3, array
    MAP_del_y = -np.array([30./ct, 300./ct])*np.exp(MAP_x + MAP_y - np.exp(MAP_y)*np.array([30./ct, 300./ct]))  # 2, array

    dxx = -(1/sigma_prime**2)*((N_k*np.sum(MAP_del_x**2)) + np.sum(MAP_residual*MAP_del_x)) - 1/nu**2
    dyy = -(1/sigma_prime**2)*((N_k*np.sum(MAP_del_y**2)) + np.sum(MAP_residual*(MAP_del_y*(1-np.exp(MAP_y)*np.array([30./ct, 300./ct]))))) - beta_prime*np.exp(MAP_y)
    dxy = -(1/sigma_prime**2)*((N_k*np.sum(MAP_del_y*MAP_del_x)) + np.sum(MAP_residual*MAP_del_y))

    if (-dxx < 0) or (dxx*dyy-dxy**2) < 0:  # MCMC way of doing it, slightly more expensive
        print('oops')
        return normal_approximation(block_data, curr_x, curr_y, sigma, n_iter=n_iter, n_normal=n_normal, alpha=alpha, beta=beta, mu=mu, nu=nu)
    else:
        inv_neg_hess = np.linalg.inv(-np.array([[dxx, dxy], [dxy, dyy]]))
        inv_scale_mtrx = np.linalg.cholesky(inv_neg_hess)

    # scaled, shifted integrand on log scale
    def alt_log_integrand_scaled(x, y):
        input = np.array([x,y])
        new_input = inv_scale_mtrx @ input
        new_x = new_input[0]+MAP_x
        new_y = new_input[1]+MAP_y
        return joint_log_lkd_repar_scaled(block_data, sigma, new_x, new_y, alpha=alpha, beta=beta, mu=mu, nu=nu, c=c, ct=ct) - max_integrand

    # simple IS estimator of the marginal lkd. Note that the integrand is centered at 0, and has hessian=I at 0.
    # In addition, exp(-exp()) should decay faster than the normal pdf exp(-x^2/2), so should be ok
    # proposal = N(0,3I)
    test_points_x, test_points_y = 1.5*np.random.randn(400), 1.5*np.random.randn(400)
    test_points_x = np.r_[test_points_x, -1*test_points_x]
    test_points_y = np.r_[test_points_y, -1*test_points_y]  # also try antithetic variable

    log_ratio = alt_log_integrand_scaled(test_points_x, test_points_y) + (test_points_x**2+test_points_y**2)/(2*(1.5)**2) + np.log(2*np.pi*(1.5)**2)
    max_log_ratio = np.max(log_ratio)
    # Gaussian proposal with 3I or 2I covariance seems sufficient for this dataset

    log_marginal_lkd = np.log(np.mean(np.exp(log_ratio - max_log_ratio))) + \
                       max_log_ratio + max_integrand + np.log(np.abs(inv_scale_mtrx[0, 0]*inv_scale_mtrx[1, 1]))

    # resulting log integral should be int_alt + logdet(inv_scale_mtrx)+ max_integrand
    return log_marginal_lkd


# modified from https://github.com/Saket97/Mondrian-Processes/blob/master/Mondrian.py
class MondrianBlock:
    """
    one node of the Mondrian tree
    """
    def __init__(self, budget, rowLB, rowUB, columnLB, columnUB,
                 cutPos, cutDir, leftChild, rightChild, parent):
        """
        :param budget: budget for the block
        :param rowLB: row lower bound of the rectangular block
        :param rowUB: row upper bound of the rectangular block
        :param columnLB: column lower bound of the rectangular block
        :param columnUB: column upper bound of the rectangular block
        :param cutPos: position of the cut (if there is one)
        :param cutDir: direction of the cut (if there is one), 0=row cut, 1 = column cut
        :param leftChild: left child node of this node (if there is one)
        :param rightChild: right child node of this node (if there is one)
        :param parent: parent node of this node (if there is one)
        """

        # check if it is a binary tree
        if (leftChild is not None
            and rightChild is None) or (leftChild is None
                                        and rightChild is not None):
            exit('One child node is None and the other is not None!')
        # position and direction of a cut must be both none or both something
        if (cutPos is not None
            and cutDir is None) or (cutPos is None
                                    and cutDir is not None):
            exit('Cutting position and direction are not consistent!')
        # if the node has a cut in it, it can not be a leaf node and must have two children
        if cutPos is not None:
            if leftChild is None or rightChild is None:
                exit('Non-leaf node must have two child nodes!')
        # cut direction is binary
        if cutDir is not None and (cutPos > 1 or cutPos < 0):
            exit('Illegal cutting position!')
        # block size must be legal
        if rowLB >= rowUB or columnLB >= columnUB:
            exit('Illegal block size!')

        # initialize parameters
        self.budget = budget * 1.0
        self.rowLB = rowLB * 1.0
        self.rowUB = rowUB * 1.0
        self.columnLB = columnLB * 1.0
        self.columnUB = columnUB * 1.0
        self.cutPos = cutPos
        self.cutDir = cutDir
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.parent = parent
        self.DataPoints = None
        # if is a leaf, carry the column and row index that fall inside this block, in the form of [[row idx],[col idx]]
        self.marginal_lkd = None  # if it is a leaf, store the marginal lkd for this block

    # return the parent node of this one
    def getParent(self):
        return self.parent

    # return the left child
    def getLeftChild(self):
        return self.leftChild

    # return the right child
    def getRightChild(self):
        return self.rightChild

    def getDatapoints(self):
        return self.DataPoints

    # given a cut direction, cut position, the resulting left and right leaf node, register/update this cut to this node
    def setCut(self, cutDir, cutPos, leftChild, rightChild):
        """
        things to do when a leaf becomes a leaf cut:
        add the cutting info and the two child nodes. remove the marginal lkd and the data points in it
        leftChild and rightChild's marginal lkd and data points should be added in advance
        :param cutDir: direction of the cut
        :param cutPos: position of the cut
        :param leftChild: the resulting left child due to the cut, no need to compute the marginal lkd
        :param rightChild: the resulting right child due to the cut, no need to compute the marginal lkd
        :return: updates the node info, returns nothing
        """
        # check if inputs are legal
        if cutDir is None or cutPos is None or leftChild is None or rightChild is None:
            exit('Error cut info!')
        else:
            self.cutPos = cutPos
            self.cutDir = cutDir
            self.leftChild = leftChild
            self.rightChild = rightChild
            self.DataPoints = None
            self.marginal_lkd = None

    # remove the cut in this node, update the info
    def removeCut(self):
        """
        things to do when removing a cut: remove the cut info and the child nodes
        marginal lkd and data points will be added once we introduce the data set
        :return: update a block, return nothing
        """
        self.cutPos = None
        self.cutDir = None
        self.leftChild = None
        self.rightChild = None


    def isLeaf(self):
        """
         check if this node is a leaf node,  leaf node has cut=None, parameter=not None
         non leaf node has cut=not None, parameter=None
        :return:
        """
        return self.cutPos is None

    # check if two nodes are equal
    def __eq__(self, another):
        if isinstance(another, MondrianBlock):
            if (  # self.budget == another.budget and  # ignore budget, should do no harm
                    self.rowLB == another.rowLB
                    and self.rowUB == another.rowUB
                    and self.columnLB == another.columnLB
                    and self.columnUB == another.columnUB):
                return True
        return False

    # str(node) representation, name of the node
    def __str__(self):  # ignore budget, should do no harm
        # return "block with budget {} at ({}, {}) x ({}, {})".format(self.budget, self.rowLB, self.rowUB,
        #                                                             self.columnLB, self.columnUB)
        return "block at ({}, {}) x ({}, {})".format(self.rowLB, self.rowUB,
                                                     self.columnLB, self.columnUB)

    def __hash__(self):
        return hash(self.__str__())

    def register_data_trapped(self, xi, eta):
        """
        find the list of cell index that falls inside this block
        :param xi: latent row coordinate
        :param eta: latent column coordinate
        :return: a list of cell index that falls inside this block
        """
        # list of column index such that the corresponding column coordinate is inside this node
        # recall eta is the column coordinate, xi is the row coordinate
        columnIdxLst = []
        # for each column coordinate, check if it is inside this node, record its index if it is in
        for columnIdx in range(len(eta)):
            columnPos = eta[columnIdx]
            if self.columnLB < columnPos < self.columnUB:
                columnIdxLst.append(columnIdx)

        if len(columnIdxLst) == 0:
            self.DataPoints = None
            return self.DataPoints

        # same thing for rows
        rowIdxLst = []
        for rowIdx in range(len(xi)):
            rowPos = xi[rowIdx]
            if self.rowLB < rowPos < self.rowUB:
                rowIdxLst.append(rowIdx)

        if len(rowIdxLst) == 0:
            self.DataPoints = None
            return self.DataPoints

        self.DataPoints = [rowIdxLst, columnIdxLst]
        return self.DataPoints

    def register_data_trapped2(self, xi, eta):
        """
        find the list of cell index that falls inside this block
        :param xi: latent row coordinate
        :param eta: latent column coordinate
        :return: a list of cell index that falls inside this block
        """
        # list of column index such that the corresponding column coordinate is inside this node
        # recall eta is the column coordinate, xi is the row coordinate
        columnIdxLst = []
        # for each column coordinate, check if it is inside this node, record its index if it is in
        for columnIdx in range(len(eta)):
            columnPos = eta[columnIdx]
            if self.columnLB < columnPos < self.columnUB:
                columnIdxLst.append(columnIdx)

        if len(columnIdxLst) == 0:
            return None

        # same thing for rows
        rowIdxLst = []
        for rowIdx in range(len(xi)):
            rowPos = xi[rowIdx]
            if self.rowLB < rowPos < self.rowUB:
                rowIdxLst.append(rowIdx)

        if len(rowIdxLst) == 0:
            return None

        return [rowIdxLst, columnIdxLst]

    def marginal_lkd_block_level(self, data, xi, eta, p, marginal_lkd_func, row_link, col_link, test=False, **kwargs):
        """
        block level method for evaluating marginal lkd
        :param data: 2d np.array
        :param xi: latent row coord
        :param eta: latent col coord
        :param p: dimension of the response in each cell (i,j)
        :param marginal_lkd_func: function for evaluating the marginal lkd
        takes data as its first argument, kwarg as other inputs, its data should be in np.array with dim K*p, where
        K is the number of data points in the block, p is the dimension of the entry. For protein data, p=3
        :param row_link, col_link: r_link[i] give all rows in data that is related to the ith peptide,
        c_link[i] give all columns in data that is related to the ith treatment,
        :param test: do we want to double check?
        :param kwargs: other inputs for marginal_lkd_func
        :return: marginal lkd for the data fall inside this block, also register the self.marginal_lkd
        """
        if self.DataPoints is None:
            self.marginal_lkd = 0.0
            return 0.0
        else:
            if test:
                testidx = self.register_data_trapped(xi,eta)
                if sorted(self.DataPoints[0]) != sorted(testidx[0]) or sorted(self.DataPoints[1]) != sorted(testidx[1]):
                    exit("something wrong with the data point tracking")
            # find all rows and columns associated with the peptide-treatment pairs
            rows, cols = cov_idx_to_data_idx(self.DataPoints[0], self.DataPoints[1], row_link, col_link)
            data_in_block = data[rows][:, cols].reshape(-1, p)
            marginal_lkd = marginal_lkd_func(data_in_block, **kwargs)
            self.marginal_lkd = marginal_lkd
            return marginal_lkd



class MondrianTree:
    """
    the class that represents a Mondrian tree
    """

    def __init__(self, budget, rowLB, rowUB, columnLB, columnUB, random_generate=False, max_cut=None):
        """
        initialize a Mondrian tree
        :param budget: initial budget
        :param rowLB: initial row lower bound
        :param rowUB: initial row upper bound
        :param columnLB: initial column lower bound
        :param columnUB: initial column upper bound
        :param random_generate: if true, sample from MP, false -> trivial partition
        """

        self.budget = budget
        # specify the root node, no cut or what so ever
        self.root = MondrianBlock(budget, rowLB, rowUB, columnLB, columnUB,
                                  None, None, None, None, None)
        # this is the only leaf block so far
        self.leafBlockDic = {self.root: True}

        # records all the nodes whose children are both leaf nodes
        self.leafCutDic = {}

        # records all the row cuts made, key is row cut position, value is the Mondrian block containing the cut
        self.rowCutDic = {}

        # records all the column cuts made, key is column cut position, value is the Mondrian block containing the cut
        self.columnCutDic = {}

        # get a list of all leaf blocks
        leafBlockLst = list(self.leafBlockDic.keys())
        if random_generate:
            # now initialize by constructing a realization of MP if random_generate=True
            cutNum = 0  # number of total cuts
            level = 0  # current level/depth of the tree
            while len(leafBlockLst) > 0:  # if we got one or more than one leaf in the current level
                if max_cut is not None and cutNum > max_cut:  # restrict the number of cuts if necessary
                    break
                level += 1
                newLeafBlockLst = []  # store leaf node that will be generated in the next level(depth +1)
                for leafBlock in leafBlockLst:  # scan over the current leaf nodes
                    # compute the half perimeter of the rectangle
                    length = leafBlock.rowUB - leafBlock.rowLB
                    width = leafBlock.columnUB - leafBlock.columnLB

                    # cost is a exponential rv with rate = half perimeter of the rectangle, expectation = 1/(height+width)
                    cost = random.expovariate(length + width)

                    # if cost not too high:
                    if not cost > leafBlock.budget:

                        # get the direction of the cut, 0 is row cut, 1 is column cut
                        if random.random() < length/(length + width):
                            cutDir = 0
                        else:
                            cutDir = 1

                        if cutDir == 0:  # if a row cut, split horizontally into two parts
                            cutPos = leafBlock.rowLB + random.random() * length  # position of the cut
                            leftChild = MondrianBlock(leafBlock.budget-cost, leafBlock.rowLB, cutPos,
                                                      leafBlock.columnLB, leafBlock.columnUB, None, None,
                                                      None, None, leafBlock)  # this will be a new left child leaf block
                            rightChild = MondrianBlock(leafBlock.budget-cost, cutPos, leafBlock.rowUB,
                                                       leafBlock.columnLB, leafBlock.columnUB, None, None,
                                                       None, None, leafBlock)  # this will be a new right child leaf block

                        else:  # if is a column cut, split vertically into two parts
                            cutPos = leafBlock.columnLB + random.random() * width  # position of the cut
                            leftChild = MondrianBlock(leafBlock.budget-cost, leafBlock.rowLB, leafBlock.rowUB,
                                                      leafBlock.columnLB, cutPos, None, None,
                                                      None, None, leafBlock)  # this will be a new left child leaf block
                            rightChild = MondrianBlock(leafBlock.budget-cost, leafBlock.rowLB, leafBlock.rowUB,
                                                       cutPos, leafBlock.columnUB, None, None,
                                                       None, None, leafBlock)  # this will be a new right child leaf block

                        cutNum += 1  # total number of cuts +1

                        # update the cut direction ,position, resulting blocks of the current leaf node
                        # see self.addCut
                        self.addCut(leafBlock, cutDir, cutPos, leftChild, rightChild)

                        newLeafBlockLst.append(leftChild)  # this is a new leaf now
                        newLeafBlockLst.append(rightChild)  # this is a new leaf now

                    else:  # if cost too high, terminate generative process, register it as a leaf
                        self.leafBlockDic[leafBlock] = True

                # update the leaf nodes of the next level/depth, do the for loop to scan over this new leaf list again
                leafBlockLst = newLeafBlockLst  # now this will be the next level of leaf nodes for the while loop

                print('cutNum = {}, level = {}, leafNum = {}, leafCutNum = {}'.format(cutNum, level,
                                                                                      len(self.leafBlockDic),
                                                                                      len(self.leafCutDic)))

    # get a random leaf block from the leaf block dictionary, return the name of the block
    def getRandomLeafBlock(self):  # get a random leaf
        leafBlock = list(self.leafBlockDic.keys())[random.randint(0, len(self.leafBlockDic)-1)]
        return leafBlock

    # get a random leaf cut from the leaf cut dictionary, return the name of the block
    # leafCut: the collection of nodes whose children are two leaf nodes
    def getRandomLeafCut(self):
        leafCut = None
        if len(self.leafCutDic) > 0:
            leafCut = list(self.leafCutDic.keys())[random.randint(0, len(self.leafCutDic)-1)]
        return leafCut

    # update the tree information when adding a cut to an existing leaf block
    def addCut(self, leafBlock, cutDir, cutPos, leftChild, rightChild):
        # remove the original leaf block being cut
        self.leafBlockDic.pop(leafBlock)

        # check if leafBlock is legal
        if not leftChild.isLeaf() or not rightChild.isLeaf():
            exit('this block should have leaf nodes as children!')

        # recall that setCut simply add direction, position and two resulting children nodes to the given leafBlock
        # also removes existing data points in it and the corresponding marginal lkd
        leafBlock.setCut(cutDir, cutPos, leftChild, rightChild)

        # add the two new leaf blocks to the tree
        self.leafBlockDic[leafBlock.leftChild] = True
        self.leafBlockDic[leafBlock.rightChild] = True

        # if the parent of leafBlock is a leaf cut, it is not anymore
        if leafBlock.getParent() in self.leafCutDic.keys():
            self.leafCutDic.pop(leafBlock.getParent())

        # leafBlock has become a leaf cut (two leaf children nodes)
        self.leafCutDic[leafBlock] = True

        if leafBlock.cutDir == 0:
            # if is a row cut, record the position as key, the block (now a leafCut) containing the cut as value
            self.rowCutDic[leafBlock.cutPos] = leafBlock

        else:
            # if is a column cut, record the position as key, the block (now a leafCut) containing the cut cut as value
            self.columnCutDic[leafBlock.cutPos] = leafBlock

    # removing a cut i.e. merging two leaf nodes
    def removeLeafCut(self, leafCut):  # note that now input is a leafCut i.e. a node with two leaf children

        # remove the two leaf nodes, add itself as a leaf
        self.leafBlockDic.pop(leafCut.leftChild)
        self.leafBlockDic.pop(leafCut.rightChild)
        self.leafBlockDic[leafCut] = True

        # remove the row or column cut position, recall that key of the dict is the cut position with value being the
        # leafCut which contains the cut
        if leafCut.cutDir == 0:
            self.rowCutDic.pop(leafCut.cutPos)
        else:
            self.columnCutDic.pop(leafCut.cutPos)

        # it is no longer a leaf cut
        self.leafCutDic.pop(leafCut)
        # remove cut information from this block
        leafCut.removeCut()

        # ask if its parent becomes a leaf cut
        if leafCut.getParent() is not None:  # if this node is not the root
            # if the parent of leafCut has two leaf children after removing the cut from leafCut, add it to leafcutdic
            if ((leafCut.getParent().leftChild is not None)
                    and (leafCut.getParent().rightChild is not None)
                    and leafCut.getParent().leftChild.isLeaf()
                    and leafCut.getParent().rightChild.isLeaf()):
                self.leafCutDic[leafCut.getParent()] = True

    # get the row cut dictionary
    def getRowCutDic(self):
        return self.rowCutDic

    # get the column cut dictionary
    def getColumnCutDic(self):
        return self.columnCutDic

    # get the leaf block dictionary with key=leaf nodes, value=True
    def getLeafBlockDic(self):
        return self.leafBlockDic

    # get the leaf cut dictionary with key=leafcut nodes, value=True
    def getLeafCutDic(self):
        return self.leafCutDic

    # given xi, eta, data matrix, assign data points to each block, i.e. initialize .DataPoints
    # will initialize .marginal_lkd after introducing the data
    def cell_allocate(self, xi, eta):
        for block in self.leafBlockDic.keys():
            # for each block, register the row and col index fall inside this block
            block.register_data_trapped(xi, eta)


    def crude_plot(self, row_coord=None, col_coord=None, txt=False):
        my_cm = cm.get_cmap('viridis', len(self.leafBlockDic))
        shuffled_id = np.random.choice(range(len(self.leafBlockDic)), len(self.leafBlockDic), replace=False)
        shuffled_cm = {l: shuffled_id[i] for i, l in enumerate(self.leafBlockDic.keys())}

        plt.axes()
        leaf = list(self.leafBlockDic)
        for l in leaf:
            rectangle = plt.Rectangle((l.columnLB, l.rowLB), l.columnUB-l.columnLB, l.rowUB-l.rowLB,
                                      edgecolor='k', facecolor=my_cm(shuffled_cm[l]), alpha=0.3)
            plt.gca().add_patch(rectangle)

        if row_coord is not None:
            for r in range(len(row_coord)):
                for c in range(len(col_coord)):
                    plt.scatter(x=col_coord[c], y=row_coord[r], c="k")
                    if txt:
                        plt.annotate(text="({},{})".format(r, c), xy=(col_coord[c], row_coord[r]))

        plt.gca().invert_yaxis()
        plt.show()


def leaf_string(tree, fOutput=None):
    my_str_rep = ""
    for i, l in enumerate(tree.getLeafBlockDic().keys()):
        my_str_rep += "{}|{}|{}|{}|{};".format(l.rowLB, l.rowUB, l.columnLB, l.columnUB, i)
    if fOutput is None:
        return my_str_rep[:-1]
    else:
        fOutput.write(my_str_rep[:-1] + '\n')


def leaf_string_decoder(str_tree, xi, eta):
    """
    decode the str_tree
    :param str_tree: str representation of the partition
    :param xi: latent row coord
    :param eta: latent col coord
    :return: clustering of cells as a dictionary, cells are represented as an int (r*C+r);
    pairs is a binary matrix, two cells are in the same cluster iff pairs[cell1][cell2] = 1
    """
    R = len(xi)
    C = len(eta)
    leaf_lst = str_tree.split(";")
    leaf_array = [[] for _ in range(len(leaf_lst))]
    # decode the string
    for i, s in enumerate(leaf_lst):
        leaf_array[i] = [eval(j) for j in s.split("|")]
        # leaf_array[i] contains rowLB, rowUB, colLB, colUB, label of the block
    # each value of clustering contains the scalar index r*C+c of cells that fall inside this leaf
    clustering = {str(my_leaf[4]): [] for my_leaf in leaf_array}
    for r in range(R):
        for c in range(C):
            for vertices in leaf_array:
                if vertices[0] <= xi[r] <= vertices[1] and vertices[2] <= eta[c] <= vertices[3]:
                    clustering[str(vertices[4])] += [r*C+c]
                    break
    # pairs[i][j] = 1 means cell (k,l) and cell (m,n) in same cluster, i=k*C+l, j=m*C+n, just scalar index for the entry
    pairs = sparse.eye(R*C, dtype=np.int8).tolil()
    for leaf in clustering.keys():
        for iii in clustering[leaf]:
            for jjj in clustering[leaf]:
                pairs[iii, jjj] = 1
    return clustering, pairs


# util function, search for the idx such that elem >= sortedLst[idx] and elem < sortedLst[idx+1]
# return -1 when elem less than sortedLst[0]
def binarySearch(elem, sortedLst):
    lb = 0
    ub = len(sortedLst) - 1
    idx = 0
    while lb <= ub:
        idx = lb + (ub - lb)//2
        if sortedLst[idx] == elem:
            return idx
        elif sortedLst[idx] > elem:
            ub = idx - 1
            idx -= 1
        else:
            lb = idx + 1
    return idx


# given a cut position, cut list and intervals due to the cuts,
# find idx of the desired interval, cut i corresponds to the i+1 interval
# return resultIDX such that cutPos sits in intervalLst[resultIdx]
def findIntervalIdx(cutPos, cutLst, intervalLst):
    if cutPos == intervalLst[0][0]:  # if position is 0, return idx 0
        return 0
    elif cutPos == intervalLst[-1][-1]:  # if position is 1, return last idx
        return len(intervalLst)-1
    else:  # get the idx of the corresponding interval containing cutPos

        # find the index of cuts cutPosIdx such that cutLst[cutPosIdx] <= cutPos <= cutLst[cutPosIdx+1]
        # and cutPos in intervalLst[I+1] except for corner cases
        cutPosIdx = binarySearch(cutPos, cutLst)

        if len(intervalLst) > 1:
            # recallthe ith element in cutLst is the starting point of the i+1th interval in intervalLst
            resultIdx = cutPosIdx+1
        else:
            resultIdx = cutPosIdx

        (start, end) = intervalLst[resultIdx]  # double check if the cut position sits inside the interval
        if not (start <= cutPos <= end):
            print(start, cutPos, end)
            exit('Wrong interval for cut position!')
        return resultIdx


# xi, eta: latent coordinates
# leafNodeLst: list of leaf nodes in a MP tree
# rowCutLst, rowIntervalLst, columnCutLst, columnIntervalLst: corresponding items in a MP tree
# position of leaf cuts, interval lists are the corresponding lists defined by cuts

# rowIntervalLeafNodeDic, columnIntervalLeafNodeDic: what are they?

def initialLeafNodeStats(leafNodeLst, data, xi, eta,
                         rowIntervalLeafNodeDic, columnIntervalLeafNodeDic,
                         rowCutLst, rowIntervalLst, columnCutLst, columnIntervalLst):
    """

    :param leafNodeLst: list of leaf nodes (use list(self.LeafDic))
    :param data: 2D data array, or at least data[i,j] gives the response
    :param xi: latent row coordinates
    :param eta: latent column coordinates

    :param rowIntervalLeafNodeDic: a dictionary with key=index of row coord interval, value=list of leaf nodes
    that overlap with the horizontal band on [0,1]^2 produces by [0,1] * this row interval

    :param columnIntervalLeafNodeDic: a dictionary with key=index of column coord interval, value=list of leaf nodes
    that overlap with the vertical band on [0,1]^2 produces by this column interval * [0,1]

    :param rowCutLst: all row cuts in a list (use list(self.rowCutDic))
    :param rowIntervalLst: list of row intervals due to the cuts, l+1 of them if there are l cuts

    :param columnCutLst: all column cuts in a list (use list(self.rowCutDic))
    :param columnIntervalLst: list of column intervals due to the cuts, l+1 of them if there are l cuts

    :return:
    LeafNodetoData: dictionary, LeafNodetoData[leaf]=[list of row idx, list of col idx] list of row/col index that fall
    inside the leaf

    datumAssignedLeafNodeDic: nested dictionary, datumAssignedLeafNodeDic[i][j] gives the node that cell i,j is in

    rowIdxRowIntervalIdxLst: for each row coord, get the index of row interval it resides in
    i.e. if the ith row coordinate is rowPos, rowIdxRowIntervalIdxLst[i] returns idx such that
    rowIntervalLst[idx][0]<= rowPos < rowIntervalLst[idx][1]

    columnIdxColumnIntervalIdxLst: for each col coord, get the index of interval it resides in
    i.e. if the jth col coordinate is colPos, columnIdxColumnIntervalIdxLst[j] returns idx such that
    colIntervalLst[idx][0]<= colPos < colIntervalLst[idx][1]

    """
    # Initialize
    # dictionary with leaf node as key, list of two lists as values
    LeafNodetoData = dict([(leafNode, None) for leafNode in leafNodeLst])

    # TODO: Check this much faster alternative
    LeafNodetoData2 = {leafNode: leafNode.DataPoints for leafNode in leafNodeLst}

    # for a given data cell, that leaf node is it in?
    datumAssignedLeafNodeDic = {}

    datumAssignedLeafNodeDic2 = {_: {__: [] for __ in range(len(eta))} for _ in range(len(xi))}
    for l in leafNodeLst:
        if l.DataPoints is not None:
            for _ in l.DataPoints[0]:  # row indices
                for __ in l.DataPoints[1]:  # column indices
                    datumAssignedLeafNodeDic2[_][__] = [l]

    # for each row coord, get the index of interval it resides in
    # i.e. rowIntervalLst[idx][0]<= rowPos < rowIntervalLst[idx][1]
    # recall this is what findIntervalIdx does
    rowIdxRowIntervalIdxLst = [findIntervalIdx(rowPos, rowCutLst, rowIntervalLst)
                               for rowPos in xi]

    # for each col coord, get the index of interval it resides in
    # i.e. colIntervalLst[idx][0]<= colPos < colIntervalLst[idx][1]
    columnIdxColumnIntervalIdxLst = [findIntervalIdx(columnPos, columnCutLst, columnIntervalLst)
                                     for columnPos in eta]
    #
    # # scan over all row coordinates
    # for rowIdx, rowPos in enumerate(xi):
    #     # the index of the row interval the ith row coordinate resides in
    #     rowIntervalIdx = rowIdxRowIntervalIdxLst[rowIdx]
    #
    #     # list of all leaf nodes that overlap with the horizontal band on [0,1]^2 produces by [0,1] * this row interval
    #     rowIntervalRelatedLeafNodeDic = rowIntervalLeafNodeDic[rowIntervalIdx]
    #
    #     # column and row indices assigned to the ith row coordinate, an empty dict for now
    #     datumAssignedLeafNodeDic[rowIdx] = {}
    #     # scan over all column coordinates
    #     for columnIdx, columnPos in enumerate(eta):
    #         # add new elements to datumAssignedLeafNodeDic[rowIdx]
    #         datumAssignedLeafNodeDic[rowIdx][columnIdx] = []
    #
    #         # the index of the column interval the jth col coordinate resides in
    #         # recall  colIntervalLst[columnIntervalIdx][0]<= colPos < colIntervalLst[columnIntervalIdx][1]
    #         columnIntervalIdx = columnIdxColumnIntervalIdxLst[columnIdx]
    #
    #         # list of all leaf nodes that overlap with the vertical band on [0,1]^2
    #         # produces by this column interval * [0,1]
    #         columnIntervalRelatedLeafNodeDic = columnIntervalLeafNodeDic[columnIntervalIdx]  # what is it?
    #
    #         # for all leaf nodes that overlap with the horizontal band on [0,1]^2 produces by [0,1] * this row interval
    #         for leafNode in rowIntervalRelatedLeafNodeDic.keys():  # TODO: is it a list? or a dictionary?
    #             # check if the node also overlaps with the vertical band on [0,1]^2
    #             # produces by this column interval * [0,1]
    #             if leafNode in columnIntervalRelatedLeafNodeDic.keys():
    #                 # like a cross, with the target node in the intersection
    #                 # once we have found the leaf node data (i,j) resides in:
    #                 # assign this block to data cell i,j in the nested dict
    #                 datumAssignedLeafNodeDic[rowIdx][columnIdx].append(leafNode)   # value is a list of one element!
    #                 # also register the rowIdx, columnIdx to the leaf node
    #                 if LeafNodetoData[leafNode] is None:
    #                     # if no previous indices, add them directly
    #                     LeafNodetoData[leafNode] = [[rowIdx], [columnIdx]]
    #                 else:
    #                     # add the row or column index if it is not already in
    #                     if rowIdx not in LeafNodetoData[leafNode][0]:
    #                         LeafNodetoData[leafNode][0].append(rowIdx)
    #                     if columnIdx not in LeafNodetoData[leafNode][1]:
    #                         LeafNodetoData[leafNode][1].append(columnIdx)

    return [LeafNodetoData2, datumAssignedLeafNodeDic2,
            rowIdxRowIntervalIdxLst, columnIdxColumnIntervalIdxLst]

# TODO: if we have carried all the LeafNodetoData inside each Mondrian Block, why bother doing all these when we
# TODO: can read them off from the MP tree object? Double check

# TODO: START FROM HERE, KEEP TRACK OF POINTS FALL INSIDE EACH BLOCK!
# after all, countDic will return dict with key=leafnodes, value=dictionary of data fall inside this node, key=data label, value=number of appearance
# sumDIC will return dict that contains the number of data fall inside each leafnode
# datumAssignedLeafNodeDic: nested dict, a[i][j] returns the leaf node it resides in
# rowIdxRowIntervalIdxLst, columnIdxRowIntervalIdxLst: row and col index of intervals each data point resides in


# sampling from a discrete distribution using inverse CDF
def sampleFromDiscreteDist(distDic):
    """
    :param distDic: a dictionary with key=label, value = un normalized prob of the key
    :return: a sample from the discrete distribution
    """
    cdf = []
    sum = 0.0
    idx = 0
    selectedKey = None
    urv = random.random()
    for key, prob in distDic.items():
        sum += prob
        cdf.append((key, sum))  # get the un normalized discrete CDF for each key in the distDIC
    for i in range(0, len(cdf)):
        prob = cdf[i][1] / sum  # normalize the current prob
        if prob > urv:  # sample from the discrete distribution using inverse of the normalized CDF
            idx = i
            selectedKey = cdf[i][0]
            break
    return selectedKey


# this part should perturb the latent coordinates using Gibbs sampler
def gibbsOneDimensionStep(tree, sigma, data, idx, xi, eta,
                          intervalLst, IdxIntervalIdxLst,
                          otherIdxOtherIntervalIdxLst,
                          rowIntervalColumnIntervalLeafNodeDic, isRow,
                          row_link, col_link, fixed_ordering=False, gamma=1.):
    """
    updating one latent coordinate, row or column using Gibbs sampler
    no need
    :param tree: the MP tree
    :param sigma: error term parameter, std of the Gaussian noise term

    :param data: 2D data array, with the corresponding idx row or column removed

    :param xi, eta: latent coordinates
    :param idx: index of the row or col coordinate we want to update
    :param intervalLst: list of row intervals if isRow is true, column intervals otherwise

    :param IdxIntervalIdxLst： if isRow is true, it is the dictionary with key being the index of row coord, value
    being the index of row interval list such that the corresponding row coord is in the idx th row interval

    :param otherIdxOtherIntervalIdxLst: if isRow is true, it is the dictionary with key being the index of column coord,
    value being the index of column interval list such that the corresponding column coord is inside the
    index'th column interval, i.e. i.e. for ith coord, return j such that colIntervalLst[j] contains the ith coord.
    Similarly, for isRow being false, this is dictionary with key being index of row coord, value being the index of
    row interval list such that the corresponding row coord is inside the index'th row interval in the interval list

    :param rowIntervalColumnIntervalLeafNodeDic: rowIntervalColumnIntervalLeafNodeDic[i][j][0] gives the leaf node
    that contains the little block rowInterval[i]*colInterval[j], where i,j are interval indices

    :param: LeafNodetoData: dictionary, LeafNodetoData[leaf]=[list of row idx, list of col idx]
    list of row/col index that fall inside the leaf, WITH THE idx row or col REMOVED

    :param isRow: is the idx for row update or column?

    :param row_link, col_link:

    :param: fixed_ordering: is the ordering of row/col fixed?

    :param: gamma: tempering parameter

    :return: gibbs update of idx th coordinate
    alongside with the corresponding updated marginal lkd for the affected blocks
    """

    logLikelihoodDic = {}
    maxLogLikelihood = None
    updated_lkd_store = {}
    updated_allocation_store = {}

    if isRow:  # if is row, return idx th row of data
        otherIdxData = data[idx]
    else:  # otherwise return the idx th column
        otherIdxData = data[:, idx, :]  # TODO: FOR THIS DATA, RESPONSE IS R*C*3, more generic?

    # ####### the below procedure reallocates the rth row of the data matrix inside each row interval induced
    # by the row cuts. keep in mind that the given LeafNodetoData is generated based on the data without this
    # row or column, here we are trying to put it back into each row interval
    # TODO: START FROM THERE, ALSO CHECK THE MORE EFFICIENT LEAFNODETODATA

    if fixed_ordering:
        # for the ith row, can only choose intervals that overlaps with [xi[i-1], xi[i+1]]
        # use this to work out the right subset of intervalLst

        if isRow:
            if idx == 0:  # corner case 1, first index. bound is the support of xi[0], in term of interval,
                # xi[i] can be in or between the intervals containing xi[i-1] and xi[i+1]
                bound = [tree.root.rowLB, xi[1]]
                interval_idx_bound = list(range(0, IdxIntervalIdxLst[1]+1))  # note that the 0th coord can appear in
                # the interval containing the 1st coord, but must be less than the 1st coord
            elif idx == len(xi)-1:
                bound = [xi[idx-1], tree.root.rowUB]
                # note that when idx is the last, IdxIntervalIdxLst[idx] will be the last interval there, no more
                interval_idx_bound = list(range(IdxIntervalIdxLst[idx-1], len(intervalLst)))  # TODO: check if len(intervalLst) or IdxIntervalIdxLst[idx]+1
                # note that the Rth coord can appear in
                # the interval containing the R-1th coord, but must be greater than the R-1th coord coord
            else:
                bound = [xi[idx-1], xi[idx+1]]
                interval_idx_bound = list(range(IdxIntervalIdxLst[idx-1], IdxIntervalIdxLst[idx+1]+1))
                # note that the ith coord can appear in
                # the interval containing the i-1th to the i+1 coord , but must be greater than the i-1th coord coord
                # and lower than the i+1 coord

        else:  # if isRow is false, IdxIntervalIdxLst will be the column list, no need to change
            if idx == 0:
                bound = [tree.root.columnLB, eta[1]]
                interval_idx_bound = list(range(0, IdxIntervalIdxLst[1]+1))
            elif idx == len(eta)-1:
                bound = [eta[idx-1], tree.root.columnUB]
                interval_idx_bound = list(range(IdxIntervalIdxLst[idx-1], len(intervalLst)))  # TODO: check if len(intervalLst) or IdxIntervalIdxLst[idx]+1
            else:
                bound = [eta[idx-1], eta[idx+1]]
                interval_idx_bound = list(range(IdxIntervalIdxLst[idx-1], IdxIntervalIdxLst[idx+1]+1))
        # worked our the legal intervals

        for intervalIdx in interval_idx_bound:  # this is the "legal" interval indices that the ith coord can be in
            tempDataPoints = {}
            tmpLeafNodeDic = {}
            # the other index, row or col with key=idx, label=data in otherIdxData
            # if isRow true, it means scan all columns at row idx, if is not row, it means scan all rows at this column idx

            # THIS FOR LOOP REALLOCATES THE ITH ROW OR COL SO THE ITH LATENT COORD IS IN THE intervalIdx th INTERVAL
            for otherIdx, datum in enumerate(otherIdxData):
                if datum is not None:
                    # if isRow is true, input COLUMN index i, get the index of the COL interval that the ith COL is in
                    # similar effect for isRow being false
                    otherIntervalIdx = otherIdxOtherIntervalIdxLst[otherIdx]

                    # for each data point, if we put the ith row coord in intervalIdx th row interval, then which leaf block
                    # contains the latent coord of data[i, otherIdx]?
                    if isRow:  # if is row, row interval idx is the idx th of the row interval lists,
                        rowIntervalIdx = intervalIdx  # we put the ith row in the index th row interval
                        columnIntervalIdx = otherIntervalIdx  # the corresponding col interval containing the latent coord
                        # of data[i, otherIdx]
                    else:
                        rowIntervalIdx = otherIntervalIdx
                        columnIntervalIdx = intervalIdx

                    # the leaf node that contains this small block, i.e. the leaf node that the data[i, otherIdx] falls into
                    # if we put the latent coord of the ith row in the intervalIdx th row interval
                    leafNode = rowIntervalColumnIntervalLeafNodeDic[rowIntervalIdx][columnIntervalIdx][0]

                    # record the leaf nodes affected by adding this data cell back
                    tmpLeafNodeDic[leafNode] = None  # value is kind of a place holder, only need this leaf node

                    # additional points fall inside this leafNode
                    # if not seen before, first copy the existing points in leafNode, then add this new one
                    # if have seen before, just add the new one
                    if leafNode not in tempDataPoints:
                        # if leafNode is new to tempDataPoints:
                        if leafNode.DataPoints is not None:
                            tempDataPoints[leafNode] = [leafNode.DataPoints[0][:], leafNode.DataPoints[1][:]]
                        else:
                            tempDataPoints[leafNode] = [[], []]

                    # now add the new data cell [idx, other idx] or [other idx, idx] to it
                    if isRow:
                        if idx not in tempDataPoints[leafNode][0]:
                            tempDataPoints[leafNode][0].append(idx)
                        if otherIdx not in tempDataPoints[leafNode][1]:
                            tempDataPoints[leafNode][1].append(otherIdx)
                    else:
                        if idx not in tempDataPoints[leafNode][1]:
                            tempDataPoints[leafNode][1].append(idx)
                        if otherIdx not in tempDataPoints[leafNode][0]:
                            tempDataPoints[leafNode][0].append(otherIdx)

            # record the updated allocation if we put the ith row into this interval
            updated_allocation_store[intervalIdx] = tempDataPoints

            # ONCE WE HAVE REALLOCATED THE ITH ROW, COMPUTE THE UPDATED LKD
            # NOTE THAT WE ONLY NEED TO WORRY ABOUT THE LEAF NODES INSIDE tmpLeafNodeDic,
            # IF we have removed the ith row/col before, then adding this row/col will only affect leaf nodes
            # in tmpLeafNodeDic, EVERYTHING ELSE REMAINS SAME
            # IMAGINE dividing the updated joint lkd by the joint lkd WITHOUT THE idx row/col, only need to worry about
            # the ratio of lkd with vs without the idx row for leaf nodes in tmpLeafNodeDic

            # TODO: compare with block level marginal lkd method
            interval_specific_old_lkd = 0  # for this choice of row interval, cumulate the old lkd for each block
            interval_specific_updated_lkd = {}  # for this choice of row interval, record updated lkd for each block
            for leafNode in tmpLeafNodeDic:  # tmpLeafNodeDic： all leaf nodes affected by adding the row/col
                temp_row, temp_col = cov_idx_to_data_idx(tempDataPoints[leafNode][0], tempDataPoints[leafNode][1],
                                                         row_link, col_link)
                candidate_data = data[temp_row][:, temp_col, :].reshape(-1, 2)
                # recall tempDataPoints[leafNode][0], tempDataPoints[leafNode][1] contains all the row/col indices of the
                # updated partition i.e. after we have inserted the ith row
                interval_specific_updated_lkd[leafNode] = marginal_lkd_MC(candidate_data, sigma)
                # while leafNode.marginal_lkd should be the marginal lkd of this leaf node without the ith row
                interval_specific_old_lkd += leafNode.marginal_lkd

            # note that key is the interval index, value is the log lkd component
            # for this row/col interval choice, finally get the log lkd up additive constant!
            logLikelihoodDic[intervalIdx] = sum(interval_specific_updated_lkd.values()) - interval_specific_old_lkd
            # also store the updated marginal lkd for each block at this interval, will use for updating the blocks
            updated_lkd_store[intervalIdx] = interval_specific_updated_lkd

            if maxLogLikelihood is None:  # keep track of the max of log lkd for each row interval, for numerical stability
                maxLogLikelihood = logLikelihoodDic[intervalIdx]
            elif logLikelihoodDic[intervalIdx] > maxLogLikelihood:
                maxLogLikelihood = logLikelihoodDic[intervalIdx]

        # ONCE WE HAVE TRIED ALL INTERVALS, DO GIBBS UPDATE FOR CHOOSING THE INTERVAL
        # sum1 is the "normalizing constant"
        sum1 = 0.0

        # key is interval index, value should be the corresponding un-normalized probability choosing this interval
        distDic = {}
        # UNIFORM CUT POSITION HAS THE ADVANTAGE OF GIVING EASY-TO-COMPUTE, ALMOST TRACTABLE DISTRIBUTION ON LATENT COORD
        # logLikelihoodDic is dict with key=interval index, value= corresponding lkd when putting the ith row in the
        # chosen interval
        # Note that when ith and i-1th or i and i+1th are in the same interval, the boundry show involve the i+1
        # or i-1 coord instead of the simple start/end

        for intervalIdx, logLikelihood in logLikelihoodDic.items():  # contains log lkd for each legal cut
            start = max(bound[0], intervalLst[intervalIdx][0])  # since the ith coord can not go beyond i-1 th coord
            end = min(bound[1], intervalLst[intervalIdx][1])  # since the ith coord can not go beyond i+1 th coord
            # record un-normalized prob for each interval

            # TEMPERING MODIFICATION 1: when targeting a tempered posterior, we calculate the marginal lkd as usual,
            # but when doing update, multiply log probabilities by the tempering parameter.

            distDic[intervalIdx] = (end - start) * np.exp(gamma*(logLikelihood - maxLogLikelihood))
            sum1 += distDic[intervalIdx]
        # now the key of distDic will only contain indices of the legal intervals

        # sample an interval in distDIC, prob determined by probs for each interval
        chosen_idx = sampleFromDiscreteDist(distDic)
        # return the gibbs updated interval idx, and the updated marginal lkd for all leaf nodes affected by it
        # again recall we start from the configuration WITHOUT the ith row or column
        return chosen_idx, updated_lkd_store[chosen_idx], updated_allocation_store[chosen_idx], \
               [max(bound[0], intervalLst[chosen_idx][0]), min(bound[1], intervalLst[chosen_idx][1])]

    # TODO: the two chunks works pretty much the same, merge them for more concise code, leave it for now
    else:  # this is the unconstrained ordering case
        # give a interval list (either col or row), if we put the row or column in this interval...
        for intervalIdx in range(0, len(intervalLst)):
            tempDataPoints = {}
            tmpLeafNodeDic = {}
            # the other index, row or col with key=idx, label=data in otherIdxDataDic
            # if isRow true, it means scan all columns at row idx, if is not row, it means scan all rows at this column idx

            # THIS FOR LOOP REALLOCATES THE ITH ROW OR COL SO THE ITH LATENT COORD IS IN THE intervalIdx th INTERVAL
            for otherIdx, datum in enumerate(otherIdxData):
                if datum is not None:
                    # if isRow is true, input COLUMN index i, get the index of the COL interval that the ith COL is in
                    # similar effect for isRow being false
                    otherIntervalIdx = otherIdxOtherIntervalIdxLst[otherIdx]

                    # for each data point, if we put the ith row coord in intervalIdx th row interval, then which leaf block
                    # contains the latent coord of data[i, otherIdx]?
                    if isRow:  # if is row, row interval idx is the idx th of the row interval lists,
                        rowIntervalIdx = intervalIdx  # we put the ith row in the index th row interval
                        columnIntervalIdx = otherIntervalIdx  # the corresponding col interval containing the latent coord
                        # of data[i, otherIdx]
                    else:
                        rowIntervalIdx = otherIntervalIdx
                        columnIntervalIdx = intervalIdx

                    # the leaf node that contains this small block, i.e. the leaf node that the data[i, otherIdx] falls into
                    # if we put the latent coord of the ith row in the intervalIdx th row interval
                    leafNode = rowIntervalColumnIntervalLeafNodeDic[rowIntervalIdx][columnIntervalIdx][0]

                    # record the leaf nodes affected by adding this data cell back
                    if leafNode not in tmpLeafNodeDic:
                        tmpLeafNodeDic[leafNode] = None  # kind of a place holder

                    # additional points fall inside this leafNode
                    # if not seen before, first copy the existing points in leafNode, then add this new one
                    # if have seen before, just add the new one
                    if leafNode not in tempDataPoints:
                        # if leafNode is new to tempDataPoints:
                        if leafNode.DataPoints is not None:
                            tempDataPoints[leafNode] = [leafNode.DataPoints[0][:], leafNode.DataPoints[1][:]]
                        else:
                            tempDataPoints[leafNode] = [[],[]]

                    # now add the new data cell [idx, other idx] or [other idx, idx] to it
                    if isRow:
                        if idx not in tempDataPoints[leafNode][0]:
                            tempDataPoints[leafNode][0].append(idx)
                        if otherIdx not in tempDataPoints[leafNode][1]:
                            tempDataPoints[leafNode][1].append(otherIdx)
                    else:
                        if idx not in tempDataPoints[leafNode][1]:
                            tempDataPoints[leafNode][1].append(idx)
                        if otherIdx not in tempDataPoints[leafNode][0]:
                            tempDataPoints[leafNode][0].append(otherIdx)

            # record the updated allocation if we put the ith row into this interval
            updated_allocation_store[intervalIdx] = tempDataPoints

            # ONCE WE HAVE REALLOCATED THE ITH ROW, COMPUTE THE UPDATED LKD
            # NOTE THAT WE ONLY NEED TO WORRY ABOUT THE LEAF NODES INSIDE tmpLeafNodeDic,
            # IF we have removed the ith row/col before, then adding this row/col will only affect leaf nodes
            # in tmpLeafNodeDic, EVERYTHING ELSE REMAINS SAME
            # IMAGINE dividing the updated joint lkd by the joint lkd WITHOUT THE idx row/col, only need to worry about
            # the ratio of lkd with vs without the idx row for leaf nodes in tmpLeafNodeDic
            # TODO: compare it with block level marginal lkd method
            interval_specific_old_lkd = 0  # for this choice of row interval, cumulate the old lkd for each block
            interval_specific_updated_lkd = {}  # for this choice of row interval, record updated lkd for each block
            for leafNode in tmpLeafNodeDic:  # tmpLeafNodeDic： all leaf nodes affected by adding the row/col
                temp_row, temp_col = cov_idx_to_data_idx(tempDataPoints[leafNode][0], tempDataPoints[leafNode][1],
                                                         row_link, col_link)
                candidate_data = data[temp_row][:, temp_col, :].reshape(-1, 2)
                # recall tempDataPoints[leafNode][0], tempDataPoints[leafNode][1] contains all the row/col indices of the
                # updated partition i.e. after we have inserted the ith row
                interval_specific_updated_lkd[leafNode] = marginal_lkd_MC(candidate_data, sigma)
                # while leafNode.marginal_lkd should be the marginal lkd of this leaf node without the ith row
                interval_specific_old_lkd += leafNode.marginal_lkd

            # note that key is the interval index, value is the log lkd component
            # for this row/col interval choice, finally get the log lkd up additive constant!
            logLikelihoodDic[intervalIdx] = sum(interval_specific_updated_lkd.values()) - interval_specific_old_lkd
            # also store the updated marginal lkd for each block at this interval, will use for updating the blocks
            updated_lkd_store[intervalIdx] = interval_specific_updated_lkd

            if maxLogLikelihood is None:  # keep track of the max of log lkd for each row interval, for numerical stability
                maxLogLikelihood = logLikelihoodDic[intervalIdx]
            elif logLikelihoodDic[intervalIdx] > maxLogLikelihood:
                maxLogLikelihood = logLikelihoodDic[intervalIdx]

        # ONCE WE HAVE TRIED ALL INTERVALS, DO GIBBS UPDATE FOR CHOOSING THE INTERVAL
        # sum1 is the "normalizing constant"
        sum1 = 0.0

        # key is interval index, value should be the corresponding un-normalized probability choosing this interval
        distDic = {}
        # UNIFORM CUT POSITION HAS THE ADVANTAGE OF GIVING EASY-TO-COMPUTE, ALMOST TRACTABLE DISTRIBUTION ON LATENT COORD
        # logLikelihoodDic is dict with key=interval index, value= corresponding lkd when putting the ith row in the
        # chosen interval
        for intervalIdx, logLikelihood in logLikelihoodDic.items():
            (start, end) = intervalLst[intervalIdx]
            # record un-normalized prob for each interval

            # TEMPERING MODIFICATION2: multiply the Gibbs update log prob by gamma,
            # again we calculate and store the marginal lkd as usual

            distDic[intervalIdx] = (end - start) * np.exp(gamma*(logLikelihood - maxLogLikelihood))
            sum1 += distDic[intervalIdx]

        # sample an interval in distDIC, prob determined by probs for each interval
        chosen_idx = sampleFromDiscreteDist(distDic)

        # return the gibbs updated interval idx, and the updated marginal lkd for all leaf nodes affected by it
        # again recall we start from the configuration WITHOUT the ith row or column
        return chosen_idx, updated_lkd_store[chosen_idx], updated_allocation_store[chosen_idx], \
               intervalLst[chosen_idx]


# full Gibbs sampler for all or a randomly selected subset row and col coordinates
def Gibbs_for_coord(sigma, data, xi, eta, tree, row_link, col_link, maxGibbsIteration = 1, isPreGibbsLikelihood=True,
                    fixed_ordering_col=False, fixed_ordering_row=False, p_row=0.2, p_col=0.2, gamma=1.):
    """
    full Gibbs sampler for updating all row and col coordinates with topology of tree being fixed
    :param sigma: std of prior on Gaussian noises
    :param data: data matrix, at least data[i][j] or data[i,j] returns the data in (i,j) cell
    :param xi: all latent row coordinates
    :param eta: all latent col coordinates
    :param tree: the tree object
    :param row_link, col_link:
    :param maxGibbsIteration: max number of Gibbs iterations
    :param isPreGibbsLikelihood: do we calculate the lkd given parameters before running the Gibbs sampler? or after?
    :param fixed_ordering_col: do we fix the ordering of the columns?
    :param fixed_ordering_row: do we fix the ordering of the rows?
    :param p_row, p_col: the proportion of entries to be updated, e.g. 0.2 means 20% of randomly selected entries
    :param gamma: tempering parameter
    :return: updated xi, eta, and log likelihood before update if isPre = True, else the log lkd after the Gibbs updates
    """

    # given the tree, i.e. the Mondrian tree, record all its leaf blocks
    leafBlockLst = list(tree.getLeafBlockDic().keys())
    # record all row cut positions, recall that key=cut position, value=node of the cut
    rowCutLst = list(tree.getRowCutDic().keys())
    # similar, all col cut positions
    columnCutLst = list(tree.getColumnCutDic().keys())
    # print("# of row cuts: {}, # of column cuts: {}".format(len(rowCutLst), len(columnCutLst)))
    # sort the row and col cuts
    rowCutLst = sorted(list(rowCutLst))
    columnCutLst = sorted(list(columnCutLst))
    # initialize the row interval list
    rowIntervalLst = [(tree.root.rowLB, tree.root.rowUB)]
    # get all row intervals produced by row cuts, these cuts are much finer than the actual tree
    if len(rowCutLst) > 0:
        rowIntervalLst = [(tree.root.rowLB, rowCutLst[0])] + [(rowCutLst[idx], rowCutLst[idx+1])
                                                              for idx in range(0, len(rowCutLst)-1)] + [(rowCutLst[-1], tree.root.rowUB)]
    # initialize the col interval list
    columnIntervalLst = [(tree.root.columnLB, tree.root.columnUB)]
    # similar, get all col intervals produced by col cuts
    if len(columnCutLst) > 0:
        columnIntervalLst = [(tree.root.columnLB, columnCutLst[0])] + [(columnCutLst[idx], columnCutLst[idx+1])
                                                                       for idx in range(0, len(columnCutLst)-1)] + [(columnCutLst[-1], tree.root.columnUB)]

    # given idx of a row/col interval, find all leaf that overlaps with the horizontal row interval*[0,1] or
    # the vertical [0,1]*that col interval
    rowIntervalLeafNodeDic = {}
    columnIntervalLeafNodeDic = {}
    # scan over each leaf
    for i, leafNode in enumerate(leafBlockLst):
        # for each block, work out the row intervals it overlaps with
        # find the index of the row interval that the starting and ending points of the block are in
        # recall findIntervalIdx returns the interval index i such that interval[i][0]<= x < interval[i][1]
        rowLBIdx = findIntervalIdx(leafNode.rowLB, rowCutLst, rowIntervalLst)
        rowUBIdx = findIntervalIdx(leafNode.rowUB, rowCutLst, rowIntervalLst)

        # for each of the covered row interval
        for rowIntervalIdx in range(rowLBIdx, rowUBIdx+1):
            (start, end) = rowIntervalLst[rowIntervalIdx]
            # if the leaf node do cover this interval, register this leaf node to the index of this interval
            if leafNode.rowLB <= start and leafNode.rowUB >= end:
                if rowIntervalIdx in rowIntervalLeafNodeDic.keys():
                    rowIntervalLeafNodeDic[rowIntervalIdx][leafNode] = 0  # 0 is just a place holder
                else:
                    rowIntervalLeafNodeDic[rowIntervalIdx] = {leafNode: 0}

        # similar thing, for column intervals
        columnLBIdx = findIntervalIdx(leafNode.columnLB, columnCutLst, columnIntervalLst)
        columnUBIdx = findIntervalIdx(leafNode.columnUB, columnCutLst, columnIntervalLst)
        for columnIntervalIdx in range(columnLBIdx, columnUBIdx+1):
            (start, end) = columnIntervalLst[columnIntervalIdx]
            if leafNode.columnLB <= start and leafNode.columnUB >= end:
                if columnIntervalIdx in columnIntervalLeafNodeDic.keys():
                    columnIntervalLeafNodeDic[columnIntervalIdx][leafNode] = 0
                else:
                    columnIntervalLeafNodeDic[columnIntervalIdx] = {leafNode: 0}

    # given row and col interval idx i,j, return the node that contains this little block AS A LENGTH 1 LIST smh
    # i.e. Dic[i][j][0] will return the node in which block does the little square row interval i * col interval j live in
    rowIntervalColumnIntervalLeafNodeDic = {}
    for rowIntervalIdx in range(0, len(rowIntervalLst)):
        rowIntervalColumnIntervalLeafNodeDic[rowIntervalIdx] = {}
        for columnIntervalIdx in range(0, len(columnIntervalLst)):
            tmpLst = []
            for leafNode in rowIntervalLeafNodeDic[rowIntervalIdx].keys():
                if leafNode in columnIntervalLeafNodeDic[columnIntervalIdx].keys():
                    tmpLst.append(leafNode)
            if len(tmpLst) > 1:
                for _ in tmpLst:
                    print(_)
                exit('There should be only one leaf block!')
            rowIntervalColumnIntervalLeafNodeDic[rowIntervalIdx][columnIntervalIdx] = tmpLst

    # initialize a bunch of quantities
    # recall that countDic is a dict with key=leaf node, value=dict of data with key=label, value=number of appearance
    # sumDic is a dict with key=leaf Node, value=number of data fall inside this node
    # datumAssignedLeafNodeDic: dict[i][j] gives the leaf node cell i,j resides in as a length 1 list
    # rowIdxRowIntervalIdxLst: given index of latent row coord, return the index of row interval that coord is in
    # columnIdxColumnIntervalIdxLst: given index of latent col coord, return the index of col interval that coord is in

    [LeafNodetoData, datumAssignedLeafNodeDic, rowIdxRowIntervalIdxLst,
     columnIdxColumnIntervalIdxLst] = initialLeafNodeStats(leafBlockLst, data, xi, eta,
                                                           rowIntervalLeafNodeDic, columnIntervalLeafNodeDic,
                                                           rowCutLst, rowIntervalLst, columnCutLst, columnIntervalLst)

    logLikelihood = 0
    # do you want to compute the marginal log lkd of the current configuration with given sigma before the gibbs update?
    if isPreGibbsLikelihood:
        preGibbsLogLikelihood = 0
        for leafNode in leafBlockLst:
            if leafNode.marginal_lkd is not None:   # none or 0.0 if there is no data point in the block
                # recall that we store the random estimate of marginal lkd in each block
                preGibbsLogLikelihood += leafNode.marginal_lkd
        logLikelihood = preGibbsLogLikelihood

    # actual gibbs updates
    for itr in range(0, maxGibbsIteration):
        # updates for rows
        # for rowIdx, rowPos in enumerate(xi):  # NOTE: if we want to update every row
        # selected_rows = sorted(np.random.choice(list(range(len(xi))), size=int(p_row*len(xi)), replace=False))
        selected_rows = sorted(random.sample(list(range(len(xi))), k=min(len(xi), int(p_row*len(xi))+1)))
        for rowIdx in selected_rows:  # NOTE: only updating a randomly selected p_row portion of the rows
            # for each row coordinate, first sample a new interval, then sample new coord uniformly from this interval,
            # then update block information, similar for cols

            # remove the ith row from the Mondrian tree records, update the marginal log lkd of the block without it
            leaf_affected = {}  # record the leaf nodes affected by removing the ith row
            for columnIdx, columnPos in enumerate(eta):
                leafNodes = datumAssignedLeafNodeDic[rowIdx][columnIdx]
                if len(leafNodes) > 1:
                    exit('One datum can only belong to one leaf block!')
                else:
                    leafNode = leafNodes[0]
                    # if it is a new node we have not seen before
                    if leafNode not in leaf_affected:
                        leaf_affected[leafNode] = None
                        # recall leafNode.DataPoints should be None if no data is in the block, otherwise a list
                        leafNode.DataPoints[0].remove(rowIdx)
                        # if removing rowIdx leads to no row idx in .DataPoints, change it to None
                        if leafNode.DataPoints[0] == []:
                            leafNode.DataPoints = None
                    # if it is already in the affected blocks, double check
                    else:
                        # .DataPoints can only be None or must not contain rowIdx
                        if leafNode.DataPoints is not None and rowIdx in leafNode.DataPoints[0]:
                            exit('something wrong with the removing row/col from tree process')

                    # if leafNode.DataPoints is not None and rowIdx in leafNode.DataPoints[0]:
                    #     # remove the row idx from this leaf block if it is in this block
                    #     leafNode.DataPoints[0].remove(rowIdx)
                    # elif leafNode.DataPoints[0] == []:
                    #     # if rowIdx has been removed, do nothing, if leafNode.DataPoints[0]=[], remove all column idx
                    #     leafNode.DataPoints = None

            # update the marginal lkd for the affected blocks. These are now lkd WITHOUT the ith row
            # i.e. all blocks now contain corresponding marginal lkd without the ith row
            # TODO: compare it to block level method?
            for leafNode in leaf_affected.keys():
                if leafNode.DataPoints is None:
                    leafNode.marginal_lkd = 0.0
                else:
                    temp_row, temp_col = cov_idx_to_data_idx(leafNode.DataPoints[0], leafNode.DataPoints[1],
                                                             row_link, col_link)
                    candidate_data = data[temp_row][:, temp_col, :].reshape(-1, 2)
                    # recall leafNode.DataPoints contains all the updated row/col indices for affected leaf nodes
                    leafNode.marginal_lkd = marginal_lkd_MC(candidate_data, sigma)

            # once we have removed the rth row , perform one gibbs step, recall it returns 4 things,
            # chosen idx of interval, dictionary of marginal lkd of the affected blocks, updated data allocation
            # and the chosen interval.
            # when fixed_ordering=False, the chosen interval and the idxth interval is the same
            # then fixed_ordering=True, it is truncated according to the i-1 and i+1th coord

            # TEMPERING MODIFICATION 3: NOW ROW COORD ARE UPDATED ACCORDING TO THE TEMPERED DISTRIBUTION

            newRowIntervalIdx, updated_marginal, \
            updated_allocation, bound = gibbsOneDimensionStep(tree, sigma, data, rowIdx, xi, eta,
                                                              rowIntervalLst, rowIdxRowIntervalIdxLst,
                                                              columnIdxColumnIntervalIdxLst,
                                                              rowIntervalColumnIntervalLeafNodeDic,
                                                              True, row_link, col_link, fixed_ordering_row, gamma)

            if not fixed_ordering_row:
                # when fixed_ordering = false, bound should be the newIdx th interval
                if rowIntervalLst[newRowIntervalIdx] != bound:
                    exit('error in Gibbs update for row coord')
            else:
                # when fixed_ordering = true, bound should be the newIdx th interval truncated at i-1 and i+1 th coord
                if rowIntervalLst[newRowIntervalIdx][0] > bound[0] or rowIntervalLst[newRowIntervalIdx][1] < bound[1]:
                    exit('error in Gibbs update for row coord')

            # assign the rth row coord to a new row interval
            rowIdxRowIntervalIdxLst[rowIdx] = newRowIntervalIdx
            # after the interval idx is chosen, sample xi uniformly over the interval
            (start, end) = bound
            newRowPos = start + (end - start) * random.random()
            # update the rth row coordinate
            xi[rowIdx] = newRowPos

            # update the data<->tree assignment after making the update, note that we only have to worry about this row
            for leafNode, points in updated_allocation.items():
                leafNode.DataPoints = points  # update data allocation
                if leafNode not in list(updated_marginal.keys()):
                    exit("Sth wrong when recording the updated marginal lkds")
                leafNode.marginal_lkd = updated_marginal[leafNode]  # update marginal lkd of the affected blocks

            for columnIdx, columnPos in enumerate(eta):
                columnIntervalIdx = columnIdxColumnIntervalIdxLst[columnIdx]
                # now the i,j cell is in a different node! the ith row coord now link to interval newRowIntervalIdx
                datumAssignedLeafNodeDic[rowIdx][columnIdx] = rowIntervalColumnIntervalLeafNodeDic[newRowIntervalIdx][columnIntervalIdx]

        # similar gibbs updates for each column coordinate
        # for columnIdx, columnPos in enumerate(eta):
        # selected_cols = sorted(np.random.choice(list(range(len(eta))), size=int(p_col*len(eta)), replace=False))
        selected_cols = sorted(random.sample(list(range(len(eta))), k=min(len(eta), int(p_col*len(eta))+1)))
        for columnIdx in selected_cols:
            # remove jth column from the data, update countDic, sumDic
            leaf_affected = {}
            for rowIdx, rowPos in enumerate(xi):
                leafNodes = datumAssignedLeafNodeDic[rowIdx][columnIdx]
                if len(leafNodes) > 1:
                    exit('One datum can only belong to one leaf block!')
                else:
                    # here leafNode.DataPoints should always be non-None since rowIdx,colIdx is in it,
                    # but may become empty if we remove it
                    leafNode = leafNodes[0]
                    # if it is a new node we have not seen before
                    if leafNode not in leaf_affected:
                        leaf_affected[leafNode] = None
                        # recall leafNode.DataPoints should be None if no data is in the block, otherwise a list of list
                        leafNode.DataPoints[1].remove(columnIdx)
                        # if removing colIdx leads to no col idx in .DataPoints, change it to None
                        if leafNode.DataPoints[1] == []:
                            leafNode.DataPoints = None
                    # if it is already in the affected blocks, double check
                    else:
                        # .DataPoints can only be None or must not contain rowIdx
                        if leafNode.DataPoints is not None and columnIdx in leafNode.DataPoints[1]:
                            exit('something wrong with the removing row/col from tree process')

                # leafNodes = datumAssignedLeafNodeDic[rowIdx][columnIdx]
                # if len(leafNodes) > 1:
                #     exit('One datum can only belong to one leaf block!')
                # else:
                #     leafNode = leafNodes[0]
                #     if leafNode not in leaf_affected:
                #         leaf_affected[leafNode] = None
                #     if leafNode.DataPoints is not None and columnIdx in leafNode.DataPoints[1]:
                #         leafNode.DataPoints[1].remove(columnIdx)
                #     elif leafNode.DataPoints[1] == []:
                #         leafNode.DataPoints = None

            # update the marginal lkd for the affected blocks. These are now lkd WITHOUT the jth col
            # i.e. every block contains corresponding marginal lkd with the data set without the jth column
            # TODO: compare it to block level method?
            for leafNode in leaf_affected.keys():
                if leafNode.DataPoints is None:
                    leafNode.marginal_lkd = 0.0
                else:
                    temp_row, temp_col = cov_idx_to_data_idx(leafNode.DataPoints[0], leafNode.DataPoints[1],
                                                             row_link, col_link)
                    candidate_data = data[temp_row][:, temp_col, :].reshape(-1, 2)
                    # recall leafNode.DataPoints contains all the updated row/col indices for affected leaf nodes
                    leafNode.marginal_lkd = marginal_lkd_MC(candidate_data, sigma)

            # TEMPERING MODIFICATION 4: NOW COLUMN COORD ARE UPDATED ACCORDING TO THE TEMPERED DISTRIBUTION
            # do one gibbs update for col coordinate
            newColumnIntervalIdx, updated_marginal, \
            updated_allocation, bound = gibbsOneDimensionStep(tree, sigma, data, columnIdx, xi, eta,
                                                              columnIntervalLst, columnIdxColumnIntervalIdxLst,
                                                              rowIdxRowIntervalIdxLst,
                                                              rowIntervalColumnIntervalLeafNodeDic,
                                                              False, row_link, col_link, fixed_ordering_col, gamma)

            if not fixed_ordering_col:
                # when fixed_ordering = false, bound should be the newIdx th interval
                if columnIntervalLst[newColumnIntervalIdx] != bound:
                    exit('error in Gibbs update for col coord')
            else:
                # when fixed_ordering = true, bound should be the newIdx th interval truncated at i-1 and i+1 th coord
                if columnIntervalLst[newColumnIntervalIdx][0] > bound[0] or columnIntervalLst[newColumnIntervalIdx][1] < bound[1]:
                    exit('error in Gibbs update for col coord')

            # assign the jth col coord to the new interval
            columnIdxColumnIntervalIdxLst[columnIdx] = newColumnIntervalIdx

            # update the actual col coord
            (start, end) = bound
            newColumnPos = start + (end - start) * random.random()
            eta[columnIdx] = newColumnPos

            # update the data<->tree assignment after making the update, note that we only have to worry about this row
            for leafNode, points in updated_allocation.items():
                leafNode.DataPoints = points  # update the data allocation
                if leafNode not in list(updated_marginal.keys()):
                    exit("Sth wrong when recording the updated marginal lkds")
                leafNode.marginal_lkd = updated_marginal[leafNode]  # update the marginal lkd for affected blocks

            for rowIdx, rowPos in enumerate(xi):
                rowIntervalIdx = rowIdxRowIntervalIdxLst[rowIdx]  # get the row interval for ith row coord
                datumAssignedLeafNodeDic[rowIdx][columnIdx] = rowIntervalColumnIntervalLeafNodeDic[rowIntervalIdx][newColumnIntervalIdx]

    if not isPreGibbsLikelihood:
        postGibbsLogLikelihood = 0
        for leafNode in leafBlockLst:
            if leafNode.marginal_lkd is not None:   # none or 0 if there is no data point in the block
                # recall that we store the random estimate of marginal lkd in each block
                postGibbsLogLikelihood += leafNode.marginal_lkd
        logLikelihood = postGibbsLogLikelihood

    return [xi, eta, logLikelihood]



def MHOneDimensionStep(tree, sigma, data, idx, xi, eta, old_lkd,
                       intervalLst, IdxIntervalIdxLst,
                       otherIdxOtherIntervalIdxLst,
                       rowIntervalColumnIntervalLeafNodeDic, isRow,
                       row_link, col_link, fixed_ordering=False, gamma=1.):
    """
    updating one latent coordinate, row or column using Gibbs sampler
    no need
    :param tree: the MP tree
    :param sigma: error term parameter, std of the Gaussian noise term

    :param data: 2D data array, with the corresponding idx row or column removed

    :param xi, eta: latent coordinates
    :param old_lkd: lkd of the current state
    :param idx: index of the row or col coordinate we want to update
    :param intervalLst: list of row intervals if isRow is true, column intervals otherwise

    :param IdxIntervalIdxLst： if isRow is true, it is the dictionary with key being the index of row coord, value
    being the index of row interval list such that the corresponding row coord is in the idx th row interval

    :param otherIdxOtherIntervalIdxLst: if isRow is true, it is the dictionary with key being the index of column coord,
    value being the index of column interval list such that the corresponding column coord is inside the
    index'th column interval, i.e. i.e. for ith coord, return j such that colIntervalLst[j] contains the ith coord.
    Similarly, for isRow being false, this is dictionary with key being index of row coord, value being the index of
    row interval list such that the corresponding row coord is inside the index'th row interval in the interval list

    :param rowIntervalColumnIntervalLeafNodeDic: rowIntervalColumnIntervalLeafNodeDic[i][j][0] gives the leaf node
    that contains the little block rowInterval[i]*colInterval[j], where i,j are interval indices

    :param: LeafNodetoData: dictionary, LeafNodetoData[leaf]=[list of row idx, list of col idx]
    list of row/col index that fall inside the leaf, WITH THE idx row or col REMOVED

    :param isRow: is the idx for row update or column?

    :param row_link, col_link:

    :param: fixed_ordering: is the ordering of row/col fixed?

    :param: gamma: tempering parameter

    :return: gibbs update of idx th coordinate
    alongside with the corresponding updated marginal lkd for the affected blocks
    """

    updated_lkd_store = {}
    updated_allocation_store = {}

    if isRow:  # if is row, return idx th row of data
        otherIdxData = data[idx]
    else:  # otherwise return the idx th column
        otherIdxData = data[:, idx, :]  # TODO: FOR THIS DATA, RESPONSE IS R*C*2, more generic?

    # ####### the below procedure reallocates the rth row of the data matrix inside each row interval induced
    # by the row cuts. keep in mind that the given LeafNodetoData is generated based on the data without this
    # row or column, here we are trying to put it back into each row interval
    # TODO: START FROM THERE, ALSO CHECK THE MORE EFFICIENT LEAFNODETODATA

    if fixed_ordering:
        # for the ith row, can only choose intervals that overlaps with [xi[i-1], xi[i+1]]
        # use this to work out the right subset of intervalLst

        if isRow:
            if idx == 0:  # corner case 1, first index. bound is the support of xi[0], in term of interval,
                # xi[i] can be in or between the intervals containing xi[i-1] and xi[i+1]
                bound = [tree.root.rowLB, xi[1]]
                interval_idx_bound = list(range(0, IdxIntervalIdxLst[1]+1))  # note that the 0th coord can appear in
                # the interval containing the 1st coord, but must be less than the 1st coord
            elif idx == len(xi)-1:
                bound = [xi[idx-1], tree.root.rowUB]
                # note that when idx is the last one, can be anything from IdxIntervalIdxLst[idx-1] to the last element in intervalLst
                interval_idx_bound = list(range(IdxIntervalIdxLst[idx-1], len(intervalLst)))
                # note that the Rth coord can appear in
                # the interval containing the R-1th coord, but must be greater than the R-1th coord coord
            else:
                bound = [xi[idx-1], xi[idx+1]]
                interval_idx_bound = list(range(IdxIntervalIdxLst[idx-1], IdxIntervalIdxLst[idx+1]+1))
                # note that the ith coord can appear in
                # the interval containing the i-1th to the i+1 coord , but must be greater than the i-1th coord coord
                # and lower than the i+1 coord

        else:  # if isRow is false, IdxIntervalIdxLst will be the column list, no need to change
            if idx == 0:
                bound = [tree.root.columnLB, eta[1]]
                interval_idx_bound = list(range(0, IdxIntervalIdxLst[1]+1))
            elif idx == len(eta)-1:
                bound = [eta[idx-1], tree.root.columnUB]
                interval_idx_bound = list(range(IdxIntervalIdxLst[idx-1], len(intervalLst)))  # TODO: CHECK if IdxIntervalIdxLst[idx]+1 or len(intervalLst)
            else:
                bound = [eta[idx-1], eta[idx+1]]
                interval_idx_bound = list(range(IdxIntervalIdxLst[idx-1], IdxIntervalIdxLst[idx+1]+1))
        # worked out the legal intervals

        #propose a random location
        coord_proposed = random.uniform(bound[0], bound[1])
        # what interval is it in?
        cutLst = sorted(list(tree.getRowCutDic().keys())) if isRow else sorted(list(tree.getColumnCutDic().keys()))
        # if (cutLst[0], cutLst[1]) != intervalLst[1]:
        #     print((cutLst[0], cutLst[1]), intervalLst[1])
        #     exit('Wrong cut list')
        proposed_interval_idx = findIntervalIdx(coord_proposed, cutLst, intervalLst)
        if intervalLst[proposed_interval_idx][0] > coord_proposed or intervalLst[proposed_interval_idx][1] < coord_proposed:
            exit('Wrong prpopsed_interval_idx')
        if proposed_interval_idx not in interval_idx_bound:
            print(isRow, idx, coord_proposed, proposed_interval_idx, interval_idx_bound, IdxIntervalIdxLst, intervalLst)
            exit('Illegal proposed_interval_idx')
        # replace this one interval with idx_bound

        intervalIdx = proposed_interval_idx
        tempDataPoints = {}
        tmpLeafNodeDic = {}
        # the other index, row or col with key=idx, label=data in otherIdxData
        # if isRow true, it means scan all columns at row idx, if is not row, it means scan all rows at this column idx

        # THIS FOR LOOP REALLOCATES THE ITH ROW OR COL SO THE ITH LATENT COORD IS IN THE intervalIdx th INTERVAL
        for otherIdx, datum in enumerate(otherIdxData):
            if datum is not None:
                # if isRow is true, input COLUMN index i, get the index of the COL interval that the ith COL is in
                # similar effect for isRow being false
                otherIntervalIdx = otherIdxOtherIntervalIdxLst[otherIdx]

                # for each data point, if we put the ith row coord in intervalIdx th row interval, then which leaf block
                # contains the latent coord of data[i, otherIdx]?
                if isRow:  # if is row, row interval idx is the idx th of the row interval lists,
                    rowIntervalIdx = intervalIdx  # we put the ith row in the index th row interval
                    columnIntervalIdx = otherIntervalIdx  # the corresponding col interval containing the latent coord
                    # of data[i, otherIdx]
                else:
                    rowIntervalIdx = otherIntervalIdx
                    columnIntervalIdx = intervalIdx

                # the leaf node that contains this small block, i.e. the leaf node that the data[i, otherIdx] falls into
                # if we put the latent coord of the ith row in the intervalIdx th row interval
                leafNode = rowIntervalColumnIntervalLeafNodeDic[rowIntervalIdx][columnIntervalIdx][0]

                # record the leaf nodes affected by adding this data cell back
                tmpLeafNodeDic[leafNode] = None  # value is kind of a place holder, only need this leaf node

                # additional points fall inside this leafNode
                # if not seen before, first copy the existing points in leafNode, then add this new one
                # if have seen before, just add the new one
                if leafNode not in tempDataPoints:
                    # if leafNode is new to tempDataPoints:
                    if leafNode.DataPoints is not None:
                        tempDataPoints[leafNode] = [leafNode.DataPoints[0][:], leafNode.DataPoints[1][:]]
                    else:
                        tempDataPoints[leafNode] = [[], []]

                # now add the new data cell [idx, other idx] or [other idx, idx] to it
                if isRow:
                    if idx not in tempDataPoints[leafNode][0]:
                        tempDataPoints[leafNode][0].append(idx)
                    if otherIdx not in tempDataPoints[leafNode][1]:
                        tempDataPoints[leafNode][1].append(otherIdx)
                else:
                    if idx not in tempDataPoints[leafNode][1]:
                        tempDataPoints[leafNode][1].append(idx)
                    if otherIdx not in tempDataPoints[leafNode][0]:
                        tempDataPoints[leafNode][0].append(otherIdx)

        # record the updated allocation if we put the ith row into this interval
        updated_allocation_store[intervalIdx] = tempDataPoints

        # ONCE WE HAVE REALLOCATED THE ITH ROW, COMPUTE THE UPDATED LKD
        # NOTE THAT WE ONLY NEED TO WORRY ABOUT THE LEAF NODES INSIDE tmpLeafNodeDic,
        # IF we have removed the ith row/col before, then adding this row/col will only affect leaf nodes
        # in tmpLeafNodeDic, EVERYTHING ELSE REMAINS SAME
        # IMAGINE dividing the updated joint lkd by the joint lkd WITHOUT THE idx row/col, only need to worry about
        # the ratio of lkd with vs without the idx row for leaf nodes in tmpLeafNodeDic

        # TODO: compare with block level marginal lkd method
        interval_specific_updated_lkd = {}  # for this choice of row interval, record updated lkd for each block
        for leafNode in tmpLeafNodeDic:  # tmpLeafNodeDic： all leaf nodes affected by adding the row/col
            temp_row, temp_col = cov_idx_to_data_idx(tempDataPoints[leafNode][0], tempDataPoints[leafNode][1],
                                                     row_link, col_link)
            candidate_data = data[temp_row][:, temp_col, :].reshape(-1, 2)
            # recall tempDataPoints[leafNode][0], tempDataPoints[leafNode][1] contains all the row/col indices of the
            # updated partition i.e. after we have inserted the ith row
            interval_specific_updated_lkd[leafNode] = marginal_lkd_MC(candidate_data, sigma)  # will add up later

        # also store the updated marginal lkd for each block at this interval, will use for updating the blocks
        updated_lkd_store[intervalIdx] = interval_specific_updated_lkd

        proposed_log_lkd = 0
        for l in tree.leafBlockDic.keys():
            if l in interval_specific_updated_lkd:
                proposed_log_lkd += interval_specific_updated_lkd[l]
            else:
                if l.marginal_lkd is not None:
                    proposed_log_lkd += l.marginal_lkd  # if being updated, add the new value, otherwise add the old one

        log_MH_ratio = gamma*(proposed_log_lkd - old_lkd)
        # print(log_MH_ratio)
        if np.log(random.random()) < log_MH_ratio:
            chosen_idx = intervalIdx  # just return the interval idx is fine, as density is const within it
            return chosen_idx, updated_lkd_store[chosen_idx], updated_allocation_store[chosen_idx], \
                   [max(bound[0], intervalLst[chosen_idx][0]), min(bound[1], intervalLst[chosen_idx][1])], coord_proposed
        else:
            return None, None, None, None, None  # just a bunch of NANs if rejected

    # TODO: the two chunks works pretty much the same, merge them for more concise code, leave it for now
    else:  # this is the unconstrained ordering case
        #propose a random location
        coord_proposed = random.uniform(intervalLst[0][0], intervalLst[-1][1])  # should be the two ends of the box
        if isRow and [intervalLst[0][0], intervalLst[-1][1]] != [tree.root.rowLB, tree.root.rowUB]:
            print([intervalLst[0][0], intervalLst[-1][1]], [tree.root.rowLB, tree.root.rowUB])
            exit('wrong unconstrained row bound')
        if not isRow and [intervalLst[0][0], intervalLst[-1][1]] != [tree.root.columnLB, tree.root.columnUB]:
            print([intervalLst[0][0], intervalLst[-1][1]], [tree.root.columnLB, tree.root.columnUB])
            exit('wrong unconstrained row bound')

        # what interval is it in?
        cutLst = sorted(list(tree.getRowCutDic().keys())) if isRow else sorted(list(tree.getColumnCutDic().keys()))
        # if (cutLst[0], cutLst[1]) != intervalLst[1]:
        #     print((cutLst[0], cutLst[1]), intervalLst[1])
        #     exit('Wrong cut list')
        proposed_interval_idx = findIntervalIdx(coord_proposed, cutLst, intervalLst)
        if intervalLst[proposed_interval_idx][0] > coord_proposed or intervalLst[proposed_interval_idx][1] < coord_proposed:
            exit('Wrong prpopsed_interval_idx')

        # give a interval list (either col or row), if we put the row or column in this interval...
        intervalIdx = proposed_interval_idx
        tempDataPoints = {}
        tmpLeafNodeDic = {}
        # the other index, row or col with key=idx, label=data in otherIdxDataDic
        # if isRow true, it means scan all columns at row idx, if is not row, it means scan all rows at this column idx

        # THIS FOR LOOP REALLOCATES THE ITH ROW OR COL SO THE ITH LATENT COORD IS IN THE intervalIdx th INTERVAL
        for otherIdx, datum in enumerate(otherIdxData):
            if datum is not None:
                # if isRow is true, input COLUMN index i, get the index of the COL interval that the ith COL is in
                # similar effect for isRow being false
                otherIntervalIdx = otherIdxOtherIntervalIdxLst[otherIdx]

                # for each data point, if we put the ith row coord in intervalIdx th row interval, then which leaf block
                # contains the latent coord of data[i, otherIdx]?
                if isRow:  # if is row, row interval idx is the idx th of the row interval lists,
                    rowIntervalIdx = intervalIdx  # we put the ith row in the index th row interval
                    columnIntervalIdx = otherIntervalIdx  # the corresponding col interval containing the latent coord
                    # of data[i, otherIdx]
                else:
                    rowIntervalIdx = otherIntervalIdx
                    columnIntervalIdx = intervalIdx

                # the leaf node that contains this small block, i.e. the leaf node that the data[i, otherIdx] falls into
                # if we put the latent coord of the ith row in the intervalIdx th row interval
                leafNode = rowIntervalColumnIntervalLeafNodeDic[rowIntervalIdx][columnIntervalIdx][0]

                # record the leaf nodes affected by adding this data cell back
                if leafNode not in tmpLeafNodeDic:
                    tmpLeafNodeDic[leafNode] = None  # kind of a place holder

                # additional points fall inside this leafNode
                # if not seen before, first copy the existing points in leafNode, then add this new one
                # if have seen before, just add the new one
                if leafNode not in tempDataPoints:
                    # if leafNode is new to tempDataPoints:
                    if leafNode.DataPoints is not None:
                        tempDataPoints[leafNode] = [leafNode.DataPoints[0][:], leafNode.DataPoints[1][:]]
                    else:
                        tempDataPoints[leafNode] = [[],[]]

                # now add the new data cell [idx, other idx] or [other idx, idx] to it
                if isRow:
                    if idx not in tempDataPoints[leafNode][0]:
                        tempDataPoints[leafNode][0].append(idx)
                    if otherIdx not in tempDataPoints[leafNode][1]:
                        tempDataPoints[leafNode][1].append(otherIdx)
                else:
                    if idx not in tempDataPoints[leafNode][1]:
                        tempDataPoints[leafNode][1].append(idx)
                    if otherIdx not in tempDataPoints[leafNode][0]:
                        tempDataPoints[leafNode][0].append(otherIdx)

        # record the updated allocation if we put the ith row into this interval
        updated_allocation_store[intervalIdx] = tempDataPoints

        interval_specific_updated_lkd = {}  # for this choice of row interval, record updated lkd for each block
        for leafNode in tmpLeafNodeDic:  # tmpLeafNodeDic： all leaf nodes affected by adding the row/col
            temp_row, temp_col = cov_idx_to_data_idx(tempDataPoints[leafNode][0], tempDataPoints[leafNode][1],
                                                     row_link, col_link)
            candidate_data = data[temp_row][:, temp_col, :].reshape(-1, 2)
            # recall tempDataPoints[leafNode][0], tempDataPoints[leafNode][1] contains all the row/col indices of the
            # updated partition i.e. after we have inserted the ith row
            interval_specific_updated_lkd[leafNode] = marginal_lkd_MC(candidate_data, sigma)

        # note that key is the interval index, value is the log lkd component
        # for this row/col interval choice, finally get the log lkd up additive constant!
        # also store the updated marginal lkd for each block at this interval, will use for updating the blocks
        updated_lkd_store[intervalIdx] = interval_specific_updated_lkd

        proposed_log_lkd = 0
        for l in tree.leafBlockDic.keys():
            if l in interval_specific_updated_lkd:
                proposed_log_lkd += interval_specific_updated_lkd[l]
            else:
                if l.marginal_lkd is not None:
                    proposed_log_lkd += l.marginal_lkd  # if being updated, add the new value, otherwise add the old one

        log_MH_ratio = gamma*(proposed_log_lkd - old_lkd)  # since the iteration only run once
        # print(log_MH_ratio)
        if np.log(random.random()) < log_MH_ratio:
            chosen_idx = intervalIdx  # just return the interval idx is fine, as density is const within it
            return chosen_idx, updated_lkd_store[chosen_idx], updated_allocation_store[chosen_idx], \
                   intervalLst[chosen_idx], coord_proposed
        else:
            return None, None, None, None, None  # just a bunch of NANs if rejected



# full MH sampler for all or a randomly selected subset row and col coordinates
def MH_for_coord(sigma, data, xi, eta, tree, row_link, col_link, maxMHIteration = 1, isPreMHLikelihood=True,
                 fixed_ordering_col=False, fixed_ordering_row=False, p_row=0.2, p_col=0.2, gamma=1.):
    """
    full MH sampler for updating row and col coordinates with topology of tree being fixed
    :param sigma: std of prior on Gaussian noises
    :param data: data matrix, at least data[i][j] or data[i,j] returns the data in (i,j) cell
    :param xi: all latent row coordinates
    :param eta: all latent col coordinates
    :param tree: the tree object
    :param row_link, col_link:
    :param maxMHIteration: max number of Gibbs iterations
    :param isPreMHLikelihood: do we calculate the lkd given parameters before running the MH sampler? or after?
    :param fixed_ordering_col: do we fix the ordering of the columns?
    :param fixed_ordering_row: do we fix the ordering of the rows?
    :param p_row, p_col: the proportion of entries to be updated, e.g. 0.2 means 20% of randomly selected entries
    :param gamma: tempering parameter
    :return: updated xi, eta, and log likelihood before update if isPre = True, else the log lkd after the Gibbs updates
    """

    # given the tree, i.e. the Mondrian tree, record all its leaf blocks
    leafBlockLst = list(tree.getLeafBlockDic().keys())
    # record all row cut positions, recall that key=cut position, value=node of the cut
    rowCutLst = list(tree.getRowCutDic().keys())
    # similar, all col cut positions
    columnCutLst = list(tree.getColumnCutDic().keys())
    # print("# of row cuts: {}, # of column cuts: {}".format(len(rowCutLst), len(columnCutLst)))
    # sort the row and col cuts
    rowCutLst = sorted(list(rowCutLst))
    columnCutLst = sorted(list(columnCutLst))
    # initialize the row interval list
    rowIntervalLst = [(tree.root.rowLB, tree.root.rowUB)]
    # get all row intervals produced by row cuts, these cuts are much finer than the actual tree
    if len(rowCutLst) > 0:
        rowIntervalLst = [(tree.root.rowLB, rowCutLst[0])] + [(rowCutLst[idx], rowCutLst[idx+1])
                                                              for idx in range(0, len(rowCutLst)-1)] + [(rowCutLst[-1], tree.root.rowUB)]
    # initialize the col interval list
    columnIntervalLst = [(tree.root.columnLB, tree.root.columnUB)]
    # similar, get all col intervals produced by col cuts
    if len(columnCutLst) > 0:
        columnIntervalLst = [(tree.root.columnLB, columnCutLst[0])] + [(columnCutLst[idx], columnCutLst[idx+1])
                                                                       for idx in range(0, len(columnCutLst)-1)] + [(columnCutLst[-1], tree.root.columnUB)]

    # given idx of a row/col interval, find all leaf that overlaps with the horizontal row interval*[0,1] or
    # the vertical [0,1]*that col interval
    rowIntervalLeafNodeDic = {}
    columnIntervalLeafNodeDic = {}
    # scan over each leaf
    for i, leafNode in enumerate(leafBlockLst):
        # for each block, work out the row intervals it overlaps with
        # find the index of the row interval that the starting and ending points of the block are in
        # recall findIntervalIdx returns the interval index i such that interval[i][0]<= x < interval[i][1]
        rowLBIdx = findIntervalIdx(leafNode.rowLB, rowCutLst, rowIntervalLst)
        rowUBIdx = findIntervalIdx(leafNode.rowUB, rowCutLst, rowIntervalLst)

        # for each of the covered row interval
        for rowIntervalIdx in range(rowLBIdx, rowUBIdx+1):
            (start, end) = rowIntervalLst[rowIntervalIdx]
            # if the leaf node do cover this interval, register this leaf node to the index of this interval
            if leafNode.rowLB <= start and leafNode.rowUB >= end:
                if rowIntervalIdx in rowIntervalLeafNodeDic.keys():
                    rowIntervalLeafNodeDic[rowIntervalIdx][leafNode] = 0  # 0 is just a place holder
                else:
                    rowIntervalLeafNodeDic[rowIntervalIdx] = {leafNode: 0}

        # similar thing, for column intervals
        columnLBIdx = findIntervalIdx(leafNode.columnLB, columnCutLst, columnIntervalLst)
        columnUBIdx = findIntervalIdx(leafNode.columnUB, columnCutLst, columnIntervalLst)
        for columnIntervalIdx in range(columnLBIdx, columnUBIdx+1):
            (start, end) = columnIntervalLst[columnIntervalIdx]
            if leafNode.columnLB <= start and leafNode.columnUB >= end:
                if columnIntervalIdx in columnIntervalLeafNodeDic.keys():
                    columnIntervalLeafNodeDic[columnIntervalIdx][leafNode] = 0
                else:
                    columnIntervalLeafNodeDic[columnIntervalIdx] = {leafNode: 0}

    # given row and col interval idx i,j, return the node that contains this little block AS A LENGTH 1 LIST smh
    # i.e. Dic[i][j][0] will return the node in which block does the little square row interval i * col interval j live in
    rowIntervalColumnIntervalLeafNodeDic = {}
    for rowIntervalIdx in range(0, len(rowIntervalLst)):
        rowIntervalColumnIntervalLeafNodeDic[rowIntervalIdx] = {}
        for columnIntervalIdx in range(0, len(columnIntervalLst)):
            tmpLst = []
            for leafNode in rowIntervalLeafNodeDic[rowIntervalIdx].keys():
                if leafNode in columnIntervalLeafNodeDic[columnIntervalIdx].keys():
                    tmpLst.append(leafNode)
            if len(tmpLst) > 1:
                for _ in tmpLst:
                    print(_)
                exit('There should be only one leaf block!')
            rowIntervalColumnIntervalLeafNodeDic[rowIntervalIdx][columnIntervalIdx] = tmpLst

    # initialize a bunch of quantities
    # recall that countDic is a dict with key=leaf node, value=dict of data with key=label, value=number of appearance
    # sumDic is a dict with key=leaf Node, value=number of data fall inside this node
    # datumAssignedLeafNodeDic: dict[i][j] gives the leaf node cell i,j resides in as a length 1 list
    # rowIdxRowIntervalIdxLst: given index of latent row coord, return the index of row interval that coord is in
    # columnIdxColumnIntervalIdxLst: given index of latent col coord, return the index of col interval that coord is in

    [LeafNodetoData, datumAssignedLeafNodeDic, rowIdxRowIntervalIdxLst,
     columnIdxColumnIntervalIdxLst] = initialLeafNodeStats(leafBlockLst, data, xi, eta,
                                                           rowIntervalLeafNodeDic, columnIntervalLeafNodeDic,
                                                           rowCutLst, rowIntervalLst, columnCutLst, columnIntervalLst)

    logLikelihood = 0
    # do you want to compute the marginal log lkd of the current configuration with given sigma before the gibbs update?
    if isPreMHLikelihood:
        preMHLogLikelihood = 0
        for leafNode in leafBlockLst:
            if leafNode.marginal_lkd is not None:   # none or 0.0 if there is no data point in the block
                # recall that we store the random estimate of marginal lkd in each block
                preMHLogLikelihood += leafNode.marginal_lkd
        logLikelihood = preMHLogLikelihood

    # actual gibbs updates
    for itr in range(0, maxMHIteration):
        # updates for rows
        # for rowIdx, rowPos in enumerate(xi):  # NOTE: if we want to update every row
        # selected_rows = sorted(np.random.choice(list(range(len(xi))), size=int(p_row*len(xi)), replace=False))
        selected_rows = sorted(random.sample(list(range(len(xi))), k=min(len(xi), int(p_row*len(xi))+1)))
        for rowIdx in selected_rows:  # NOTE: only updating a randomly selected p_row portion of the rows
            # first record the current allocation and current lkd
            current_allocation = {l: copy.deepcopy(l.DataPoints) if l.DataPoints is not None else None for l in tree.leafBlockDic.keys()}
            current_lkd_dict = {l: l.marginal_lkd*1.0 if l.marginal_lkd is not None else 0.0 for l in tree.leafBlockDic.keys()}
            current_LogLikelihood = sum(current_lkd_dict.values())

            # for each row coordinate, first sample a sample new coord uniformly from the legal interval,
            # then update block information, similar for cols
            # remove the ith row from the Mondrian tree records, update the marginal log lkd of the block without it
            leaf_affected = {}  # record the leaf nodes affected by removing the ith row
            for columnIdx, columnPos in enumerate(eta):
                leafNodes = datumAssignedLeafNodeDic[rowIdx][columnIdx]
                # print(leafNodes[0].DataPoints, rowIdx, xi[rowIdx], columnIdx, eta[columnIdx])
                if len(leafNodes) > 1:
                    exit('One datum can only belong to one leaf block!')
                else:
                    leafNode = leafNodes[0]
                    # if it is a new node we have not seen before
                    if leafNode not in leaf_affected:
                        leaf_affected[leafNode] = None
                        # recall leafNode.DataPoints should be None if no data is in the block, otherwise a list
                        leafNode.DataPoints[0].remove(rowIdx)
                        # if removing rowIdx leads to no row idx in .DataPoints, change it to None
                        if leafNode.DataPoints[0] == []:
                            leafNode.DataPoints = None
                    # if it is already in the affected blocks, double check
                    else:
                        # .DataPoints can only be None or must not contain rowIdx
                        if leafNode.DataPoints is not None and rowIdx in leafNode.DataPoints[0]:
                            exit('something wrong with the removing row/col from tree process')

            # update the marginal lkd for the affected blocks. These are now lkd WITHOUT the ith row
            # i.e. all blocks now contain corresponding marginal lkd without the ith row
            for leafNode in leaf_affected.keys():
                if leafNode.DataPoints is None:
                    leafNode.marginal_lkd = 0.0
                else:
                    temp_row, temp_col = cov_idx_to_data_idx(leafNode.DataPoints[0], leafNode.DataPoints[1],
                                                             row_link, col_link)
                    candidate_data = data[temp_row][:, temp_col, :].reshape(-1, 2)
                    # recall leafNode.DataPoints contains all the updated row/col indices for affected leaf nodes
                    leafNode.marginal_lkd = marginal_lkd_MC(candidate_data, sigma)

            # once we have removed the rth row and updated the lkd, perform one MH step, recall it returns 5 things,
            # chosen idx of interval, dictionary of marginal lkd of the affected blocks, updated data allocation
            # and the chosen interval and the chosen position.
            # when fixed_ordering=False, the chosen interval and the idxth interval is the same
            # then fixed_ordering=True, it is truncated according to the i-1 and i+1th coord

            # TEMPERING MODIFICATION 3: NOW ROW COORD ARE UPDATED ACCORDING TO THE TEMPERED DISTRIBUTION

            newRowIntervalIdx, updated_marginal, \
            updated_allocation, bound, proposed_coord = MHOneDimensionStep(tree, sigma, data, rowIdx, xi, eta,
                                                                           current_LogLikelihood,
                                                                           rowIntervalLst, rowIdxRowIntervalIdxLst,
                                                                           columnIdxColumnIntervalIdxLst,
                                                                           rowIntervalColumnIntervalLeafNodeDic,
                                                                           True, row_link, col_link, fixed_ordering_row,
                                                                           gamma)

            if proposed_coord is None:  # if update is rejected, just go back to before, everything else is unchanged
                # print('row rejected')
                for leafNode in leaf_affected.keys():
                    # print('current data points {} lkd {}'.format(leafNode.DataPoints, leafNode.marginal_lkd))
                    leafNode.DataPoints = current_allocation[leafNode]
                    leafNode.marginal_lkd = current_lkd_dict[leafNode]
                    # print('recovered data points {} lkd {}'.format(leafNode.DataPoints, leafNode.marginal_lkd))

            if proposed_coord is not None:  # if update is accepted
                # print('new row {} coord {}'.format(rowIdx, proposed_coord))
                if not fixed_ordering_row:
                    # when fixed_ordering = false, bound should be the newIdx th interval
                    if rowIntervalLst[newRowIntervalIdx] != bound:
                        exit('error in Gibbs update for row coord')
                else:
                    # when fixed_ordering = true, bound should be the newIdx th interval truncated at i-1 and i+1 th coord
                    if rowIntervalLst[newRowIntervalIdx][0] > bound[0] or rowIntervalLst[newRowIntervalIdx][1] < bound[1]:
                        exit('error in Gibbs update for row coord')

                # assign the rth row coord to a new row interval
                rowIdxRowIntervalIdxLst[rowIdx] = newRowIntervalIdx
                # update the rth row coordinate
                xi[rowIdx] = proposed_coord

                # update the data<->tree assignment after making the update, note that we only have to worry about this row
                for leafNode, points in updated_allocation.items():
                    leafNode.DataPoints = points  # update data allocation
                    if leafNode not in list(updated_marginal.keys()):
                        exit("Sth wrong when recording the updated marginal lkds")
                    leafNode.marginal_lkd = updated_marginal[leafNode]  # update marginal lkd of the affected blocks

                for columnIdx, columnPos in enumerate(eta):
                    columnIntervalIdx = columnIdxColumnIntervalIdxLst[columnIdx]
                    # now the i,j cell is in a different node! the ith row coord now link to interval newRowIntervalIdx
                    datumAssignedLeafNodeDic[rowIdx][columnIdx] = rowIntervalColumnIntervalLeafNodeDic[newRowIntervalIdx][columnIntervalIdx]

        # similar MH updates for each column coordinate
        # for columnIdx, columnPos in enumerate(eta):
        # selected_cols = sorted(np.random.choice(list(range(len(eta))), size=int(p_col*len(eta)), replace=False))
        selected_cols = sorted(random.sample(list(range(len(eta))), k=min(len(eta), int(p_col*len(eta))+1)))
        for columnIdx in selected_cols:
            # first record the current allocation
            current_allocation = {l: copy.deepcopy(l.DataPoints) if l.DataPoints is not None else None for l in tree.leafBlockDic.keys()}
            current_lkd_dict = {l: l.marginal_lkd*1.0 if l.marginal_lkd is not None else 0.0 for l in tree.leafBlockDic.keys()}
            current_LogLikelihood = sum(current_lkd_dict.values())

            # remove jth column from the data
            leaf_affected = {}
            for rowIdx, rowPos in enumerate(xi):
                leafNodes = datumAssignedLeafNodeDic[rowIdx][columnIdx]
                # print(leafNodes[0].DataPoints, rowIdx, xi[rowIdx], columnIdx, eta[columnIdx])
                if len(leafNodes) > 1:
                    exit('One datum can only belong to one leaf block!')
                else:
                    # here leafNode.DataPoints should always be non-None since rowIdx,colIdx is in it,
                    # but may become empty if we remove it
                    leafNode = leafNodes[0]
                    # if it is a new node we have not seen before
                    if leafNode not in leaf_affected:
                        leaf_affected[leafNode] = None
                        # recall leafNode.DataPoints should be None if no data is in the block, otherwise a list of list
                        leafNode.DataPoints[1].remove(columnIdx)
                        # if removing colIdx leads to no col idx in .DataPoints, change it to None
                        if leafNode.DataPoints[1] == []:
                            leafNode.DataPoints = None
                    # if it is already in the affected blocks, double check
                    else:
                        # .DataPoints can only be None or must not contain rowIdx
                        if leafNode.DataPoints is not None and columnIdx in leafNode.DataPoints[1]:
                            exit('something wrong with the removing row/col from tree process')

            # update the marginal lkd for the affected blocks. These are now lkd WITHOUT the jth col
            # i.e. every block contains corresponding marginal lkd with the data set without the jth column
            # TODO: compare it to block level method?
            for leafNode in leaf_affected.keys():
                if leafNode.DataPoints is None:
                    leafNode.marginal_lkd = 0.0
                else:
                    temp_row, temp_col = cov_idx_to_data_idx(leafNode.DataPoints[0], leafNode.DataPoints[1],
                                                             row_link, col_link)
                    candidate_data = data[temp_row][:, temp_col, :].reshape(-1, 2)
                    # recall leafNode.DataPoints contains all the updated row/col indices for affected leaf nodes
                    leafNode.marginal_lkd = marginal_lkd_MC(candidate_data, sigma)

            # TEMPERING MODIFICATION 4: NOW COLUMN COORD ARE UPDATED ACCORDING TO THE TEMPERED DISTRIBUTION
            # do one gibbs update for col coordinate
            newColumnIntervalIdx, updated_marginal, \
            updated_allocation, bound, proposed_coord = MHOneDimensionStep(tree, sigma, data, columnIdx, xi, eta,
                                                                           current_LogLikelihood,
                                                                           columnIntervalLst,
                                                                           columnIdxColumnIntervalIdxLst,
                                                                           rowIdxRowIntervalIdxLst,
                                                                           rowIntervalColumnIntervalLeafNodeDic,
                                                                           False, row_link, col_link,
                                                                           fixed_ordering_col, gamma)

            if proposed_coord is None:  # if rejected, put everything back
                # print('rejected')
                for leafNode in leaf_affected.keys():
                    # print('current data points {} lkd {}'.format(leafNode.DataPoints, leafNode.marginal_lkd))
                    leafNode.DataPoints = current_allocation[leafNode]
                    leafNode.marginal_lkd = current_lkd_dict[leafNode]
                    # print('recovered data points {} lkd {}'.format(leafNode.DataPoints, leafNode.marginal_lkd))

            if proposed_coord is not None:
                # print('new col {} coord {}'.format(columnIdx, proposed_coord))
                if not fixed_ordering_col:
                    # when fixed_ordering = false, bound should be the newIdx th interval
                    if columnIntervalLst[newColumnIntervalIdx] != bound:
                        exit('error in Gibbs update for col coord')
                else:
                    # when fixed_ordering = true, bound should be the newIdx th interval truncated at i-1 and i+1 th coord
                    if columnIntervalLst[newColumnIntervalIdx][0] > bound[0] or columnIntervalLst[newColumnIntervalIdx][1] < bound[1]:
                        exit('error in Gibbs update for col coord')

                # assign the jth col coord to the new interval
                columnIdxColumnIntervalIdxLst[columnIdx] = newColumnIntervalIdx
                eta[columnIdx] = proposed_coord

                # update the data<->tree assignment after making the update, note that we only have to worry about this row
                for leafNode, points in updated_allocation.items():
                    leafNode.DataPoints = points  # update the data allocation
                    if leafNode not in list(updated_marginal.keys()):
                        exit("Sth wrong when recording the updated marginal lkds")
                    leafNode.marginal_lkd = updated_marginal[leafNode]  # update the marginal lkd for affected blocks

                for rowIdx, rowPos in enumerate(xi):
                    rowIntervalIdx = rowIdxRowIntervalIdxLst[rowIdx]  # get the row interval for ith row coord
                    datumAssignedLeafNodeDic[rowIdx][columnIdx] = rowIntervalColumnIntervalLeafNodeDic[rowIntervalIdx][newColumnIntervalIdx]

    if not isPreMHLikelihood:
        postMHLogLikelihood = 0
        for leafNode in leafBlockLst:
            if leafNode.marginal_lkd is not None:   # none or 0 if there is no data point in the block
                # recall that we store the random estimate of marginal lkd in each block
                postMHLogLikelihood += leafNode.marginal_lkd
        logLikelihood = postMHLogLikelihood

    return [xi, eta, logLikelihood]


# prior of gamma is Gamma(3,50), kinda arbitrary
# TODO: try random walk on log scale
def sigma_MH_in_Gibbs(data, sigma, tree, row_link, col_link, jump_size=0.8, max_iter=1, gamma=1.):
    # to update sigma while fixing everything else, only need to propose a new state using symmetric move,
    # workout the marginal lkd p(DATA|sigma) and do the usual MH update with
    # acceptance rate = [p(DATA|new sig)p(new sig)]/[p(DATA|old sig)p(old sig)]
    for itr in range(max_iter):
        # proposed_sigma = sigma + jump_size * np.random.randn()  # symmetric proposal
        # or random walk on log scale

        u = random.uniform(jump_size, 1/jump_size)
        proposed_sigma = sigma*u
        if proposed_sigma < 0:
            pass
        else:
            new_lkd_dict = {}
            for l in tree.leafBlockDic.keys():  # work out the marginal lkd under new sigma
                if l.DataPoints is None:
                    new_lkd_dict[l] = 0.0
                else:
                    temp_row, temp_col = cov_idx_to_data_idx(l.DataPoints[0], l.DataPoints[1],
                                                             row_link, col_link)
                    new_lkd_dict[l] = marginal_lkd_MC(data[temp_row][:, temp_col, :].reshape(-1, 2),
                                                      proposed_sigma)
            proposed_lkd = sum(new_lkd_dict.values())
            old_lkd = sum([l.marginal_lkd if l.marginal_lkd is not None else 0.0 for l in tree.leafBlockDic.keys()])
            log_lkd_ratio = proposed_lkd - old_lkd  # work out lkd ratio
            log_prior_ratio = (3-1)*(np.log(proposed_sigma) - np.log(sigma)) - 50*(proposed_sigma - sigma)  # work out prior ratio Gamma(3,50)
            # log_MH_ratio = log_lkd_ratio+log_prior_ratio   # symmetric update
            log_MH_ratio = gamma*(log_lkd_ratio + log_prior_ratio) - np.log(u)
            # if gamma=1 we end up with the target posterior
            if np.log(random.random()) < log_MH_ratio:
                sigma = proposed_sigma  # if accept, update sigma
                for l in tree.leafBlockDic.keys():  # if accept, update marginal lkd at each leaf block
                    l.marginal_lkd = new_lkd_dict[l]
            # if reject, do nothing
    return sigma


# update the cutting cost of a randomly selected cut, recall MP is defined through two latent sequence, cutPos, cutCost
def MH_update_cutting_cost(tree, jump_size=0.8, gamma=1.):
    if len(tree.leafCutDic) == 0:
        return

    old_tree = copy.deepcopy(tree)
    proposed_tree = copy.deepcopy(tree)
    candidate_cuts = {**proposed_tree.columnCutDic, **proposed_tree.rowCutDic}
    candidate_cut_pos = random.choice(list(candidate_cuts.keys()))
    candidate_cut = candidate_cuts[candidate_cut_pos]
    curr_cost = candidate_cut.budget - candidate_cut.leftChild.budget

    # proposed_perturbation = jump_size*np.random.randn()  # symmetric updates
    # proposed_cost = curr_cost+proposed_perturbation

    u = random.uniform(jump_size, 1/jump_size)  # try random walk on log scale
    proposed_cost = u*curr_cost  # proposed_cost = curr cost+proposed_perturbation
    proposed_perturbation = (u-1)*curr_cost

    # update the proposed tree with new cutting cost
    affected_blocks = [candidate_cut.leftChild, candidate_cut.rightChild]
    while len(affected_blocks) != 0:
        next_level = []
        for l in affected_blocks:
            l.budget -= proposed_perturbation  # existing budgets are reduced by proposed_pert
            # note that new_budget = old_budget - u*curr_cost = old_budget-curr_cost+(1-u)*curr_cost
            # = curr_budget-(u-1)*curr_cost
            if l.budget <= 0:
                return  # illegal cutting cost! bad bad
            if not l.isLeaf():
                next_level += [l.leftChild, l.rightChild]
        affected_blocks = next_level

    # TODO: now work out the density ratio between proposed_tree and old_tree
    # recursively removing cuts in tree1, tree2 until the trivial partition
    log_tree_proposed = 0.
    while len(proposed_tree.leafCutDic) != 0:
        leafCut = proposed_tree.getRandomLeafCut()
        # randomly choose a leaf cut, compute the density ratio between the current tree and the tree with this cut
        # merged, repeat this process until going back to trivial partition, get p(tree)/p(trivial)
        if leafCut.cutDir == 0:
            log_tree_proposed += -(leafCut.columnUB - leafCut.columnLB)*leafCut.rightChild.budget
        else:
            log_tree_proposed += -(leafCut.rowUB - leafCut.rowLB)*leafCut.rightChild.budget
        proposed_tree.removeLeafCut(leafCut)
    log_tree_proposed -= 2*proposed_tree.budget  # trivial partition: no cut, i.e. cost > old_budget, cost ~ EXP(2)

    log_tree_old = 0.
    while len(old_tree.leafCutDic) != 0:
        leafCut = old_tree.getRandomLeafCut()
        # randomly choose a leaf cut, compute the density ratio between the current tree and the tree with this cut
        # merged, repeat this process until going back to trivial partition, get p(tree)/p(trivial)
        if leafCut.cutDir == 0:
            log_tree_old += -(leafCut.columnUB - leafCut.columnLB)*leafCut.rightChild.budget
        else:
            log_tree_old += -(leafCut.rowUB - leafCut.rowLB)*leafCut.rightChild.budget
        old_tree.removeLeafCut(leafCut)
    log_tree_old -= 2*old_tree.budget  # trivial partition: no cut, i.e. cost > new_budget, cost ~ EXP(2)

    # log_MH_ratio = log_tree_proposed - log_tree_old  # symmetric update
    log_MH_ratio = gamma*(log_tree_proposed - log_tree_old) - np.log(u)  # random walk on log scale

    if np.log(random.random()) < log_MH_ratio:  # if accepted
        tree_cuts = {**tree.columnCutDic, **tree.rowCutDic}
        tree_cut = tree_cuts[candidate_cut_pos]
        # update the proposed tree with new cutting cost
        affected_blocks = [tree_cut.leftChild, tree_cut.rightChild]
        while len(affected_blocks) != 0:
            next_level = []
            for l in affected_blocks:
                l.budget -= proposed_perturbation  # existing budgets are reduced by proposed_pert
                if l.budget <= 0:
                    exit('Illegal cost update!')  # illegal cutting cost! bad bad
                if not l.isLeaf():
                    next_level += [l.leftChild, l.rightChild]
            affected_blocks = next_level

    # TODO: accept the proposed_perturbation with prob
    # TODO: update budgets in tree


def MP_prior(tree):
    old_budget = tree.budget*1.0
    old_tree = copy.deepcopy(tree)
    half_peri = (old_tree.root.rowUB-old_tree.root.rowLB)+(old_tree.root.columnUB-old_tree.root.columnLB)
    # then compute the density of the sequence that generates the tree
    # recursively removing cuts in old_tree until the trivial partition
    log_tree_old = 0.
    while len(old_tree.leafCutDic) != 0:
        leafCut = old_tree.getRandomLeafCut()
        # randomly choose a leaf cut, compute the density ratio between the current tree and the tree with this cut
        # merged, repeat this process until going back to trivial partition, get p(tree)/p(trivial)
        if leafCut.cutDir == 0:
            log_tree_old += -(leafCut.columnUB - leafCut.columnLB)*leafCut.rightChild.budget
        else:
            log_tree_old += -(leafCut.rowUB - leafCut.rowLB)*leafCut.rightChild.budget
        old_tree.removeLeafCut(leafCut)
    log_tree_old -= half_peri*old_budget  # trivial partition: no cut, i.e. cost > old_budget, cost ~ EXP(0.5peri)
    return log_tree_old


# update the cutting position of a randomly selected cut
# TODO: redo the cut proposal within the block! also add error checking lines!
def MH_update_cutting_position(tree, xi, eta, data, col_link, row_link, sigma, gamma=1.):
    # sliding the cutting positions, proposal is uniform between the two neighbouring cuts
    if len(tree.leafCutDic) == 0:
        return
    # choosing a cut uniformly, find its two neighbours, uniformly propose a new one,
    # then update the boundaries of affected blocks, register the xi,eta points,
    # work out MH ratio = lkd ratio * prior ratio, if accepted, update the current tree
    proposed_tree = copy.deepcopy(tree)

    is_row = False  # are we going to update a row or a column cut?
    if random.random() < len(proposed_tree.rowCutDic)/(len(proposed_tree.rowCutDic)+len(proposed_tree.columnCutDic)):
        is_row = True

    # randomly select a row or column cut, identify the legal range of the uniform proposal
    if is_row:  # if updating a row cut
        # print('update a row cut')
        candidate_cuts = proposed_tree.rowCutDic
        candidate_cut_pos = random.choice(list(candidate_cuts.keys()))  # choose an existing cut position
        candidate_cut_block = candidate_cuts[candidate_cut_pos]  # find the block that contains this cut
        # note that all blocks affected by this cut update ARE CHILDREN NODES OF candidate_cut_block

        # first work out the legal range of the proposal, find all row cuts contained in the block
        row_cuts_in_block = []
        affected_blocks = [candidate_cut_block]  # first traverse all children of candidate_cut_block
        while len(affected_blocks) != 0:
            next_level = []
            for l in affected_blocks:
                if l.cutDir == 0:  # if a row cut, add it to row_cuts_in_block
                    row_cuts_in_block += [l.cutPos]
                if not l.leftChild.isLeaf():  # continue if the children are not leaf nodes
                    next_level += [l.leftChild]
                if not l.rightChild.isLeaf():
                    next_level += [l.rightChild]
            affected_blocks = next_level

        sorted_cuts = np.array(sorted(row_cuts_in_block))
        cut_rank = np.where(sorted_cuts == candidate_cut_pos)[0][0]
        temp_pos = [candidate_cut_block.rowLB] + sorted_cuts.tolist() + [candidate_cut_block.rowUB]
        legal_range = (temp_pos[cut_rank], temp_pos[cut_rank+2])

        # there is at least one cut, so temp_pos has at least
        # length 3, temp_pos[cut_rank], temp_pos[cut_rank+2] should return the cutting positions right next to the
        # candidate cut
        proposed_cut = random.uniform(legal_range[0], legal_range[1])  # uniformly generate a cut position
        if proposed_cut > candidate_cut_block.rowUB or proposed_cut < candidate_cut_block.rowLB:
            print('block UB: {}, block LB: {}, old cut: {}, new cut: {}, legal_range: {}'.format(candidate_cut_block.rowUB,
                                                                                                 candidate_cut_block.rowLB, candidate_cut_block.cutPos,
                                                                                                 proposed_cut, legal_range))
            exit('Illegal cutting position')
        proposed_tree.rowCutDic.pop(candidate_cut_pos)
        proposed_tree.rowCutDic[proposed_cut] = candidate_cut_block

    else:  # if updating a column cut
        # print('update a column cut')
        candidate_cuts = proposed_tree.columnCutDic
        candidate_cut_pos = random.choice(list(candidate_cuts.keys()))  # choose a random cut position
        candidate_cut_block = candidate_cuts[candidate_cut_pos]  # find the block that contains this cut

        # first work out the legal range of the proposal, find all row cuts contained in the block
        col_cuts_in_block = []
        affected_blocks = [candidate_cut_block]  # first traverse all children of candidate_cut_block
        while len(affected_blocks) != 0:
            next_level = []
            for l in affected_blocks:
                if l.cutDir == 1:  # if a column cut, add it to col_cuts_in_block
                    col_cuts_in_block += [l.cutPos]
                if not l.leftChild.isLeaf():  # continue if the children are not leaf nodes
                    next_level += [l.leftChild]
                if not l.rightChild.isLeaf():
                    next_level += [l.rightChild]
            affected_blocks = next_level

        sorted_cuts = np.array(sorted(col_cuts_in_block))
        cut_rank = np.where(sorted_cuts == candidate_cut_pos)[0][0]
        temp_pos = [candidate_cut_block.columnLB] + sorted_cuts.tolist() + [candidate_cut_block.columnUB]
        legal_range = (temp_pos[cut_rank], temp_pos[cut_rank+2])

        proposed_cut = random.uniform(legal_range[0], legal_range[1])  # uniformly generate a cut position
        if proposed_cut > candidate_cut_block.columnUB or proposed_cut < candidate_cut_block.columnLB:
            print('block UB: {}, block LB: {}, old cut: {}, new cut: {}, legal_range: {}'.format(candidate_cut_block.columnUB,
                                                                                                 candidate_cut_block.columnLB, candidate_cut_block.cutPos,
                                                                                                 proposed_cut, legal_range))
            exit('Illegal cutting position')
        proposed_tree.columnCutDic.pop(candidate_cut_pos)
        proposed_tree.columnCutDic[proposed_cut] = candidate_cut_block

    # now we have generated a new cutting position, and have registered it in the row/colCutDic
    # update the boundaries of all blocks affected by this cut
    # then for the rest of the blocks inside cut_block, also update their LB,RB according to the cutting direction
    # redo cut based on the updated row/columnCutDic, start from the chosen block
    # for leafCutDic, leafDic, rowCutDic, colCutDic, update the MP block whose boundaries has changed
    candidate_cut_block.cutPos = proposed_cut  # update the cutting position
    affected_blocks = [candidate_cut_block]  # record all blocks affected by this new cut
    while affected_blocks:  # while it is not empty, terminate if a block is a leaf node
        temp_store = []
        for l in affected_blocks:  # for each l, look at the two children nodes of it, if it is a leaf, skip
            if l.cutDir == 0:  # if a row/horizontal cut
                cutPos = l.cutPos
                is_leaf = False
                is_cut = False
                if l.leftChild in proposed_tree.leafCutDic:
                    # if l.leftChild is a leafCut, first remove it from the Dic, will add
                    # back the updated one later
                    proposed_tree.leafCutDic.pop(l.leftChild)
                    is_cut = True
                if l.leftChild.isLeaf():
                    proposed_tree.leafBlockDic.pop(l.leftChild)
                    is_leaf = True

                l.leftChild.rowLB = l.rowLB
                l.leftChild.rowUB = cutPos
                if l.rowLB > cutPos:
                    print('block row UB: {}, block row LB: {}, cut: {}'.format(l.leftChild.rowUB,l.leftChild.rowLB, l.cotPos))
                    exit('Illegal cutPos proposal!')
                l.leftChild.columnLB = l.columnLB
                l.leftChild.columnUB = l.columnUB

                if is_cut:
                    proposed_tree.leafCutDic[l.leftChild] = True  # put the updated version back
                if is_leaf:
                    proposed_tree.leafBlockDic[l.leftChild] = True

                # repeat for right child
                is_leaf = False
                is_cut = False
                if l.rightChild in proposed_tree.leafCutDic:
                    proposed_tree.leafCutDic.pop(l.rightChild)
                    is_cut = True
                if l.rightChild.isLeaf():
                    proposed_tree.leafBlockDic.pop(l.rightChild)
                    is_leaf = True

                l.rightChild.rowLB = cutPos
                l.rightChild.rowUB = l.rowUB
                if cutPos > l.rowUB:
                    print('block row UB: {}, block row LB: {}, cut: {}'.format(l.rightChild.rowUB,l.rightChild.rowLB, l.cotPos))
                    exit('Illegal cutPos proposal!')
                l.rightChild.columnLB = l.columnLB
                l.rightChild.columnUB = l.columnUB

                if is_cut:
                    proposed_tree.leafCutDic[l.rightChild] = True  # put the updated version back
                if is_leaf:
                    proposed_tree.leafBlockDic[l.rightChild] = True

                # finally update the block entry in rowCutDic at this cutting position
                proposed_tree.rowCutDic[cutPos] = l

            else:
                cutPos = l.cutPos

                is_leaf = False
                is_cut = False
                if l.leftChild in proposed_tree.leafCutDic:
                    # if l.leftChild is a leafCut, first remove it from the Dic, will add
                    # back the updated one later
                    proposed_tree.leafCutDic.pop(l.leftChild)
                    is_cut = True
                if l.leftChild.isLeaf():
                    proposed_tree.leafBlockDic.pop(l.leftChild)
                    is_leaf = True

                l.leftChild.rowLB = l.rowLB
                l.leftChild.rowUB = l.rowUB
                l.leftChild.columnLB = l.columnLB
                l.leftChild.columnUB = cutPos
                if l.columnLB > cutPos:
                    print('block col UB: {}, block col LB: {}, cut: {}'.format(l.leftChild.columnUB,l.leftChild.columnLB, l.cutPos))
                    exit('Illegal cutPos proposal!')

                if is_cut:
                    proposed_tree.leafCutDic[l.leftChild] = True  # put the updated version back
                if is_leaf:
                    proposed_tree.leafBlockDic[l.leftChild] = True

                # repeat for right child
                is_leaf = False
                is_cut = False
                if l.rightChild in proposed_tree.leafCutDic:
                    proposed_tree.leafCutDic.pop(l.rightChild)
                    is_cut = True
                if l.rightChild.isLeaf():
                    proposed_tree.leafBlockDic.pop(l.rightChild)
                    is_leaf = True

                l.rightChild.rowLB = l.rowLB
                l.rightChild.rowUB = l.rowUB
                l.rightChild.columnLB = cutPos
                l.rightChild.columnUB = l.columnUB
                if cutPos > l.columnUB:
                    print('block col UB: {}, block col LB: {}, cut: {}'.format(l.rightChild.columnUB,l.rightChild.columnLB, l.cutPos))
                    exit('Illegal cutPos proposal!')

                if is_cut:
                    proposed_tree.leafCutDic[l.rightChild] = True  # put the updated version back
                if is_leaf:
                    proposed_tree.leafBlockDic[l.rightChild] = True

                # finally update the block entry in rowCutDic at this cutting position
                proposed_tree.columnCutDic[cutPos] = l

            if not l.leftChild.isLeaf():
                temp_store += [l.leftChild]
            if not l.rightChild.isLeaf():
                temp_store += [l.rightChild]
        affected_blocks = temp_store

    # if not good_tree_check(proposed_tree):
    #     exit('sth wrong in proposing the new tree')

    # for the proposed tree, we have updated the cut position
    # now put data points back
    proposed_tree.cell_allocate(xi, eta)
    for l in proposed_tree.leafBlockDic.keys():
        l.marginal_lkd_block_level(data=data, xi=xi, eta=eta, p=2, marginal_lkd_func=marginal_lkd_MC,
                                   row_link=row_link, col_link=col_link, test=False, sigma=sigma)
    # likelihood ratio
    proposed_lkd = sum([l.marginal_lkd if l.marginal_lkd is not None else 0.0 for l in proposed_tree.leafBlockDic.keys()])
    old_lkd = sum([l.marginal_lkd if l.marginal_lkd is not None else 0.0 for l in tree.leafBlockDic.keys()])
    # prior ratio
    prior_ratio = MP_prior(proposed_tree) - MP_prior(tree)
    # MH rate
    MH_ratio = gamma*(prior_ratio + proposed_lkd - old_lkd)
    # update cut structure and latent point allocation on the original tree
    if np.log(random.random()) < MH_ratio:  # if proposed cut got accepted
        # update the cut candidate_cut_pos in cutDic, update block boundaries
        # TODO: for leafCutDic, leafDic, rowCutDic, colCutDic, update the MP block whose boundaries has changed
        if is_row:
            # print('row cut update accepted')
            updated_block = tree.rowCutDic[candidate_cut_pos]  # update the cutPos inside the corresponding block
            updated_block.cutPos = proposed_cut
            tree.rowCutDic.pop(candidate_cut_pos)  # update the cutDic
            tree.rowCutDic[proposed_cut] = updated_block
        else:
            # print('column cut update accepted')
            updated_block = tree.columnCutDic[candidate_cut_pos]  # update the cutPos inside the corresponding block
            updated_block.cutPos = proposed_cut
            tree.columnCutDic.pop(candidate_cut_pos)  # update the cutDic
            tree.columnCutDic[proposed_cut] = updated_block

        affected_blocks = [updated_block]  # record all blocks affected by sliding
        while affected_blocks:  # while it is not empty, terminate if a block is a leaf node
            temp_store = []
            for l in affected_blocks:  # for each l, look at the two children nodes of it, if it is a leaf, skip
                if l.cutDir == 0:  # if a row/horizontal cut
                    cutPos = l.cutPos
                    is_leaf = False
                    is_cut = False
                    if l.leftChild in tree.leafCutDic:
                        # if l.leftChild is a leafCut, first remove it from the Dic, will add
                        # back the updated one later
                        tree.leafCutDic.pop(l.leftChild)
                        is_cut = True
                    if l.leftChild.isLeaf():
                        tree.leafBlockDic.pop(l.leftChild)
                        is_leaf = True

                    l.leftChild.rowLB = l.rowLB
                    l.leftChild.rowUB = cutPos
                    if l.rowLB > cutPos:
                        exit('Illegal cutPos proposal!')
                    l.leftChild.columnLB = l.columnLB
                    l.leftChild.columnUB = l.columnUB

                    if is_cut:
                        tree.leafCutDic[l.leftChild] = True  # put the updated version back
                    if is_leaf:
                        tree.leafBlockDic[l.leftChild] = True

                    # repeat for right child
                    is_leaf = False
                    is_cut = False
                    if l.rightChild in tree.leafCutDic:
                        tree.leafCutDic.pop(l.rightChild)
                        is_cut = True
                    if l.rightChild.isLeaf():
                        tree.leafBlockDic.pop(l.rightChild)
                        is_leaf = True

                    l.rightChild.rowLB = cutPos
                    l.rightChild.rowUB = l.rowUB
                    if cutPos > l.rowUB:
                        exit('Illegal cutPos proposal!')
                    l.rightChild.columnLB = l.columnLB
                    l.rightChild.columnUB = l.columnUB

                    if is_cut:
                        tree.leafCutDic[l.rightChild] = True  # put the updated version back
                    if is_leaf:
                        tree.leafBlockDic[l.rightChild] = True

                    # finally update the block entry in rowCutDic at this cutting position
                    tree.rowCutDic[cutPos] = l

                else:
                    cutPos = l.cutPos

                    is_leaf = False
                    is_cut = False
                    if l.leftChild in tree.leafCutDic:
                        # if l.leftChild is a leafCut, first remove it from the Dic, will add
                        # back the updated one later
                        tree.leafCutDic.pop(l.leftChild)
                        is_cut = True
                    if l.leftChild.isLeaf():
                        tree.leafBlockDic.pop(l.leftChild)
                        is_leaf = True

                    l.leftChild.rowLB = l.rowLB
                    l.leftChild.rowUB = l.rowUB
                    l.leftChild.columnLB = l.columnLB
                    l.leftChild.columnUB = cutPos
                    if l.columnLB > cutPos:
                        exit('Illegal cutPos proposal!')


                    if is_cut:
                        tree.leafCutDic[l.leftChild] = True  # put the updated version back
                    if is_leaf:
                        tree.leafBlockDic[l.leftChild] = True

                    # repeat for right child
                    is_leaf = False
                    is_cut = False
                    if l.rightChild in tree.leafCutDic:
                        tree.leafCutDic.pop(l.rightChild)
                        is_cut = True
                    if l.rightChild.isLeaf():
                        tree.leafBlockDic.pop(l.rightChild)
                        is_leaf = True

                    l.rightChild.rowLB = l.rowLB
                    l.rightChild.rowUB = l.rowUB
                    l.rightChild.columnLB = cutPos
                    l.rightChild.columnUB = l.columnUB
                    if cutPos > l.columnUB:
                        exit('Illegal cutPos proposal!')

                    if is_cut:
                        tree.leafCutDic[l.rightChild] = True  # put the updated version back
                    if is_leaf:
                        tree.leafBlockDic[l.rightChild] = True

                    # finally update the block entry in rowCutDic at this cutting position
                    tree.columnCutDic[cutPos] = l

                if not l.leftChild.isLeaf():
                    temp_store += [l.leftChild]
                if not l.rightChild.isLeaf():
                    temp_store += [l.rightChild]
            affected_blocks = temp_store
        # if not good_tree_check(tree):
        #     exit('sth wrong in proposing the new tree')

        # now we have updated the MP pattern, now introduce the data points back
        # can we just copy marginal lkd/ points allocation from the proposed_tree? e.g. tree.leafDic? do they
        # have the same order?  if so, update the registered coordinates and the corresponding marginal lkd
        proposed_leaf_blocks = list(proposed_tree.leafBlockDic.keys())
        original_leaf_blocks = list(tree.leafBlockDic.keys())
        if len(proposed_leaf_blocks) != len(original_leaf_blocks):
            exit('copied tree went wrong')
        for _ in range(len(proposed_leaf_blocks)):
            # check if we are talking about the same block
            if any([original_leaf_blocks[_].rowLB != proposed_leaf_blocks[_].rowLB,
                    original_leaf_blocks[_].rowUB != proposed_leaf_blocks[_].rowUB,
                    original_leaf_blocks[_].columnLB != proposed_leaf_blocks[_].columnLB,
                    original_leaf_blocks[_].columnUB != proposed_leaf_blocks[_].columnUB]):
                exit('sth went wrong when updating cutting positions')

            original_leaf_blocks[_].marginal_lkd = proposed_leaf_blocks[_].marginal_lkd
            original_leaf_blocks[_].DataPoints = proposed_leaf_blocks[_].DataPoints




# TODO: try random walk on log scale
def budget_MH_in_Gibbs(tree, jump_size=0.8, gamma=1.):
    old_budget = tree.budget*1.0
    old_tree = copy.deepcopy(tree)

    # delta_budget = jump_size * np.random.randn()  # perturbation
    # new_budget = old_tree.budget + delta_budget  # propose a new budget

    u = random.uniform(jump_size, 1/jump_size)  # try random walk on log scale
    new_budget = u*old_tree.budget
    delta_budget = (u-1)*old_tree.budget

    proposed_tree = copy.deepcopy(tree)
    # if all cost are fixed, then for all blocks in new_tree, the block budget will be altered by delta_budget
    old_remaining_budgets = [l.budget for l in old_tree.leafBlockDic.keys()]
    if (min(old_remaining_budgets) + delta_budget) < 0:  # if perturbing the budget makes any leaf block illegal
        pass  # do nothing
    else:
        # first update the budgets in proposed_tree
        leafBlockLst = [proposed_tree.root]
        while len(leafBlockLst) > 0:  # if we got more than one leaf in the current level
            newLeafBlockLst = []  # store leaf node that will be generated in the next level(depth +1)
            for leafBlock in leafBlockLst:  # scan over the current leaf nodes
                leafBlock.budget += delta_budget
                if leafBlock.leftChild is not None and leafBlock.rightChild is not None:
                    newLeafBlockLst.append(leafBlock.leftChild)  # this is a new leaf now
                    newLeafBlockLst.append(leafBlock.rightChild)  # this is a new leaf now
            # update the leaf nodes of the next level/depth, do for loop to scan over this new leaf list again
            leafBlockLst = newLeafBlockLst  # now this will be the next level of blocks for the while loop

        # then compute the density ratio of two trees
        # recursively removing cuts in tree1, tree2 until the trivial partition
        log_tree_old = 0.
        while len(old_tree.leafCutDic) != 0:
            leafCut = old_tree.getRandomLeafCut()
            # randomly choose a leaf cut, compute the density ratio between the current tree and the tree with this cut
            # merged, repeat this process until going back to trivial partition, get p(tree)/p(trivial)
            if leafCut.cutDir == 0:
                log_tree_old += -(leafCut.columnUB - leafCut.columnLB)*leafCut.rightChild.budget
            else:
                log_tree_old += -(leafCut.rowUB - leafCut.rowLB)*leafCut.rightChild.budget
            old_tree.removeLeafCut(leafCut)
        log_tree_old -= 2*old_budget  # trivial partition: no cut, i.e. cost > old_budget, cost ~ EXP(2)

        log_tree_proposed = 0.
        while len(proposed_tree.leafCutDic) != 0:
            leafCut = proposed_tree.getRandomLeafCut()
            # randomly choose a leaf cut, compute the density ratio between the current tree and the tree with this cut
            # merged, repeat this process until going back to trivial partition, get p(tree)/p(trivial)
            if leafCut.cutDir == 0:
                log_tree_proposed += -(leafCut.columnUB - leafCut.columnLB)*leafCut.rightChild.budget  #TODO: Should it be budget of leaf cut - cost of cut?
            else:
                log_tree_proposed += -(leafCut.rowUB - leafCut.rowLB)*leafCut.rightChild.budget
            proposed_tree.removeLeafCut(leafCut)
        log_tree_proposed -= 2*new_budget  # trivial partition: no cut, i.e. cost > new_budget, cost ~ EXP(2)

        log_tree_ratio = log_tree_proposed - log_tree_old
        # work out prior ratio Gamma(3,2) on budget
        prior_budget_ratio = (3-1)*(np.log(delta_budget+old_budget)-np.log(old_budget)) - 2*delta_budget

        # log_MH_ratio = log_tree_ratio + prior_budget_ratio  # symmetric MH update
        log_MH_ratio = gamma*(log_tree_ratio + prior_budget_ratio) - np.log(u)  # random walk on log scale

        if np.log(random.random()) < log_MH_ratio:  # if accepted
            # print("budget update accepted!")
            tree.budget += delta_budget
            # update tree.root.budget and budgets of all nodes, not just the leaf nodes
            # i.e. update that change in lambda for EVERY BLOCK in the Mondrian Tree
            leafBlockLst = [tree.root]
            while len(leafBlockLst) > 0:  # if we got more than one leaf in the current level
                newLeafBlockLst = []  # store leaf node that will be generated in the next level(depth +1)
                for leafBlock in leafBlockLst:  # scan over the current leaf nodes
                    leafBlock.budget += delta_budget
                    if leafBlock.leftChild is not None and leafBlock.rightChild is not None:
                        newLeafBlockLst.append(leafBlock.leftChild)  # this is a new leaf now
                        newLeafBlockLst.append(leafBlock.rightChild)  # this is a new leaf now
                # update the leaf nodes of the next level/depth, do for loop to scan over this new leaf list again
                leafBlockLst = newLeafBlockLst  # now this will be the next level of blocks for the while loop


def dissect_a_block(leafBlock, xi, eta, data, sigma, row_link, col_link):
    """
    try to add a split between every two consecutive latent coord (either row or column), compute the corresponding
    marginal log lkd of the two smaller blocks and the corresponding interval
    :param leafBlock: the Mondrian block to be dissected
    :param xi: latent row coord
    :param eta: latent column coord
    :param data: the observed 2d array
    :param sigma: std parameter of the Gaussian noise
    :param row_link, col_link
    :return: list of intervals between latent coord, and the marginal log lkds for the two smaller blocks
    if split is in that interval
    """
    if leafBlock.DataPoints is None:  # i.e. if there is no data point in it
        row_interval = [[leafBlock.rowLB, leafBlock.rowUB]]
        col_interval = [[leafBlock.columnLB, leafBlock.columnUB]]
        row_marginal_log_lkd = [[0.0, 0.0]]
        col_marginal_log_lkd = [[0.0, 0.0]]
        row_partition = [[None, None]]
        column_partition = [[None, None]]
        return [row_interval, row_marginal_log_lkd, row_partition, col_interval, col_marginal_log_lkd, column_partition]

    else:  # if it is not empty
        # row/col coord in this block
        row_coord = np.array([xi[_] for _ in leafBlock.DataPoints[0]])
        row_coord_sorted = np.sort(row_coord)
        column_coord = np.array([eta[_] for _ in leafBlock.DataPoints[1]])
        column_coord_sorted = np.sort(column_coord)
        # get all the possible row/column intervals
        row_intervals = [[leafBlock.rowLB, row_coord_sorted[0]]] + [[row_coord_sorted[_], row_coord_sorted[_+1]]
                                                                    for _ in range(len(row_coord_sorted)-1)] + \
                        [[row_coord_sorted[-1], leafBlock.rowUB]]
        column_intervals = [[leafBlock.columnLB, column_coord_sorted[0]]] + [[column_coord_sorted[_], column_coord_sorted[_+1]]
                                                                             for _ in range(len(column_coord_sorted)-1)] + \
                           [[column_coord_sorted[-1], leafBlock.columnUB]]

        # for each possible position, try adding a split, get the resulting marginal lkd based on two little nodes
        row_proposals = []
        row_partition = []
        for interval_idx in range(len(row_intervals)):
            # recall we define the ``left" node to be the one with smaller boundary values
            # get row coord smaller than upperbound of the row interval
            row_indicator = row_coord < row_intervals[interval_idx][1]
            # split the row index into two parts
            # TODO: is it correct? will it return the indices that satisfy the row_indicator?
            block_data_left = list(compress(leafBlock.DataPoints[0], row_indicator))
            block_data_right = list(compress(leafBlock.DataPoints[0], ~row_indicator))

            # register data partition
            if block_data_left == [] and block_data_right == []:
                exit('something wrong with the data tracking process')
            elif block_data_left == []:
                row_partition.append([None,
                                      [block_data_right, leafBlock.DataPoints[1][:]]])
                if block_data_right != leafBlock.DataPoints[0]:
                    exit('something wrong with the data tracking process')
            elif block_data_right == []:
                row_partition.append([[block_data_left, leafBlock.DataPoints[1][:]],
                                      None])
                if block_data_left != leafBlock.DataPoints[0]:
                    exit('something wrong with the data tracking process')
            else:
                row_partition.append([[block_data_left, leafBlock.DataPoints[1][:]],
                                      [block_data_right, leafBlock.DataPoints[1][:]]])
                if sorted(block_data_left + block_data_right) != sorted(leafBlock.DataPoints[0]):
                    exit('something wrong with the data tracking process')

            # if one part contains no data, just return 0
            if block_data_left == []:
                marginal_log_lkd_left = 0.0
            else:
                temp_row, temp_col = cov_idx_to_data_idx(block_data_left, leafBlock.DataPoints[1],
                                                         row_link, col_link)
                marginal_log_lkd_left = marginal_lkd_MC(data[temp_row][:, temp_col, :].reshape(-1, 2), sigma)

            if block_data_right == []:
                marginal_log_lkd_right = 0.0
            else:
                temp_row, temp_col = cov_idx_to_data_idx(block_data_right, leafBlock.DataPoints[1],
                                                         row_link, col_link)
                marginal_log_lkd_right = marginal_lkd_MC(data[temp_row][:, temp_col, :].reshape(-1, 2), sigma)

            row_proposals.append([marginal_log_lkd_left, marginal_log_lkd_right])  # should be the marginal lkd of the smaller blocks

        # do the same thing to column intervals
        column_proposals = []
        column_partition = []
        for interval_idx in range(len(column_intervals)):
            # recall we define the ``left" node to be the one with smaller boundary values
            # get column coord smaller than upperbound of the column interval
            column_indicator = column_coord < column_intervals[interval_idx][1]
            # TODO: is it correct? will it return the indices that satisfy the row_indicator?
            block_data_left = list(compress(leafBlock.DataPoints[1], column_indicator))
            block_data_right = list(compress(leafBlock.DataPoints[1], ~column_indicator))

            # register data partition
            if block_data_left == [] and block_data_right == []:
                exit('something wrong with the data tracking process')
            elif block_data_left == []:
                if block_data_right != leafBlock.DataPoints[1]:
                    exit('something wrong with the data tracking process')
                column_partition.append([None, [leafBlock.DataPoints[0][:], block_data_right]])
            elif block_data_right == []:
                if block_data_left != leafBlock.DataPoints[1]:
                    exit('something wrong with the data tracking process')
                column_partition.append([[leafBlock.DataPoints[0][:], block_data_left], None])
            else:
                if sorted(block_data_left + block_data_right) != sorted(leafBlock.DataPoints[1]):
                    exit('something wrong with the data tracking process')
                column_partition.append([[leafBlock.DataPoints[0][:], block_data_left],
                                         [leafBlock.DataPoints[0][:], block_data_right]])


            # if one part contains no data, just return 0
            if block_data_left == []:
                marginal_log_lkd_left = 0.0
            else:
                temp_row, temp_col = cov_idx_to_data_idx(leafBlock.DataPoints[0], block_data_left,
                                                         row_link, col_link)
                marginal_log_lkd_left = marginal_lkd_MC(data[temp_row][:, temp_col, :].reshape(-1, 2), sigma)

            if block_data_right == []:
                marginal_log_lkd_right = 0.0
            else:
                temp_row, temp_col = cov_idx_to_data_idx(leafBlock.DataPoints[0], block_data_right,
                                                         row_link, col_link)
                marginal_log_lkd_right = marginal_lkd_MC(data[temp_row][:, temp_col, :].reshape(-1, 2), sigma)

            column_proposals.append([marginal_log_lkd_left, marginal_log_lkd_right])  # should be the marginal lkd of the smaller blocks

        return [row_intervals, row_proposals, row_partition, column_intervals, column_proposals, column_partition]




def rjmcmc_split2(data, sigma, tree, xi, eta, row_link, col_link, gamma=1.):
    # smarter version utilizing Gibbs style proposal

    # randomly choose a leaf block for cutting
    leafBlock = tree.getRandomLeafBlock()
    length = leafBlock.rowUB - leafBlock.rowLB
    width = leafBlock.columnUB - leafBlock.columnLB

    # dissect this block
    [row_intervals, row_marginal_log_lkd, row_partition, column_intervals, column_marginal_log_lkd, column_partition] = \
        dissect_a_block(leafBlock, xi, eta, data, sigma, row_link, col_link)

    num_of_row_interval = len(row_intervals)
    num_of_column_interval = len(column_intervals)
    gap_of_row_interval = [_[1]-_[0] for _ in row_intervals]
    gap_of_column_interval = [_[1]-_[0] for _ in column_intervals]
    updated_row_marginal_lkd = [_[1]+_[0] for _ in row_marginal_log_lkd]
    updated_column_marginal_lkd = [_[1]+_[0] for _ in column_marginal_log_lkd]

    # proposal density is proportional to the product of interval size and the updated marginal lkd based on
    # the two smaller blocks
    proposal_log_density = gamma*np.array(updated_row_marginal_lkd + updated_column_marginal_lkd) + np.log(np.array(gap_of_row_interval + gap_of_column_interval))

    max_proposal = np.max(proposal_log_density)
    proposal_log_density -= max_proposal
    proposal_prob = np.exp(proposal_log_density)
    proposal_prob = proposal_prob/np.sum(proposal_prob)
    # proposed_interval_label = np.random.choice(num_of_row_interval+num_of_column_interval, 1, p=proposal_prob)[0]
    proposed_interval_label = random.choices(population=list(range(num_of_row_interval+num_of_column_interval)),
                                             weights=proposal_prob, k=1)[0]
    # this is the chosen interval of split

    # once have obtained the label, propose cut direction and cut position
    if proposed_interval_label < num_of_row_interval:  # i.e. a row cut
        cutDir = 0  # row cut
        band = row_intervals[proposed_interval_label]
        # cutPos = np.random.uniform(low=band[0], high=band[1], size=1)[0]
        cutPos = random.uniform(band[0], band[1])
    else:
        cutDir = 1  # col cut
        band = column_intervals[proposed_interval_label-num_of_row_interval]
        # cutPos = np.random.uniform(low=band[0], high=band[1], size=1)[0]
        cutPos = random.uniform(band[0], band[1])

    # now propose cut cost conditioned on cut position and construct the blocks
    if cutDir == 0:
        cost = np.log(random.random() * (np.exp(gamma * width * leafBlock.budget) - 1) + 1)/(gamma * width)

        leftChild = MondrianBlock(leafBlock.budget-cost, leafBlock.rowLB, cutPos,
                                  leafBlock.columnLB, leafBlock.columnUB, None, None,
                                  None, None, leafBlock)
        rightChild = MondrianBlock(leafBlock.budget-cost, cutPos, leafBlock.rowUB,
                                   leafBlock.columnLB, leafBlock.columnUB, None, None,
                                   None, None, leafBlock)

        leftChild.DataPoints = row_partition[proposed_interval_label][0]
        rightChild.DataPoints = row_partition[proposed_interval_label][1]
        leftChild.marginal_lkd = row_marginal_log_lkd[proposed_interval_label][0]
        rightChild.marginal_lkd = row_marginal_log_lkd[proposed_interval_label][1]

    else:  # if it is a column cut
        cost = np.log(random.random() * (np.exp(gamma * length * leafBlock.budget) - 1) + 1)/(gamma * length)
        leftChild = MondrianBlock(leafBlock.budget-cost,
                                  leafBlock.rowLB, leafBlock.rowUB,
                                  leafBlock.columnLB, cutPos, None, None,
                                  None, None, leafBlock)
        rightChild = MondrianBlock(leafBlock.budget-cost,
                                   leafBlock.rowLB, leafBlock.rowUB,
                                   cutPos, leafBlock.columnUB, None, None,
                                   None, None, leafBlock)

        leftChild.DataPoints = column_partition[proposed_interval_label-num_of_row_interval][0]
        rightChild.DataPoints = column_partition[proposed_interval_label-num_of_row_interval][1]
        leftChild.marginal_lkd = column_marginal_log_lkd[proposed_interval_label-num_of_row_interval][0]
        rightChild.marginal_lkd = column_marginal_log_lkd[proposed_interval_label-num_of_row_interval][1]

    # log lkd ratio in acceptance rate
    log_lkd_ratio = max_proposal + np.log(np.sum(np.exp(proposal_log_density))) - gamma*leafBlock.marginal_lkd

    # proposal ratio
    proposalRatio = (len(tree.getLeafBlockDic())/(len(tree.getLeafCutDic())+1.0))
    if leafBlock.getParent() in tree.getLeafCutDic().keys():
        # if tree.getLeafCutDic().has_key(leafBlock.getParent()):
        proposalRatio = (len(tree.getLeafBlockDic())/len(tree.getLeafCutDic()))

    # prior ratio on cost
    if cutDir == 0:  # recall proposal of cut cost is conditioned on te cut direction
        prior_ratio = (1-np.exp(-gamma*width*leafBlock.budget))/(gamma*width)
    else:  # if is a column cut
        prior_ratio = (1-np.exp(-gamma*length*leafBlock.budget))/(gamma*length)

    log_MH_ratio = np.log(prior_ratio) + np.log(proposalRatio) + log_lkd_ratio

    if np.log(random.random()) < log_MH_ratio:
        # print('add cut accepted!')
        # finally ,add this new cut to the tree
        tree.addCut(leafBlock, cutDir, cutPos, leftChild, rightChild)
        return 1  # accepted
    else:
        return 0  # rejected



def rjmcmc_merge2(data, sigma, tree, xi, eta, row_link, col_link, gamma=1.):
    # smarter version utilizing Gibbs style proposal

    # randomly sample a leaf cut
    leafCut = tree.getRandomLeafCut()

    # if we got a trivial partition, just skip, otherwise:
    if leafCut is None:
        return 0
    else:
        length = leafCut.rowUB - leafCut.rowLB
        width = leafCut.columnUB - leafCut.columnLB
        leftChildLogLikelihood = leafCut.leftChild.marginal_lkd
        rightChildLogLikelihood = leafCut.rightChild.marginal_lkd

        # work out the joint indices in this block
        left_row_idx = [] if leafCut.leftChild.DataPoints is None else leafCut.leftChild.DataPoints[0][:]
        left_column_idx = [] if leafCut.leftChild.DataPoints is None else leafCut.leftChild.DataPoints[1][:]
        right_row_idx = [] if leafCut.rightChild.DataPoints is None else leafCut.rightChild.DataPoints[0][:]
        right_column_idx = [] if leafCut.rightChild.DataPoints is None else leafCut.rightChild.DataPoints[1][:]

        if leafCut.cutDir == 0:  # if we got a row cut, work out the merged indices and marginal lkd
            # first the joint indices
            if leafCut.rightChild.DataPoints is None and leafCut.leftChild.DataPoints is None:
                joint_indices = None
            elif leafCut.rightChild.DataPoints is None:
                if leafCut.leftChild.DataPoints[0] == [] or leafCut.leftChild.DataPoints[1] == []:
                    exit('error in merging tow blocks')
                joint_indices = [left_row_idx, left_column_idx]
                if joint_indices != leafCut.leftChild.DataPoints:
                    exit('error in merging tow blocks')
            elif leafCut.leftChild.DataPoints is None:
                if leafCut.rightChild.DataPoints[0] == [] or leafCut.rightChild.DataPoints[1] == []:
                    exit('error in merging tow blocks')
                joint_indices = [right_row_idx, right_column_idx]
                if joint_indices != leafCut.rightChild.DataPoints:
                    exit('error in merging tow blocks')
            else:
                if sorted(left_column_idx) != sorted(right_column_idx):
                    exit('error in merging tow blocks')
                joint_row_idx = left_row_idx + right_row_idx
                joint_indices = [joint_row_idx, left_column_idx]

            # work out the merged marginal lkd
            if leafCut.rightChild.DataPoints is None and leafCut.leftChild.DataPoints is None:
                joint_marginal_lkd = 0.0
            elif leafCut.leftChild.DataPoints is None:
                joint_marginal_lkd = rightChildLogLikelihood
            elif leafCut.rightChild.DataPoints is None:
                joint_marginal_lkd = leftChildLogLikelihood
            else:
                # TODO: replace it with block method?
                temp_row, temp_col = cov_idx_to_data_idx(joint_indices[0], joint_indices[1],
                                                         row_link, col_link)
                joint_marginal_lkd = marginal_lkd_MC(data[temp_row][:, temp_col, :].reshape(-1, 2), sigma)

        else:  # if we got a column cut, same thing, work out the merged indices and marginal lkd
            # first the joint indices
            if leafCut.rightChild.DataPoints is None and leafCut.leftChild.DataPoints is None:
                joint_indices = None
            elif leafCut.rightChild.DataPoints is None:
                if leafCut.leftChild.DataPoints[0] == [] or leafCut.leftChild.DataPoints[1] == []:
                    exit('error in merging tow blocks')
                joint_indices = [left_row_idx, left_column_idx]
                if joint_indices != leafCut.leftChild.DataPoints:
                    exit('error in merging tow blocks')
            elif leafCut.leftChild.DataPoints is None:
                if leafCut.rightChild.DataPoints[0] == [] or leafCut.rightChild.DataPoints[1] == []:
                    exit('error in merging tow blocks')
                joint_indices = [right_row_idx, right_column_idx]
                if joint_indices != leafCut.rightChild.DataPoints:
                    exit('error in merging tow blocks')
            else:
                if sorted(right_row_idx) != sorted(left_row_idx):
                    exit('error in merging tow blocks')
                joint_column_idx = left_column_idx + right_column_idx
                joint_indices = [left_row_idx, joint_column_idx]

            # then the joint lkd
            if leafCut.rightChild.DataPoints is None and leafCut.leftChild.DataPoints is None:
                joint_marginal_lkd = 0.0
            elif leafCut.leftChild.DataPoints is None:
                joint_marginal_lkd = rightChildLogLikelihood
            elif leafCut.rightChild.DataPoints is None:
                joint_marginal_lkd = leftChildLogLikelihood
            else:
                temp_row, temp_col = cov_idx_to_data_idx(joint_indices[0], joint_indices[1],
                                                         row_link, col_link)
                joint_marginal_lkd = marginal_lkd_MC(data[temp_row][:, temp_col, :].reshape(-1, 2), sigma)



        place_holder_block = MondrianBlock(leafCut.budget,
                                           leafCut.rowLB, leafCut.rowUB,
                                           leafCut.columnLB, leafCut.columnUB, None, None,
                                           None, None, None)
        place_holder_block.DataPoints = joint_indices

        # dissect this fake block, only need the boundaries and the DataPoints to run dissect
        [row_intervals, row_marginal_log_lkd, row_partition, column_intervals, column_marginal_log_lkd, column_partition] = \
            dissect_a_block(place_holder_block, xi, eta, data, sigma, row_link, col_link)

        gap_of_row_interval = [_[1]-_[0] for _ in row_intervals]
        gap_of_column_interval = [_[1]-_[0] for _ in column_intervals]
        updated_row_marginal_lkd = [_[1]+_[0] for _ in row_marginal_log_lkd]
        updated_column_marginal_lkd = [_[1]+_[0] for _ in column_marginal_log_lkd]
        # proposal density is proportional to the product of interval size and the updated marginal lkd based on
        # the two smaller blocks

        proposal_log_density = gamma*np.array(updated_row_marginal_lkd + updated_column_marginal_lkd) + np.log(np.array(gap_of_row_interval + gap_of_column_interval))

        max_proposal = np.max(proposal_log_density)
        proposal_log_density -= max_proposal

        # log lkd ratio
        log_lkd_ratio = gamma*joint_marginal_lkd - max_proposal - np.log(np.sum(np.exp(proposal_log_density)))

        # proposal ratio
        proposalRatio = len(tree.getLeafCutDic()) / (len(tree.getLeafBlockDic())-1.0)

        # prior ratio for cost
        if leafCut.cutDir == 0:  # if we got a row cut
            prior_ratio = (gamma*width)/(1-np.exp(-gamma*width*leafCut.budget))
        else:  # if we got a column cut
            prior_ratio = (gamma*length)/(1-np.exp(-gamma*length*leafCut.budget))

        log_MH_ratio = np.log(prior_ratio) + np.log(proposalRatio) + log_lkd_ratio

        if log_MH_ratio >= np.log(random.random()):  # accept removing a leaf cut
            # print('remove cut accepted!')
            # register the cut, recall we need to manually add marginal lkd and data points to it
            tree.removeLeafCut(leafCut)
            # add the marginal lkd and data points to leafCut
            leafCut.marginal_lkd = joint_marginal_lkd
            leafCut.DataPoints = joint_indices
            return 1  # accept
        else:
            return 0  # reject


def rjmcmc_redo_a_cut2(data, sigma, leafCut, xi, eta, tree, row_link, col_link, gamma=1.):
    # smarter version, redo the leaf cut of the tree using Gibbs update on cutPos and cost
    length = leafCut.rowUB - leafCut.rowLB
    width = leafCut.columnUB - leafCut.columnLB

    # construct two new blocks with new configurations i.e. cost, cutting position, marginal lkd, data points...
    # gibbs update so should always accept?

    # first redo cost:
    if leafCut.cutDir == 0:
        new_cost = np.log(random.random() * (np.exp(gamma * width * leafCut.budget) - 1) + 1)/(gamma*width)
    else:
        new_cost = np.log(random.random() * (np.exp(gamma * length * leafCut.budget) - 1) + 1)/(gamma*length)

    # now redo cut:
    # first merge the two blocks into a bigger one, taken from merge2
    leftChildLogLikelihood = leafCut.leftChild.marginal_lkd
    rightChildLogLikelihood = leafCut.rightChild.marginal_lkd

    # work out the joint indices in this block
    left_row_idx = [] if leafCut.leftChild.DataPoints is None else leafCut.leftChild.DataPoints[0][:]
    left_column_idx = [] if leafCut.leftChild.DataPoints is None else leafCut.leftChild.DataPoints[1][:]
    right_row_idx = [] if leafCut.rightChild.DataPoints is None else leafCut.rightChild.DataPoints[0][:]
    right_column_idx = [] if leafCut.rightChild.DataPoints is None else leafCut.rightChild.DataPoints[1][:]

    if leafCut.cutDir == 0:  # if we got a row cut, work out the merged indices and marginal lkd
        # first the joint indices
        if leafCut.rightChild.DataPoints is None and leafCut.leftChild.DataPoints is None:
            joint_indices = None
        elif leafCut.rightChild.DataPoints is None:
            if leafCut.leftChild.DataPoints[0] == [] or leafCut.leftChild.DataPoints[1] == []:
                exit('error in merging tow blocks')
            joint_indices = [left_row_idx, left_column_idx]
            if joint_indices != leafCut.leftChild.DataPoints:
                exit('error in merging tow blocks')
        elif leafCut.leftChild.DataPoints is None:
            if leafCut.rightChild.DataPoints[0] == [] or leafCut.rightChild.DataPoints[1] == []:
                exit('error in merging tow blocks')
            joint_indices = [right_row_idx, right_column_idx]
            if joint_indices != leafCut.rightChild.DataPoints:
                exit('error in merging tow blocks')
        else:
            if sorted(left_column_idx) != sorted(right_column_idx):
                exit('error in merging tow blocks')
            joint_row_idx = left_row_idx + right_row_idx
            joint_indices = [joint_row_idx, left_column_idx]

        # work out the merged marginal lkd
        if leafCut.rightChild.DataPoints is None and leafCut.leftChild.DataPoints is None:
            joint_marginal_lkd = 0.0
        elif leafCut.leftChild.DataPoints is None:
            joint_marginal_lkd = rightChildLogLikelihood
        elif leafCut.rightChild.DataPoints is None:
            joint_marginal_lkd = leftChildLogLikelihood
        else:
            # TODO: replace it with block method?
            temp_row, temp_col = cov_idx_to_data_idx(joint_indices[0], joint_indices[1],
                                                     row_link, col_link)
            joint_marginal_lkd = marginal_lkd_MC(data[temp_row][:, temp_col, :].reshape(-1, 2), sigma)

    else:  # if we got a column cut, same thing, work out the merged indices and marginal lkd
        # first the joint indices
        if leafCut.rightChild.DataPoints is None and leafCut.leftChild.DataPoints is None:
            joint_indices = None
        elif leafCut.rightChild.DataPoints is None:
            if leafCut.leftChild.DataPoints[0] == [] or leafCut.leftChild.DataPoints[1] == []:
                exit('error in merging tow blocks')
            joint_indices = [left_row_idx, left_column_idx]
            if joint_indices != leafCut.leftChild.DataPoints:
                exit('error in merging tow blocks')
        elif leafCut.leftChild.DataPoints is None:
            if leafCut.rightChild.DataPoints[0] == [] or leafCut.rightChild.DataPoints[1] == []:
                exit('error in merging tow blocks')
            joint_indices = [right_row_idx, right_column_idx]
            if joint_indices != leafCut.rightChild.DataPoints:
                exit('error in merging tow blocks')
        else:
            if sorted(right_row_idx) != sorted(left_row_idx):
                exit('error in merging tow blocks')
            joint_column_idx = left_column_idx + right_column_idx
            joint_indices = [left_row_idx, joint_column_idx]

        # then the joint lkd
        if leafCut.rightChild.DataPoints is None and leafCut.leftChild.DataPoints is None:
            joint_marginal_lkd = 0.0
        elif leafCut.leftChild.DataPoints is None:
            joint_marginal_lkd = rightChildLogLikelihood
        elif leafCut.rightChild.DataPoints is None:
            joint_marginal_lkd = leftChildLogLikelihood
        else:
            temp_row, temp_col = cov_idx_to_data_idx(joint_indices[0], joint_indices[1],
                                                     row_link, col_link)
            joint_marginal_lkd = marginal_lkd_MC(data[temp_row][:, temp_col, :].reshape(-1, 2), sigma)

    # the merged block, place holder
    place_holder_block = MondrianBlock(leafCut.budget,
                                       leafCut.rowLB, leafCut.rowUB,
                                       leafCut.columnLB, leafCut.columnUB, None, None,
                                       None, None, None)
    place_holder_block.DataPoints = joint_indices
    place_holder_block.marginal_lkd = joint_marginal_lkd

    # now redo a cut, slightly different from split

    # dissect this block
    [row_intervals, row_marginal_log_lkd, row_partition, column_intervals, column_marginal_log_lkd, column_partition] = \
        dissect_a_block(place_holder_block, xi, eta, data, sigma, row_link, col_link)

    num_of_row_interval = len(row_intervals)
    num_of_column_interval = len(column_intervals)
    gap_of_row_interval = [_[1]-_[0] for _ in row_intervals]
    gap_of_column_interval = [_[1]-_[0] for _ in column_intervals]
    updated_row_marginal_lkd = [_[1]+_[0] for _ in row_marginal_log_lkd]
    updated_column_marginal_lkd = [_[1]+_[0] for _ in column_marginal_log_lkd]
    # proposal density is proportional to the product of interval size and the updated marginal lkd based on
    # the two smaller blocks

    # note that we are using a different Gibbs proposal
    proposal_log_density1 = gamma*np.array(updated_row_marginal_lkd + updated_column_marginal_lkd) + np.log(np.array(gap_of_row_interval + gap_of_column_interval))
    proposal_log_density2 = gamma*(leafCut.budget - new_cost) * np.concatenate(((-2*width-length)*np.ones(num_of_row_interval), (-2*length-width)*np.ones(num_of_column_interval)))
    proposal_log_density = proposal_log_density1 + proposal_log_density2
    # note that we are using a different proposal

    max_proposal = np.max(proposal_log_density)
    proposal_log_density -= max_proposal
    proposal_prob = np.exp(proposal_log_density)
    proposal_prob = proposal_prob/np.sum(proposal_prob)
    # proposed_interval_label = np.random.choice(num_of_row_interval+num_of_column_interval, 1, p=proposal_prob)[0]
    proposed_interval_label = random.choices(population=list(range(num_of_row_interval+num_of_column_interval)),
                                             weights=proposal_prob, k=1)[0]
    # this is the chosen interval of split

    # once have obtained the label, propose cut direction and cut position
    if proposed_interval_label < num_of_row_interval:  # i.e. a row cut
        new_cutDir = 0  # row cut
        band = row_intervals[proposed_interval_label]
        # new_cutPos = np.random.uniform(low=band[0], high=band[1], size=1)[0]
        new_cutPos = random.uniform(band[0], band[1])
    else:
        new_cutDir = 1  # col cut
        band = column_intervals[proposed_interval_label-num_of_row_interval]
        # new_cutPos = np.random.uniform(low=band[0], high=band[1], size=1)[0]
        new_cutPos = random.uniform(band[0], band[1])

    # now construct the blocks
    if new_cutDir == 0:
        proposed_leftChild = MondrianBlock(leafCut.budget-new_cost, leafCut.rowLB, new_cutPos,
                                           leafCut.columnLB, leafCut.columnUB, None, None,
                                           None, None, leafCut)
        proposed_rightChild = MondrianBlock(leafCut.budget-new_cost, new_cutPos, leafCut.rowUB,
                                            leafCut.columnLB, leafCut.columnUB, None, None,
                                            None, None, leafCut)

        proposed_leftChild.DataPoints = row_partition[proposed_interval_label][0]
        proposed_rightChild.DataPoints = row_partition[proposed_interval_label][1]
        proposed_leftChild.marginal_lkd = row_marginal_log_lkd[proposed_interval_label][0]
        proposed_rightChild.marginal_lkd = row_marginal_log_lkd[proposed_interval_label][1]

    else:  # if proposed a column cut
        proposed_leftChild = MondrianBlock(leafCut.budget-new_cost, leafCut.rowLB, leafCut.rowUB,
                                           leafCut.columnLB, new_cutPos, None, None,
                                           None, None, leafCut)
        proposed_rightChild = MondrianBlock(leafCut.budget-new_cost, leafCut.rowLB, leafCut.rowUB,
                                            new_cutPos, leafCut.columnUB, None, None,
                                            None, None, leafCut)

        proposed_leftChild.DataPoints = column_partition[proposed_interval_label-num_of_row_interval][0]
        proposed_rightChild.DataPoints = column_partition[proposed_interval_label-num_of_row_interval][1]
        proposed_leftChild.marginal_lkd = column_marginal_log_lkd[proposed_interval_label-num_of_row_interval][0]
        proposed_rightChild.marginal_lkd = column_marginal_log_lkd[proposed_interval_label-num_of_row_interval][1]

    tree.removeLeafCut(leafCut)
    # then add the new blocks
    tree.addCut(leafCut, new_cutDir, new_cutPos, proposed_leftChild, proposed_rightChild)  # add the proposed cut
    # recall leftChild_proposed, rightChild_proposed are equipped with data points and marginal log lkd




# ######################### now we got all the ingredients, build the actual sampler
# TODO: update budget
# TODO: parallel tempering, deep copy, prior ratio of p(M) can be computed using the one step ratios
#  i.e. recursively remove merging until going back to trivial partition. Telescoping
# TODO: DP coloring
# TODO: random cluster coloring




def leaf_string_decoder2(str_tree, xi, eta, fig=False):
    """
    decode the str_tree
    :param str_tree: str representation of the partition
    :param xi: latent row coord
    :param eta: latent col coord
    :return: clustering of cells as a dictionary, cells are represented as an int (r*C+r);
    pairs is a binary matrix, two cells are in the same cluster iff pairs[cell1][cell2] = 1
    """
    R = len(xi)
    C = len(eta)
    leaf_lst = str_tree.split(";")
    leaf_array = [[] for _ in range(len(leaf_lst))]
    # decode the string
    for i, s in enumerate(leaf_lst):
        leaf_array[i] = [eval(j) for j in s.split("|")]
        # leaf_array[i] contains rowLB, rowUB, colLB, colUB, label of the block
    # each value of clustering contains the scalar index r*C+c of cells that fall inside this leaf
    clustering = {str(my_leaf[4]): [] for my_leaf in leaf_array}
    for r in range(R):
        for c in range(C):
            for vertices in leaf_array:
                if vertices[0] <= xi[r] <= vertices[1] and vertices[2] <= eta[c] <= vertices[3]:
                    clustering[str(vertices[4])] += [r*C+c]
                    break
    # pairs[i][j] = 1 means cell (k,l) and cell (m,n) in same cluster, i=k*C+l, j=m*C+n, just scalar index for the entry
    pairs = sparse.eye(R*C, dtype=np.int8).tolil()
    for leaf in clustering.keys():
        for iii in clustering[leaf]:
            for jjj in clustering[leaf]:
                pairs[iii, jjj] = 1

    if fig:
        plt.axes()
        for vertices in leaf_array:
            rectangle = plt.Rectangle((vertices[2], vertices[0]), vertices[3]-vertices[2], vertices[1]-vertices[0],
                                      edgecolor='k', facecolor='none')
            plt.gca().add_patch(rectangle)
        for r in range(len(xi)):
            for c in range(len(eta)):
                plt.scatter(x=eta[c], y=xi[r], c="k")
                plt.annotate(text="({},{})".format(r, c), xy=(eta[c], xi[r]))
        plt.gca().invert_yaxis()
        plt.show()

    return clustering, pairs

# TODO: check the budget and cost update?
# for budget parameter and the cost parameter for each non-leaf block, perform i round of MH within Gibbs



def partition_sampler_alt(data, row_link, col_link, scale, random_MP_init=False, budget=1.0,
                          maxGibbsIteration_coord = 1, thin=5, burnin=1000, p_row=0.3, p_col=0.3, max_init_cut=4,
                          maxIteration = 2000, new_start=False, likelihoodFileName=None,
                          representationFileName=None, sigFileName=None, budgetFileName=None,
                          fixed_ordering_col=False, fixed_ordering_row=False,
                          check_point='MCMC_check_point.dictionary', budget_update=False, gamma=1., MH=False):
    # TODO: attach a MH update for sigma, just sum the marginal lkd over all blocks should do
    # TODO: attach a MH update for budget parameter of MP, should only involve termination probabilities for each block
    # TODO: col and row should be controled by two fixed_ordering term, fix!
    """
    fix the std parameter sigma, update MP tree structure and the latent coord xi, eta
    :param data: 2d array of data, data[i][j] returns entry in cell (i,j)
    :param row_link, col_link: r_link[i] give all rows in data that is related to the ith peptide,
        c_link[i] give all columns in data that is related to the ith treatment,
    :param budget: budget parameter of MP
    :param scale: scaling factor of the box, None=unit square, float=proportional to size of mtrx
    :param random_MP_init: initialize MP randomly?
    :param maxGibbsIteration_coord: round of Gibbs update for xi, eta
    :param thin: thinning interval
    :param burnin: burin_in period
    :param p_row, p_col: proportion of row/column entries to be updated in Gibbs_coord_update
    :param maxIteration: # of max iteration
    :param new_start: true if start from scratch, false -> read check point from check_point
    :param likelihoodFileName: file to store the marginal lkd
    :param representationFileName: store tree topology and latent coordinates for each MCMC sample
    :param sigFileName: file to store sigma values
    :param fixed_ordering_col: do we fix the ordering of col?
    :param fixed_ordering_row: do we fix the ordering of row?
    :param check_point: location of the last checkpoint, will start from here if not None, random initialization o.w.
    :param budget_update: do we want to update the budget of MP?
    :param gamma: the tempering parameter
    :param: use MH to update if True, or Gibbs o.w.
    :return: o boi im tired
    """

    # Initialization
    # note that here budget hyper parameter of MP is set to be one, may want to put a prior on it
    if new_start:  # initialization
        itr = 0
        sigma = 1.0  # initialize at 1
        sigma_store, budget_store = [], []
        if scale:
            ratio = len(row_link)/len(col_link)
        else:
            ratio = 1
        box_length = np.round((2*ratio)/(1+ratio), 2)
        box_width = np.round(2/(1+ratio), 2)  # ensures that perimeter=4 while length/width=ratio
        xi = [random.uniform(0, box_length) for _ in row_link]
        if fixed_ordering_row:
            xi = sorted(xi)
        eta = [random.uniform(0, box_width) for _ in col_link if len(data) > 0]
        if fixed_ordering_col:
            eta = sorted(eta)
        tree = MondrianTree(budget, 0, box_length, 0, box_width, random_MP_init, max_cut=max_init_cut)  # initialize
        tree.cell_allocate(xi, eta)
        for l in tree.leafBlockDic.keys():
            l.marginal_lkd_block_level(data=data, xi=xi, eta=eta, p=2, marginal_lkd_func=marginal_lkd_MC,
                                       row_link=row_link, col_link=col_link, test=False, sigma=sigma)

        # initialize a Mondrian tree and random row,col coord, run a few round of Gibbs update
        [xi, eta, logLikelihood] = Gibbs_for_coord(sigma, data, xi, eta, tree, row_link, col_link,
                                                   maxGibbsIteration=maxGibbsIteration_coord,
                                                   isPreGibbsLikelihood=True,
                                                   fixed_ordering_col=fixed_ordering_col,
                                                   fixed_ordering_row=fixed_ordering_row,
                                                   p_row=1.0, p_col=1.0, gamma=gamma)  # update all coordinates

        # utility for writing simulation results into RJMCMC and Representation.txt
        if likelihoodFileName is None:
            likelihoodFileName = 'RJMCMC.txt'
        if representationFileName is None:
            representationFileName = 'Representation.txt'
        if sigFileName is None:
            sigFileName = 'Sigma_values.txt'
        if budgetFileName is None:
            budgetFileName = 'Budget_values.txt'
        fLogLikelihood = open(likelihoodFileName, 'w')
        fRepresentation = open(representationFileName, 'w')
        fSigma = open(sigFileName, 'w')
        fBudget = open(budgetFileName, 'w')

    else:  # start from the last check point
        sigma_store, budget_store = [], []
        with open(check_point, 'rb') as config_dictionary_file:
            checkpoint_dict = pickle.load(config_dictionary_file)
        tree = checkpoint_dict['tree']
        xi = checkpoint_dict['xi']
        eta = checkpoint_dict['eta']
        sigma = checkpoint_dict['sigma']
        itr = checkpoint_dict['itr']
        gamma = checkpoint_dict['gamma']
        likelihoodFileName = checkpoint_dict['lkd_file']
        representationFileName = checkpoint_dict['representation_file']
        sigFileName = checkpoint_dict['sigma_file']
        budgetFileName = checkpoint_dict['budget_file']
        fLogLikelihood = open(likelihoodFileName, 'a')
        fRepresentation = open(representationFileName, 'a')
        fSigma = open(sigFileName, 'a')
        fBudget = open(budgetFileName, 'a')


    startTime = time.time()

    split_accept = 0
    merge_accept = 0
    # RJ MCMC step followed by a few round of Gibbs update on row/col coord
    for itr in range(itr, itr+maxIteration):
        sub_tt = time.time()
        # print('iteration: {}'.format(itr))
        # first do RJMCMC update on tree topology
        print(itr)
        for n_rj in range((len(tree.leafBlockDic)//20)+1):
            move = random.choice([0, 1])  # 0=split, 1=merge
            if move == 0:  # 0=add cut
                # print('add a cut?')
                indicator = rjmcmc_split2(data, sigma, tree, xi, eta, row_link, col_link, gamma)
                split_accept += indicator
            else:  # 1=merge two leaf nodes
                # print('delete a cut?')
                indicator = rjmcmc_merge2(data, sigma, tree, xi, eta, row_link, col_link, gamma)
                merge_accept += indicator

            # randomly select a block, redo the cut
            # print('redo a cut')
            leafCut = tree.getRandomLeafCut()
            if leafCut is not None:
                rjmcmc_redo_a_cut2(data, sigma, leafCut, xi, eta, tree, row_link, col_link, gamma)
        print('Tree_topology_update: {}'.format(time.time() - sub_tt))
        sub_tt = time.time()
        # then do Gibbs update on latent coord isPre=False means
        #         # we want the log lkd after the gibbs updates!
        # print('gibbs updates for coords')
        if len(tree.leafBlockDic) > 80:
            p_col_prime = 0.5*p_col
            p_row_prime = 0.5*p_row
        else:
            p_col_prime = p_col
            p_row_prime = p_row

        if len(tree.leafBlockDic) < 100:
            if MH:
                [xi, eta, logLikelihood] = MH_for_coord(sigma, data, xi, eta, tree, row_link, col_link,
                                                        maxMHIteration=maxGibbsIteration_coord,
                                                        isPreMHLikelihood=False,
                                                        fixed_ordering_col=fixed_ordering_col,
                                                        fixed_ordering_row=fixed_ordering_row,
                                                        p_row=p_row, p_col=p_col, gamma=gamma)  # only updating a portion of the coords
            else:
                [xi, eta, logLikelihood] = Gibbs_for_coord(sigma, data, xi, eta, tree, row_link, col_link,
                                                           maxGibbsIteration=maxGibbsIteration_coord,
                                                           isPreGibbsLikelihood=False,
                                                           fixed_ordering_col=fixed_ordering_col,
                                                           fixed_ordering_row=fixed_ordering_row,
                                                           p_row=p_row_prime, p_col=p_col_prime, gamma=gamma)  # only updating a portion of the coords
        else:
            [xi, eta, logLikelihood] = MH_for_coord(sigma, data, xi, eta, tree, row_link, col_link,
                                                    maxMHIteration=maxGibbsIteration_coord,
                                                    isPreMHLikelihood=False,
                                                    fixed_ordering_col=fixed_ordering_col,
                                                    fixed_ordering_row=fixed_ordering_row,
                                                    p_row=p_row, p_col=p_col, gamma=gamma)  # only updating a portion of the coords
        print('Coordinate_update: {}'.format(time.time() - sub_tt))
        sub_tt = time.time()
        # print('sigma update')
        # now update sigma
        sigma = sigma_MH_in_Gibbs(data, sigma, tree, row_link, col_link, jump_size=0.8, max_iter=4, gamma=gamma)
        sigma_store += [sigma*1.0]
        print('Sigma_update: {}'.format(time.time() - sub_tt))
        sub_tt = time.time()
        # print('cost update')
        # now update cost and cutting position of some randomly chosen blocks
        ll = int(0.1*(len(tree.rowCutDic)+len(tree.columnCutDic)))+1
        for _ in range(min(3, ll)):
            MH_update_cutting_cost(tree, gamma=gamma)
            MH_update_cutting_position(tree, xi, eta, data, col_link, row_link, sigma, gamma=gamma)
            if budget_update:
                # print('budget update')
                # now update budget
                budget_MH_in_Gibbs(tree, jump_size=0.9, gamma=gamma)
        budget_store += [tree.budget]
        print('Budget/Cost_update: {}'.format(time.time() - sub_tt))
        if itr % thin == 0 and itr >= burnin:
            print('Running Time: {}'.format(time.time() - startTime))
            print("Iteration: {}".format(itr))
            print('log-likelihood: {}'.format(logLikelihood))
            print('num of leaf: {}'.format(len(tree.leafBlockDic)))
            print('sigma value: {}'.format(sigma))
            print('budget value: {}'.format(tree.budget))
            print([l.marginal_lkd for l in tree.leafBlockDic.keys()])
            print([l.DataPoints for l in tree.leafBlockDic.keys()])
            fLogLikelihood.write('%s\n' % logLikelihood)
            fRepresentation.write('Iteration: %d\n' % itr)
            leaf_string(tree, fRepresentation)
            fRepresentation.write('%s\n' % xi)
            fRepresentation.write('%s\n' % eta)
            fSigma.write('%s\n' % sigma)
            fBudget.write('%s\n' % tree.budget)
            # save the check point
            MCMC_dictionary = {"tree": tree, "xi": xi, "eta": eta, "sigma": sigma, "itr": itr+1, "lkd_file": likelihoodFileName,
                               'representation_file': representationFileName, "sigma_file": sigFileName, 'budget_file': budgetFileName, 'gamma': gamma}
            with open(check_point, 'wb') as MCMC_checkpoint_file:
                pickle.dump(MCMC_dictionary, MCMC_checkpoint_file)
            if itr % 10 == 0:
                fLogLikelihood.flush()
                fRepresentation.flush()
                fSigma.flush()
                fBudget.flush()

    print('Total Running Time: {}'.format(time.time() - startTime))
    fLogLikelihood.close()
    fRepresentation.close()
    fSigma.close()
    fBudget.close()

    print('Split accept prob: {}, Merge accept prob: {}'.format(1.0*split_accept/maxIteration,
                                                                1.0*merge_accept/maxIteration))
    return sigma_store, budget_store


def partition_sampler_wrap(kwargs):
    sigma_store, budget_store = partition_sampler_alt(**kwargs)
    # data = kwargs["data"]
    # row_link = kwargs["row_link"]
    # col_link = kwargs["col_link"]
    # maxGibbsIteration_coord = kwargs["maxGibbsIteration_coord"]
    # thin = kwargs["thin"]
    # burnin = kwargs["burnin"]
    # maxIteration = kwargs["maxIteration"]
    # new_start = kwargs["new_start"]
    # likelihoodFileName = kwargs["likelihoodFileName"]
    # representationFileName = kwargs["representationFileName"]
    # sigFileName = kwargs["sigFileName"]
    # budgetFileName = kwargs["budgetFileName"]
    # fixed_ordering_col = kwargs["fixed_ordering_col"]
    # fixed_ordering_row = kwargs["fixed_ordering_row"]
    # check_point = kwargs["check_point"]
    # budget_update = kwargs["budget_update"]
    #
    # sigma_store, budget_store= partition_sampler(data=data, row_link=row_link, col_link=col_link,
    #                                              maxGibbsIteration_coord=maxGibbsIteration_coord, thin=thin,
    #                                              maxIteration=maxIteration, burnin=burnin, new_start=new_start,
    #                                              likelihoodFileName=likelihoodFileName,
    #                                              representationFileName=representationFileName,
    #                                              sigFileName=sigFileName, budgetFileName=budgetFileName,
    #                                              fixed_ordering_col=fixed_ordering_col,
    #                                              fixed_ordering_row=fixed_ordering_row,
    #                                              check_point=check_point, budget_update=budget_update)

    return sigma_store, budget_store


def good_tree_check(tree):
    lst = [tree.root]
    nrowcut,ncolcut,nleaf,nleafcut=0,0,0,0
    while len(lst) > 0:
        temp=[]
        for l in lst:
            if l.rowLB > l.rowUB or l.columnLB > l.columnUB:
                print('wrong boundaries')
                return False
            if l.isLeaf():
                nleaf += 1
                if l not in tree.leafBlockDic:
                    print('wrong node in leaf block dic')
                    return False
            else:
                if l.cutDir == 0:
                    if l.cutPos > l.rowUB or l.cutPos < l.rowLB:
                        print('wrong cutting pos')
                        return False
                    if any((l.leftChild.rowLB != l.rowLB, l.leftChild.rowUB != l.cutPos, l.leftChild.columnLB != l.columnLB, l.leftChild.columnUB != l.columnUB)):
                        print('wrong children block boundaries')
                        return False
                    if any((l.rightChild.rowLB != l.cutPos, l.rightChild.rowUB != l.rowUB, l.rightChild.columnLB != l.columnLB, l.rightChild.columnUB != l.columnUB)):
                        print('wrong children block boundaries')
                        return False
                    if l.rightChild.isLeaf() and l.leftChild.isLeaf():
                        nleafcut+=1
                        if l not in tree.leafCutDic:
                            print('wrong node in leaf cut dic')
                            return False
                    nrowcut+=1
                    if l.cutPos not in tree.rowCutDic or tree.rowCutDic[l.cutPos] != l:
                        print('wrong rowCutDic')
                        return False
                else:
                    if l.cutPos > l.columnUB or l.cutPos < l.columnLB:
                        print('wrong cutting pos')
                        return False
                    if any((l.leftChild.rowLB != l.rowLB, l.leftChild.rowUB != l.rowUB, l.leftChild.columnLB != l.columnLB, l.leftChild.columnUB != l.cutPos)):
                        print('wrong children block boundaries')
                        return False
                    if any((l.rightChild.rowLB != l.rowLB, l.rightChild.rowUB != l.rowUB, l.rightChild.columnLB != l.cutPos, l.rightChild.columnUB != l.columnUB)):
                        print('wrong children block boundaries')
                        return False
                    if l.rightChild.isLeaf() and l.leftChild.isLeaf():
                        nleafcut+=1
                        if l not in tree.leafCutDic:
                            print('wrong node in leaf cut dic')
                            return False
                    ncolcut+=1
                    if l.cutPos not in tree.columnCutDic or tree.columnCutDic[l.cutPos] != l:
                        print('wrong columnCutDic')
                        return False

                temp += [l.leftChild, l.rightChild]
        lst=temp
    if any((nrowcut!=len(tree.rowCutDic),ncolcut!=len(tree.columnCutDic),nleaf!=len(tree.leafBlockDic),nleafcut!=len(tree.leafCutDic))):
        print(nrowcut, ncolcut,nleaf, nleafcut)
        print('additional strange rowcut/colcut/leaf or leafcut')
        return False

    return True


def temper_one_sweep(data, sigma, tree, xi, eta, row_link, col_link, maxGibbsIteration_coord = 1, p_row=0.15, p_col=0.15,
                     maxIteration_step = 2, fixed_ordering_col=False, fixed_ordering_row=False,
                     budget_update=False, gamma=1., coord_update=True):
    # Do maxIteration number of sweeps
    split_accept = 0
    merge_accept = 0
    logLikelihood = 0
    # RJ MCMC step followed by a few round of Gibbs update on row/col coord
    for itr in range(0, maxIteration_step):
        # first do RJMCMC update on tree topology
        for n_rj in range((len(tree.leafBlockDic)//20)+1):  # do RJ MCMC, # of attempts prop to # of leaf nodes
            move = random.choice([0, 1])  # 0=split, 1=merge
            if move == 0:  # 0=add cut
                indicator = rjmcmc_split2(data, sigma, tree, xi, eta, row_link, col_link, gamma)
                split_accept += indicator
            else:  # 1=merge two leaf nodes
                indicator = rjmcmc_merge2(data, sigma, tree, xi, eta, row_link, col_link, gamma)
                merge_accept += indicator

            # randomly select a block, redo the cut
            leafCut = tree.getRandomLeafCut()
            if leafCut is not None:
                rjmcmc_redo_a_cut2(data, sigma, leafCut, xi, eta, tree, row_link, col_link, gamma)

        if coord_update:  # do we want to update the latent coords?
            if len(tree.leafBlockDic) > 80:
                p_col_prime = 0.5*p_col
                p_row_prime = 0.5*p_row
            else:
                p_col_prime = p_col
                p_row_prime = p_row

            if len(tree.leafBlockDic) < 100:  # do Gibbs if less than 100 blocks, MH otherwise
                [xi, eta, logLikelihood] = Gibbs_for_coord(sigma, data, xi, eta, tree, row_link, col_link,
                                                           maxGibbsIteration=maxGibbsIteration_coord,
                                                           isPreGibbsLikelihood=False,
                                                           fixed_ordering_col=fixed_ordering_col,
                                                           fixed_ordering_row=fixed_ordering_row,
                                                           p_row=p_row_prime, p_col=p_col_prime, gamma=gamma)  # only updating a portion of the coords
            else:  # switch to cheaper MH update if too many leaf nodes
                [xi, eta, logLikelihood] = MH_for_coord(sigma, data, xi, eta, tree, row_link, col_link,
                                                        maxMHIteration=maxGibbsIteration_coord,
                                                        isPreMHLikelihood=False,
                                                        fixed_ordering_col=fixed_ordering_col,
                                                        fixed_ordering_row=fixed_ordering_row,
                                                        p_row=p_row, p_col=p_col, gamma=gamma)  # only updating a portion of the coords
        # now update sigma
        sigma = sigma_MH_in_Gibbs(data, sigma, tree, row_link, col_link, jump_size=0.8, max_iter=4, gamma=gamma)

        # print('cost update')
        # now update cost and cutting position of some randomly chosen blocks
        ll = int(0.1*(len(tree.rowCutDic)+len(tree.columnCutDic)))+1
        for _ in range(min(3, ll)):
            MH_update_cutting_cost(tree, gamma=gamma)
            MH_update_cutting_position(tree, xi, eta, data, col_link, row_link, sigma, gamma=gamma)
            if budget_update:
                # now update budget
                budget_MH_in_Gibbs(tree, jump_size=0.9, gamma=gamma)
        # print('Current lkd: {}'.format(logLikelihood))
        logLikelihood = sum(lll.marginal_lkd for lll in tree.leafBlockDic.keys())
    return sigma, tree, xi, eta, logLikelihood, split_accept/maxIteration_step, merge_accept/maxIteration_step


def temper_one_sweep_warp(kwargs):
    sigma, tree, xi, eta, logLikelihood, spl, mer = temper_one_sweep(**kwargs)
    return sigma, tree, xi, eta, logLikelihood, spl, mer


def parallel_chains_init(data, gamma_lst, row_link, col_link, path, maxGibbsIteration_coord=1, budget=1.0, scale=True,
                         maxIteration_step=2, burnin=700, p_row=0.3, p_col=0.3, random_MP_init=True, max_init_cut=4,
                         new_start=False, fixed_ordering_col=False, fixed_ordering_row=False, budget_update=False,
                         candidate_row_coord=None, candidate_col_coord=None, coord_update=True):
    chain_lst = []
    if new_start:
        for _, gamma in enumerate(gamma_lst):
            tempered_chain = {}
            tempered_chain["data"] = data
            if scale:
                ratio = len(row_link)/len(col_link)
            else:
                ratio = 1
            box_length = np.round((2*ratio)/(1+ratio), 2)
            box_width = np.round(2/(1+ratio), 2)  # ensures that perimeter=4 while length/width=ratio

            tempered_chain["sigma"] = 1.0
            tempered_chain["tree"] = MondrianTree(budget, 0, box_length, 0, box_width, random_MP_init, max_cut=max_init_cut)  # initialize

            if candidate_row_coord is None or len(candidate_row_coord) != len(gamma_lst):
                tempered_chain["xi"] = [random.uniform(0, box_length) for _ in row_link]
            else:
                tempered_chain["xi"] = candidate_row_coord[_]
            if fixed_ordering_row:
                tempered_chain["xi"] = sorted(tempered_chain["xi"])

            if candidate_col_coord is None or len(candidate_col_coord) != len(gamma_lst):
                tempered_chain["eta"] = [random.uniform(0, box_width) for _ in col_link if len(data) > 0]
            else:
                tempered_chain["eta"] = candidate_col_coord[_]
            if fixed_ordering_col:
                tempered_chain["eta"] = sorted(tempered_chain["eta"])

            tempered_chain["gamma"] = 1.0  # for initialization, will switch later
            tempered_chain["maxIteration_step"] = burnin  # for initialization, will switch later
            tempered_chain["maxGibbsIteration_coord"] = maxGibbsIteration_coord
            tempered_chain["fixed_ordering_col"] = fixed_ordering_col
            tempered_chain["fixed_ordering_row"] = fixed_ordering_row

            tempered_chain["p_row"] = p_row
            tempered_chain["p_col"] = p_col
            tempered_chain["row_link"] = row_link
            tempered_chain["col_link"] = col_link
            tempered_chain["budget_update"] = budget_update
            tempered_chain["coord_update"] = coord_update

            # register points to the tree
            tempered_chain["tree"].cell_allocate(tempered_chain["xi"], tempered_chain["eta"])
            for l in tempered_chain["tree"].leafBlockDic.keys():
                l.marginal_lkd_block_level(data=tempered_chain["data"], xi=tempered_chain["xi"], eta=tempered_chain["eta"],
                                           p=2, marginal_lkd_func=marginal_lkd_MC,
                                           row_link=tempered_chain["row_link"], col_link=tempered_chain["col_link"],
                                           test=False, sigma=tempered_chain["sigma"])
            # add all things to the chain list
            chain_lst += [tempered_chain]

    else:
        chain_lst = []
        for gamma in gamma_lst:
            with open(path+'tempered_chain_{}.dictionary'.format(gamma), 'rb') as config_dictionary_file:
                checkpoint_dict = pickle.load(config_dictionary_file)
            tempered_chain = {}
            tempered_chain["data"] = data
            tempered_chain["sigma"] = checkpoint_dict['sigma']
            tempered_chain["tree"] = checkpoint_dict['tree']
            tempered_chain["xi"] = checkpoint_dict['xi']
            tempered_chain["eta"] = checkpoint_dict['eta']

            tempered_chain["gamma"] = gamma
            tempered_chain["maxIteration_step"] = maxIteration_step
            tempered_chain["maxGibbsIteration_coord"] = maxGibbsIteration_coord
            tempered_chain["fixed_ordering_col"] = fixed_ordering_col
            tempered_chain["fixed_ordering_row"] = fixed_ordering_row

            tempered_chain["p_row"] = p_row
            tempered_chain["p_col"] = p_col
            tempered_chain["row_link"] = row_link
            tempered_chain["col_link"] = col_link
            tempered_chain["budget_update"] = budget_update
            tempered_chain["coord_update"] = coord_update

            chain_lst += [tempered_chain]

    return chain_lst



def partition_parallel_tempering(data, gamma_lst, row_link, col_link, maxGibbsIteration_coord=1, thin=1, budget=1.0, scale=True,
                                 burnin1=700, burnin2=700, p_row=0.3, p_col=0.3, random_MP_init=True, maxIteration_step=2,
                                 max_init_cut=4, maxIteration = 1000, new_start=False, likelihoodFileName=None,
                                 representationFileName=None, sigFileName=None, budgetFileName=None,
                                 fixed_ordering_col=False, fixed_ordering_row=False,
                                 check_point='check_points', budget_update=False):
    itr_so_far = 0
    # Initialization
    # note that here budget hyper parameter of MP is set to be one, may want to put a prior on it
    if new_start:  # initialize len(gamma_lst) number of [tree, xi, eta, sigma]
        # alongside with kwargs needed for temper_one_sweep_warp
        # if duplicated check point path
        if os.path.exists('./'+check_point):
            return 'choose another path plz'
        # otherwise
        os.mkdir('./'+check_point)
        path = './'+check_point+'/'

        # writing simulation results into RJMCMC and Representation.txt
        if likelihoodFileName is None:
            likelihoodFileName = 'RJMCMC.txt'
        if representationFileName is None:
            representationFileName = 'Representation.txt'
        if sigFileName is None:
            sigFileName = 'Sigma_values.txt'
        if budgetFileName is None:
            budgetFileName = 'Budget_values.txt'
        fLogLikelihood = open(path+likelihoodFileName, 'w')
        fRepresentation = open(path+representationFileName, 'w')
        fSigma = open(path+sigFileName, 'w')
        fBudget = open(path+budgetFileName, 'w')

    else:  # start from the last check point
        path = './'+check_point+'/'
        if not os.path.exists('./'+check_point):
            return 'no such path?'
        with open(path+'tempered_chain_{}.dictionary'.format(gamma_lst[-1]), 'rb') as config_dictionary_file:
            checkpoint_dict = pickle.load(config_dictionary_file)

        itr_so_far = checkpoint_dict['itr']  # record it for convenience
        likelihoodFileName = checkpoint_dict['lkd_file']
        representationFileName = checkpoint_dict['representation_file']
        sigFileName = checkpoint_dict['sigma_file']
        budgetFileName = checkpoint_dict['budget_file']

        fLogLikelihood = open(path+likelihoodFileName, 'a')
        fRepresentation = open(path+representationFileName, 'a')
        fSigma = open(path+sigFileName, 'a')
        fBudget = open(path+budgetFileName, 'a')

    chain_lst = parallel_chains_init(data=data, gamma_lst=gamma_lst, row_link=row_link, col_link=col_link, path=path,
                                     maxGibbsIteration_coord=maxGibbsIteration_coord, budget=budget, scale=scale,
                                     maxIteration_step=maxIteration_step, burnin=burnin1, p_row=p_row,
                                     p_col=p_col, random_MP_init=random_MP_init, max_init_cut=max_init_cut,
                                     new_start=new_start, fixed_ordering_col=fixed_ordering_col,
                                     fixed_ordering_row=fixed_ordering_row, budget_update=budget_update,
                                     candidate_row_coord=None, candidate_col_coord=None, coord_update=True)


    startTime = time.time()
    pool = multiprocessing.Pool(processes=len(gamma_lst), initializer=np.random.seed)
    # first do the burn-in stage:
    if new_start:  # two-stage burn-in
        chain_result1 = pool.map(temper_one_sweep_warp, chain_lst)  # run burnin step 1 with random initialization,
        # goal is to find a sensible permutation
        candidate_row_coord = [_[2] for _ in chain_result1]
        candidate_col_coord = [_[3] for _ in chain_result1]
        if len(candidate_row_coord) != len(gamma_lst) or len(candidate_col_coord) != len(gamma_lst):
            exit('wrong coord setup in burnin 1')
        # re initialize, find the tree structure that goes well with the given sensible latent coordinates
        chain_lst = parallel_chains_init(data=data, gamma_lst=gamma_lst, row_link=row_link, col_link=col_link, path=path,
                                         maxGibbsIteration_coord=maxGibbsIteration_coord, budget=budget, scale=scale,
                                         maxIteration_step=maxIteration_step, burnin=burnin2, p_row=p_row,
                                         p_col=p_col, random_MP_init=random_MP_init, max_init_cut=max_init_cut,
                                         new_start=True, fixed_ordering_col=fixed_ordering_col,
                                         fixed_ordering_row=fixed_ordering_row, budget_update=budget_update,
                                         candidate_row_coord=candidate_row_coord,
                                         candidate_col_coord=candidate_col_coord, coord_update=False)  # fix the coord!

        chain_result2 = pool.map(temper_one_sweep_warp, chain_lst)  # run burnin step 2 with random initialization, but fixed coord

        burnin_lkd, burnin_split, burnin_merge = np.zeros(len(chain_result2)), np.zeros(len(chain_result2)), np.zeros(len(chain_result2))
        for i in range(len(chain_result2)):
            chain_lst[i]["sigma"], chain_lst[i]["tree"], chain_lst[i]["xi"], chain_lst[i]["eta"], \
            burnin_lkd[i], burnin_split[i], burnin_merge[i] = chain_result2[i]
            chain_lst[i]["maxIteration_step"] = maxIteration_step  # now switch to a much shorter interval for swapping
            chain_lst[i]["gamma"] = gamma_lst[i]  # the corresponding tempering parameter
            chain_lst[i]["coord_update"] = True  # we allow the coord to vary now
        print('#################################################################################################')
        print('Burn-in finished, lkd: {}, split rate: {}, merge rate: {}'.format(np.round(burnin_lkd, 4),
                                                                                 np.round(burnin_split, 4),
                                                                                 np.round(burnin_merge, 4)))
        print('Time {}'.format(time.time() - startTime))
        print('################################################################################################')
        print('Now the parallel tempering step')


    split_acc_rate, merge_acc_rate = np.zeros(len(gamma_lst)), np.zeros(len(gamma_lst))
    for itr in range(itr_so_far, itr_so_far+maxIteration):
        print(itr)
        # update each chain at different temp, in parallel
        chain_result = pool.map(temper_one_sweep_warp, chain_lst)
        # chain result has len = len(gamma_lst), each element has four components,
        # [0] sigma, [1] tree, [2] xi, [3] eta, [4] loglkd [5] split acc rate [6] merge acc rate

        # update chain_lst based on the results returned from chain_result
        lkd_vec, split_temp, merge_temp = np.zeros(len(chain_result)), np.zeros(len(chain_result)), np.zeros(len(chain_result))

        for i in range(len(chain_result)):
            chain_lst[i]["sigma"], chain_lst[i]["tree"], chain_lst[i]["xi"], chain_lst[i]["eta"], \
            lkd_vec[i], split_temp[i], merge_temp[i] = chain_result[i]
        split_acc_rate += split_temp
        merge_acc_rate += merge_temp
        print('Log likelihood for each chain', np.round(lkd_vec, 4))

        # now the swap move
        if itr % 2 == 0:
            start = 0
        else:
            start = 1

        for i in range(start, len(chain_result), 2):
            # pair wise swap, (0,1),(2,3),(4,5).... if start =0, (1,2),(3,4),(5,1),... if start = 1
            idx1 = i % len(chain_result)
            idx2 = (i+1) % len(chain_result)
            chain1 = chain_result[idx1]
            chain2 = chain_result[idx2]
            log_lkd1 = lkd_vec[idx1]
            log_lkd2 = lkd_vec[idx2]

            temper_para1 = gamma_lst[idx1]
            temper_para2 = gamma_lst[idx2]
            # compute the MH ratio, swap or not
            # note these are original prior and lkd, need to multiply by the tempering parameter
            # first, prior density of the MP
            log_prior1 = MP_prior(chain1[1])
            log_prior2 = MP_prior(chain2[1])
            # add prior density for sigma Gamma(3,50)
            log_prior1 += (3-1)*np.log(chain1[0]) - 50*chain1[0]
            log_prior2 += (3-1)*np.log(chain2[0]) - 50*chain2[0]
            # add prior density for xi,eta: Constant, HA!
            # add prior density for beta Gamma(3,2)
            log_prior1 += (3-1)*np.log(chain1[1].budget) - 2*chain1[1].budget
            log_prior2 += (3-1)*np.log(chain2[1].budget) - 2*chain2[1].budget
            log_MH_ratio = (log_prior2 + log_lkd2 - log_lkd1 - log_prior1)*(temper_para1 - temper_para2)

            if np.log(np.random.uniform()) < log_MH_ratio:
                # if we want to swap the states
                print('swap ({},{})'.format(idx1, idx2))
                temp_tree, temp_xi, temp_eta, temp_sigma, temp_lkd = chain_lst[idx2]["tree"], chain_lst[idx2]["xi"], \
                                                                     chain_lst[idx2]["eta"], chain_lst[idx2]["sigma"], lkd_vec[idx2]

                chain_lst[idx2]["tree"], chain_lst[idx2]["xi"], chain_lst[idx2]["eta"], chain_lst[idx2]["sigma"], lkd_vec[idx2]= \
                    chain_lst[idx1]["tree"], chain_lst[idx1]["xi"], chain_lst[idx1]["eta"], chain_lst[idx1]["sigma"], lkd_vec[idx1]

                chain_lst[idx1]["tree"], chain_lst[idx1]["xi"], chain_lst[idx1]["eta"], chain_lst[idx1]["sigma"], lkd_vec[idx1] = \
                    temp_tree, temp_xi, temp_eta, temp_sigma, temp_lkd

        # once we have updated the chain results, register all swaps to chain_lst
        # now store all the states
        if itr % thin == 0:
            print('Running Time: {}'.format(time.time() - startTime))
            print("Iteration: {}".format(itr))
            print('Swapped log likelihood for each chain', np.round(lkd_vec, 4))
            print('num of leaf: {}'.format(len(chain_lst[-1]["tree"].leafBlockDic)))
            print('sigma value: {}'.format(chain_lst[-1]["sigma"]))
            print('budget value: {}'.format(chain_lst[-1]["tree"].budget))
            fLogLikelihood.write('%s\n' % lkd_vec[-1])
            fRepresentation.write('Iteration: %d\n' % itr)
            leaf_string(chain_lst[-1]["tree"], fRepresentation)
            fRepresentation.write('%s\n' % chain_lst[-1]["xi"])
            fRepresentation.write('%s\n' % chain_lst[-1]["eta"])
            fSigma.write('%s\n' % chain_lst[-1]["sigma"])
            fBudget.write('%s\n' % chain_lst[-1]["tree"].budget)
            if itr % 5 == 0:
                fLogLikelihood.flush()
                fRepresentation.flush()
                fSigma.flush()
                fBudget.flush()

            # now save the check points
            for _, gamma in enumerate(gamma_lst):
                MCMC_dictionary = {"tree": chain_lst[_]["tree"], "xi": chain_lst[_]["xi"], "eta": chain_lst[_]["eta"],
                                   "sigma": chain_lst[_]["sigma"], "itr": itr+1, "lkd_file": likelihoodFileName,
                                   'representation_file': representationFileName, "sigma_file": sigFileName,
                                   'budget_file': budgetFileName}
                check_point_path = path+'tempered_chain_{}.dictionary'.format(gamma)
                with open(check_point_path, 'wb') as MCMC_checkpoint_file:
                    pickle.dump(MCMC_dictionary, MCMC_checkpoint_file)

    print('Total Running Time: {}'.format(time.time() - startTime))
    fLogLikelihood.close()
    fRepresentation.close()
    fSigma.close()
    fBudget.close()
    pool.close()
    pool.join()

    print('Split accept prob: {}, Merge accept prob: {}'.format(1.0*split_acc_rate/maxIteration,
                                                                1.0*merge_acc_rate/maxIteration))

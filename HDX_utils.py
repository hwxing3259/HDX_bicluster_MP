from HDX_power_posterior import *
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
import pandas
import os
from os.path import exists
from sklearn.metrics.cluster import adjusted_rand_score
import subprocess
import shutil
from itertools import product
from sklearn.impute import KNNImputer
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from scipy.special import logsumexp

def MCMC_processing(data, path, last_N, preprocessing=False, tempering_parameter=''):
    if tempering_parameter != '':
        tempering_parameter = '_' + tempering_parameter
    R=data.shape[0]
    C=data.shape[1]
    if preprocessing:
        if os.path.exists(path+'results{}'.format(str(tempering_parameter))):
            return 'choose another path plz'
        # otherwise
        os.mkdir(path+'results{}'.format(str(tempering_parameter)))

        representations = []
        Representation_path = path+'Representation{}.txt'.format(str(tempering_parameter))
        f = open(Representation_path, 'r')
        representations_sub = f.readlines()
        f.close()
        my_cut = len(representations_sub) - last_N*4
        representations += representations_sub[max(0,my_cut):]
        # summarize the partition structures
        unique_partition = {}
        unique_row_coord = {}
        unique_col_coord = {}
        unique_int_rep = {}
        appearance = {}
        config_dict = {}
        all_cls = np.zeros((int(len(representations)/4), R*C))
        idx = 0
        my_base_total_r = np.zeros((R, R))
        my_base_total_c = np.zeros((C, C))
        my_base_split = np.zeros((R, R, C))

        for __, q in enumerate(range(0, len(representations), 4)):
            if q+3 >= len(representations):
                break
            if q%10==0:
                print(q)
                print(representations[q][:-1])
            # idx = representations[q][:-1]
            # print(idx)
            configuration = representations[q+1][:-1]
            row_coord = eval(representations[q+2][:-1])
            col_coord = eval(representations[q+3][:-1])
            block_partition, pairs = leaf_string_decoder2(configuration, row_coord, col_coord, fig=False)

            block_partition_ij = np.zeros((R, C))
            for i, j in block_partition.items():
                if len(j) != 0:
                    for k in j:
                        block_partition_ij[k//C, k%C] = eval(i)

            all_cls[__] = block_partition_ij.ravel()

            if len(unique_partition) == 0:
                unique_partition[idx] = block_partition_ij
                unique_row_coord[idx] = row_coord
                unique_col_coord[idx] = col_coord
                unique_int_rep[idx] = block_partition
                appearance[idx] = 1
                config_dict[idx] = q
                idx += 1
            else:
                for id, co in unique_partition.items():
                    if adjusted_rand_score(block_partition_ij.ravel(), co.ravel()) == 1.0:
                        appearance[id] += 1
                        break
                else:
                    idx += 1
                    appearance[idx] = 1
                    unique_partition[idx] = block_partition_ij
                    unique_row_coord[idx] = row_coord
                    unique_col_coord[idx] = col_coord
                    unique_int_rep[idx] = block_partition
                    config_dict[idx] = q

            my_base = np.zeros((R, R))
            for i in range(R):
                for j in range(i+1):
                    group_label1 = [block_partition_ij[i, k] for k in range(C)]
                    group_label2 = [block_partition_ij[j, k] for k in range(C)]
                    my_base[i,j] = sum([x==y for (x,y) in zip(group_label1, group_label2)])/C
                    my_base[j,i] = my_base[i,j]*1.0
            my_base_total_r += my_base

            my_base = np.zeros((C, C))
            for i in range(C):
                for j in range(i+1):
                    group_label1 = [block_partition_ij[k, i] for k in range(R)]
                    group_label2 = [block_partition_ij[k, j] for k in range(R)]
                    my_base[i,j] = sum([x==y for (x,y) in zip(group_label1, group_label2)])/R
                    my_base[j,i] = my_base[i,j]*1.0
            my_base_total_c += my_base

            my_base = np.zeros((R, R, C))
            for i in range(R):
                for j in range(i+1):
                    group_label1 = [block_partition_ij[i, k] for k in range(C)]
                    group_label2 = [block_partition_ij[j, k] for k in range(C)]
                    my_base[i, j] = [x==y for (x,y) in zip(group_label1, group_label2)]
            my_base_split += my_base

        my_base_split /= my_base_split[0,0,0]
        my_base_total_r /= my_base_total_r[0, 0]
        my_base_total_c /= my_base_total_c[0, 0]
        np.savetxt(fname=path+'results{}/cls.txt'.format(str(tempering_parameter)), X=all_cls)
        np.savetxt(fname=path+'results{}/row_cooccur.txt'.format(str(tempering_parameter)), X=my_base_total_r)
        np.savetxt(fname=path+'results{}/col_cooccur.txt'.format(str(tempering_parameter)), X=my_base_total_c)
        np.save(path+'results{}/splitted_partition.npy'.format(str(tempering_parameter)), my_base_split)

        partition_cls = {'unique_partition': unique_partition, 'unique_row_coord': unique_row_coord,
                         'unique_col_coord': unique_col_coord, 'unique_int_rep': unique_int_rep,
                         'appearance': appearance, 'config_dict': config_dict}
        with open(path+'results{}/co_CI.dictionary'.format(str(tempering_parameter)), 'wb') as CI_file:
            pickle.dump(obj=partition_cls, file=CI_file)

        # find the point estimate of the co-clustering
        shutil.copy('./mcclust_helper.R', path)
        subprocess.call(['C:/Program Files/R/R-4.0.3/bin/Rscript', '--vanilla',
                         path+'mcclust_helper.R', path+'results{}/'.format(str(tempering_parameter)), str(C)], shell=True)
        point_co_cluster = np.loadtxt(path+'results{}/mode_cls.txt'.format(str(tempering_parameter))).astype(int)
    else:
        if not os.path.exists(path+'results{}'.format(str(tempering_parameter))):
            return 'o boi no such dir'
        with open(path+'results{}/co_CI.dictionary'.format(str(tempering_parameter)), 'rb') as CI_file:
            partition_cls = pickle.load(file=CI_file)
        unique_partition = partition_cls['unique_partition']
        unique_row_coord = partition_cls['unique_row_coord']
        unique_col_coord = partition_cls['unique_col_coord']
        unique_int_rep = partition_cls['unique_int_rep']
        appearance = partition_cls['appearance']
        config_dict = partition_cls['config_dict']

        my_base_split = np.load(path+'results{}/splitted_partition.npy'.format(str(tempering_parameter)))
        my_base_total_c = np.loadtxt(path+'results{}/col_cooccur.txt'.format(str(tempering_parameter)))
        my_base_total_r = np.loadtxt(path+'results{}/row_cooccur.txt'.format(str(tempering_parameter)))

        point_co_cluster = np.loadtxt(path+'results{}/mode_cls.txt'.format(str(tempering_parameter))).astype(int)

    point_idx = None
    for _ in unique_partition.keys():
        if np.all(unique_partition[_].astype(int) == point_co_cluster):
            point_idx = _
            break

    config_idx = config_dict[point_idx]

    # find the 95% CI of the co-clusters
    sorted_co = dict(sorted(appearance.items(), key=lambda x: x[1], )[::-1])
    co_CI_id = []
    acc = 0
    for id, num in sorted_co.items():
        if acc/sum(appearance.values()) > 0.95:
            break
        else:
            co_CI_id += [id]
            acc += num

    co_CI = {_: unique_partition[_] for _ in co_CI_id}
    co_CI_row_coord = {_: unique_row_coord[_] for _ in co_CI_id}
    co_CI_col_coord = {_: unique_col_coord[_] for _ in co_CI_id}
    co_CI_int_rep = {_: unique_int_rep[_] for _ in co_CI_id}
    # is point_idx in co_CI?
    print(point_idx in co_CI)
    # Plot some of the partitioned trajectories
    block_partition_coord = {}
    for i in unique_int_rep[point_idx].keys():
        if len(unique_int_rep[point_idx][i])>0:
            block_partition_coord[i] = [(_//C,_%C) for _ in unique_int_rep[point_idx][i]]
    block_partition_coord = {i: j for i,j in enumerate(sorted(block_partition_coord.values(), key=lambda x: x[0][0]))}

    point_estimate_state = {"block_partition_coord": block_partition_coord,
                            "membership_matrix": unique_partition[point_idx],
                            "col_coord": unique_col_coord[point_idx],
                            "row_coord": unique_row_coord[point_idx],
                            "col_permute": np.argsort(unique_col_coord[point_idx]),
                            "row_permute": np.argsort(unique_row_coord[point_idx])}

    return point_estimate_state, co_CI, co_CI_row_coord, co_CI_col_coord, co_CI_int_rep, appearance, my_base_split, \
           my_base_total_c, my_base_total_r, point_idx, unique_partition, unique_row_coord, unique_col_coord, unique_int_rep



def posterior_predictive_alt(partitioned_data, n_iter=9000,  thin=3):
    ans_mean = np.zeros((n_iter//thin, len(partitioned_data), 2))
    ans_predictive = np.zeros((n_iter//thin, len(partitioned_data), 2))
    para_result = np.zeros((n_iter//thin, len(partitioned_data), 2))
    sigma = np.zeros(n_iter//thin)
    lkd = np.zeros((n_iter//thin, len(partitioned_data)))
    old_sigma = 1.0
    old_para = {i:[0.0,0.0] for i in partitioned_data.keys()}
    old_post = {i: joint_log_lkd_repar_scaled(partitioned_data[i], old_sigma, old_para[i][0], old_para[i][1])
                for i in partitioned_data.keys()}
    # gibbs sampler on a_k,b_k and sigma
    for n in range(n_iter):
        for _, para in old_para.items():
            # ua=random.uniform(0.8,1.25)
            # proposed_para_a = para[0]*ua
            proposed_para_a = para[0] + 0.02*np.random.randn(1)[0]
            proposed_post = joint_log_lkd_repar_scaled(partitioned_data[_], old_sigma, proposed_para_a, old_para[_][1])
            if (proposed_post-old_post[_]) > np.log(np.random.uniform()):  #
                old_para[_][0] = proposed_para_a
                old_post[_] = proposed_post

            # ub=random.uniform(0.8,1.25)
            # proposed_para_b = para[1]*ub
            proposed_para_b = para[1] + 0.02*np.random.randn(1)[0]
            proposed_post = joint_log_lkd_repar_scaled(partitioned_data[_], old_sigma, old_para[_][0], proposed_para_b)
            if (proposed_post-old_post[_]) > np.log(np.random.uniform()): #
                old_para[_][1] = proposed_para_b
                old_post[_] = proposed_post

        us = random.uniform(0.8,1.25)
        proposed_sigma = old_sigma*us
        # proposed_sigma = old_sigma + 0.05*np.random.randn(1)[0]
        if proposed_sigma>0:
            new_post = {i: joint_log_lkd_repar_scaled(partitioned_data[i], proposed_sigma, old_para[i][0], old_para[i][1])
                        for i in partitioned_data.keys()}
            log_prior_ratio = (3-1)*(np.log(proposed_sigma) - np.log(old_sigma)) - 50*(proposed_sigma - old_sigma)
            # work out prior ratio Gamma(3,50)
            log_post_ratio = sum(new_post.values()) - sum(old_post.values())
            if (log_post_ratio+log_prior_ratio-np.log(us))>np.log(np.random.uniform()):  # -np.log(u)
                old_sigma = proposed_sigma
                old_post = new_post

        # record the mean funcs at two time steps
        if n % thin==0:
            ans_mean[n//thin] = np.array([np.exp(para[0])*(1-np.exp(-np.exp(para[1])*np.array([0.5, 5.])))
                                          for para in old_para.values()])
            ans_predictive[n//thin] = ans_mean[n//thin] + old_sigma*np.random.randn(len(partitioned_data), 2)
            lkd[n//thin] = np.array(list(old_post.values())).ravel()
            para_result[n//thin] = np.array(list(old_para.values()))
            sigma[n//thin] = old_sigma

    return ans_mean[200:], ans_predictive[200:], lkd[200:], para_result[200:], sigma[200:]


def posterior_predictive(partitioned_data, n_iter=9000,  thin=3):  # this is inferring the a,b on original scale
    ans_mean = np.zeros((n_iter//thin, len(partitioned_data), 2))
    ans_predictive = np.zeros((n_iter//thin, len(partitioned_data), 2))
    para_result = np.zeros((n_iter//thin, len(partitioned_data), 2))
    sigma = np.zeros(n_iter//thin)
    lkd = np.zeros((n_iter//thin, len(partitioned_data)))
    old_sigma = 1.0
    old_para = {i:[0.1,0.1] for i in partitioned_data.keys()}
    old_post = {i: joint_log_lkd(partitioned_data[i], old_sigma, old_para[i][0], old_para[i][1])
                for i in partitioned_data.keys()}
    # gibbs sampler on a_k,b_k and sigma
    for n in range(n_iter):
        for _, para in old_para.items():
            ua=random.uniform(0.8,1.25)
            proposed_para_a = para[0]*ua
            # proposed_para_a = para[0] + 0.02*np.random.randn(1)[0]
            proposed_post = joint_log_lkd(partitioned_data[_], old_sigma, proposed_para_a, old_para[_][1])
            if (proposed_post-old_post[_]-np.log(ua)) > np.log(np.random.uniform()):  #
                old_para[_][0] = proposed_para_a
                old_post[_] = proposed_post

            ub = random.uniform(0.8,1.25)
            proposed_para_b = para[1]*ub
            # proposed_para_b = para[1] + 0.02*np.random.randn(1)[0]
            proposed_post = joint_log_lkd(partitioned_data[_], old_sigma, old_para[_][0], proposed_para_b)
            if (proposed_post-old_post[_]-np.log(ub)) > np.log(np.random.uniform()): #
                old_para[_][1] = proposed_para_b
                old_post[_] = proposed_post

        us = random.uniform(0.8,1.25)
        proposed_sigma = old_sigma*us
        # proposed_sigma = old_sigma + 0.05*np.random.randn(1)[0]
        if proposed_sigma>0:
            new_post = {i: joint_log_lkd(partitioned_data[i], proposed_sigma, old_para[i][0], old_para[i][1])
                        for i in partitioned_data.keys()}
            log_prior_ratio = (3-1)*(np.log(proposed_sigma) - np.log(old_sigma)) - 50*(proposed_sigma - old_sigma)
            # work out prior ratio Gamma(3,50)
            log_post_ratio = sum(new_post.values()) - sum(old_post.values())
            if (log_post_ratio+log_prior_ratio-np.log(us))>np.log(np.random.uniform()):  # -np.log(u)
                old_sigma = proposed_sigma
                old_post = new_post

        # record the mean funcs at two time steps
        if n % thin==0:
            ans_mean[n//thin] = np.array([para[0]*(1-np.exp(-para[1]*np.array([30,300])))
                                          for para in old_para.values()])
            ans_predictive[n//thin] = ans_mean[n//thin] + old_sigma*np.random.randn(len(partitioned_data), 2)
            lkd[n//thin] = np.array(list(old_post.values())).ravel()
            para_result[n//thin] = np.array(list(old_para.values()))
            sigma[n//thin] = old_sigma

    return ans_mean[200:], ans_predictive[200:], lkd[200:], para_result[200:], sigma[200:]


# conditioned on a given partition, what is the probability that
# the a_k,b_k of the cluster that (i,j) is in is greater then the parameter of the cluster that (k,l) is in?

def comparing_parameters(pair1, pair2, samples, co_cluster):
    id1 = int(co_cluster[pair1[0], pair1[1]])
    id2 = int(co_cluster[pair2[0], pair2[1]])
    samples_id1 = samples[:, id1, :]
    samples_id2 = samples[:, id2, :]
    prob1 = np.mean(samples_id1[:, 0] > samples_id2[:, 0])
    prob2 = np.mean(samples_id1[:, 1] > samples_id2[:, 1])
    prob3 = np.mean((samples_id1[:, 1] > samples_id2[:, 1])*(samples_id1[:, 0] > samples_id2[:, 0]))
    return prob1, prob2, prob3


def proc_vs_deproc(data, path, max_truncation=30, tempering_parameter=''):
    if tempering_parameter != '':
        tempering_parameter = '_'+str(tempering_parameter)
    if not os.path.exists(path+'results{}'.format(str(tempering_parameter))):
        return 'o boi no such dir'
    if os.path.exists(path+'results{}/averaged_protection.txt'.format(str(tempering_parameter))) and \
            os.path.exists(path+'results{}/averaged_deprotection.txt'.format(str(tempering_parameter))):
        return np.loadtxt(path+'results{}/averaged_protection.txt'.format(str(tempering_parameter))), \
               np.loadtxt(path+'results{}/averaged_deprotection.txt'.format(str(tempering_parameter)))

    with open(path+'results{}/co_CI.dictionary'.format(str(tempering_parameter)), 'rb') as CI_file:
        partition_cls = pickle.load(file=CI_file)
    unique_partition = partition_cls['unique_partition']

    unique_int_rep = partition_cls['unique_int_rep']
    appearance = partition_cls['appearance']
    config_dict = partition_cls['config_dict']

    point_co_cluster = np.loadtxt(path+'results{}/mode_cls.txt'.format(str(tempering_parameter))).astype(int)


    # find the 95% CI of the co-clusters
    sorted_co = dict(sorted(appearance.items(), key=lambda x: x[1], )[::-1])
    co_CI_id = []
    acc = 0
    for id, num in sorted_co.items():
        if acc/sum(appearance.values()) > 0.95:
            break
        else:
            co_CI_id += [id]
            acc += num

    co_CI = {_: unique_partition[_] for _ in co_CI_id}

    R=data.shape[0]
    C=data.shape[1]
    avg_protection = np.zeros((R,C))
    avg_deprotection = np.zeros((R,C))

    for ii in list(co_CI.keys())[:max_truncation]:
        print(ii)
        cc = unique_int_rep[ii]
        block_partition_coord = {}
        for i in cc.keys():
            if len(cc[i])>0:
                block_partition_coord[i] = [(_//C,_%C) for _ in cc[i]]
        block_partition_coord = {i: j for i,j in enumerate(sorted(block_partition_coord.values(), key=lambda x: x[0][0]))}

        partitioned_points = {}
        for i, j in block_partition_coord.items():
            if j != []:
                temp = []
                [temp.append(data[_[0]][_[1]]) for _ in j]
                partitioned_points[i] = np.array(temp)

        co_clustering_mtrx = np.zeros((R, C))
        for i, coord in block_partition_coord.items():
            for _ in coord:
                co_clustering_mtrx[_[0]][_[1]] = i

        ans_mean, ans_predictive, lkd, para, sig = posterior_predictive(partitioned_points, n_iter=1500)
        prob_mtrx = np.zeros((R, C))
        for r in range(R):
            for c in range(C):
                prob_mtrx[r, c]=comparing_parameters((r, c), (r, 0), ans_mean, co_clustering_mtrx)[-1]
        avg_deprotection += prob_mtrx*appearance[ii]

        prob_mtrx = np.zeros((R, C))
        for r in range(R):
            for c in range(C):
                prob_mtrx[r, c]=comparing_parameters((r, 0), (r, c), ans_mean, co_clustering_mtrx)[-1]
        avg_protection += prob_mtrx*appearance[ii]

    avg_deprotection /= np.max(avg_deprotection)
    avg_protection /= np.max(avg_protection)

    np.savetxt(path+'results{}/averaged_protection.txt'.format(str(tempering_parameter)), X=avg_protection)
    np.savetxt(path+'results{}/averaged_deprotection.txt'.format(str(tempering_parameter)), X=avg_deprotection)

    return avg_protection, avg_deprotection



def HC_HDX(data, n_row_cluster, n_column_cluster):
    R = data.shape[0]
    C = data.shape[1]
    rowName = ['Peptide {}'.format(r) for r in range(R)]
    colName = ['Treatment {}'.format(c) for c in range(C)]

    row_hier_data_raw = np.c_[data[:,:,0],data[:,:,1]]  # R*2C
    col_hier_data_raw = np.r_[data[:,:,0],data[:,:,1]].T  # C*2R
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    row_hier_data = imputer.fit_transform(row_hier_data_raw)
    col_hier_data = imputer.fit_transform(col_hier_data_raw)

    df_col = pandas.DataFrame(col_hier_data, columns=rowName + rowName, index=colName)
    df_row = pandas.DataFrame(row_hier_data, columns=colName+colName, index=rowName)

    # plt.figure(figsize=(20, 14))
    # plt.title("Peptide Dendograms")
    # dend = shc.dendrogram(shc.linkage(df_row, method='ward'), labels=rowName)
    #
    # plt.figure(figsize=(20, 14))
    # plt.title("Treatment Dendograms")
    # dend = shc.dendrogram(shc.linkage(df_col, method='ward'), labels=colName)

    row_cluster = AgglomerativeClustering(n_clusters=n_row_cluster, affinity='euclidean', linkage='ward')
    row_label = row_cluster.fit_predict(df_row)
    row_partition = {}
    for i in range(len(row_label)):
        if row_label[i] in row_partition:
            row_partition[row_label[i]] += [i]
        else:
            row_partition[row_label[i]] = [i]

    column_cluster = AgglomerativeClustering(n_clusters=n_column_cluster, affinity='euclidean', linkage='ward')
    column_label = column_cluster.fit_predict(df_col)
    column_partition = {}
    for i in range(len(column_label)):
        if column_label[i] in column_partition:
            column_partition[column_label[i]] += [i]
        else:
            column_partition[column_label[i]] = [i]


    block_partition_coord = {}
    idx = 0
    for i in row_partition.values():
        for j in column_partition.values():
            block_partition_coord[idx] = list(product(i, j))
            idx+=1


    partitioned_points = {}
    for i, j in block_partition_coord.items():
        if j != []:
            temp = []
            [temp.append(data[_[0]][_[1]]) for _ in j]
            partitioned_points[i] = np.array(temp)

    co_clustering_mtrx = np.zeros((R, C))
    for i, coord in block_partition_coord.items():
        for _ in coord:
            co_clustering_mtrx[_[0]][_[1]] = i

    avg_protection_HC = np.zeros((R,C))
    avg_deprotection_HC = np.zeros((R,C))
    ans_mean, ans_predictive, lkd, para, sig = posterior_predictive(partitioned_points, n_iter=1500)
    prob_mtrx = np.zeros((R, C))
    for r in range(R):
        for c in range(C):
            prob_mtrx[r, c]=comparing_parameters((r, c), (r, 0), ans_mean, co_clustering_mtrx)[-1]
    avg_deprotection_HC += prob_mtrx

    prob_mtrx = np.zeros((R, C))
    for r in range(R):
        for c in range(C):
            prob_mtrx[r, c]=comparing_parameters((r, 0), (r, c), ans_mean, co_clustering_mtrx)[-1]
    avg_protection_HC += prob_mtrx

    return avg_protection_HC, avg_deprotection_HC


def HDX_WAIC(data, path, last_N=None, preprocessing=None, tempering_parameter=''):
    R = data.shape[0]
    C = data.shape[1]
    if tempering_parameter != '':
        tempering_parameter = '_'+str(tempering_parameter)
    if not os.path.exists(path+'results{}'.format(str(tempering_parameter))):
        return 'o boi no such dir'

    with open(path+'results{}/co_CI.dictionary'.format(str(tempering_parameter)), 'rb') as CI_file:
        partition_cls = pickle.load(file=CI_file)
    unique_int_rep = partition_cls['unique_int_rep']
    appearance = partition_cls['appearance']

    log_lkd_array = np.zeros((1, data.size))

    for idx, cc in unique_int_rep.items():
        if idx > 100:
            break
        print(idx, idx/len(unique_int_rep))
        block_partition_coord = {}
        for i in cc.keys():
            if len(cc[i])>0:
                block_partition_coord[i] = [(_//C,_%C) for _ in cc[i]]
        block_partition_coord = {i: j for i,j in enumerate(sorted(block_partition_coord.values(), key=lambda x: x[0][0]))}

        partitioned_points = {}
        for i, j in block_partition_coord.items():
            if j != []:
                temp = []
                [temp.append(data[_[0]][_[1]]) for _ in j]
                partitioned_points[i] = np.array(temp)

        co_clustering_mtrx = np.zeros((R, C))
        for i, coord in block_partition_coord.items():
            for _ in coord:
                co_clustering_mtrx[_[0]][_[1]] = i

        for ___ in range(appearance[idx]):
            ans_mean, ans_predictive, lkd, para, sig = posterior_predictive(partitioned_points, n_iter=500, thin=2)
            my_sig = sig[-1]
            my_means = ans_mean[-1]
            lkd_mtrx = np.zeros_like(data)
            for r in range(R):
                for c in range(C):
                    my_id = int(co_clustering_mtrx[r, c])  # r,c in group i means its mean is the ith row of my_means
                    lkd_mtrx[r,c] = -((data[r,c] - my_means[my_id])**2)/(2*my_sig**2) - 0.5*np.log(2*np.pi*my_sig**2)
            log_lkd_array = np.concatenate((log_lkd_array, lkd_mtrx.reshape((1,-1))))

    log_lkd_array = log_lkd_array[1:]

    np.savetxt(fname=path+'results{}/WAIC_raw.txt'.format(str(tempering_parameter)), X=log_lkd_array)
    idxx = [True]*data.size
    for _ in range(data.size):
        if np.any(np.isnan(log_lkd_array[:,_])):
            idxx[_] = False
    log_lkd_array = log_lkd_array[:,idxx]
    lpd = np.sum(logsumexp(log_lkd_array, axis=0)) - log_lkd_array.shape[1]*np.log(log_lkd_array.shape[0])
    pwaic = np.sum(np.var(log_lkd_array, axis=0))

    # or
    # lpd = np.nansum(logsumexp(log_lkd_array, axis=0) - np.log(log_lkd_array.shape[0]))
    # pwaic = np.nansum(np.var(log_lkd_array, axis=0))

    return log_lkd_array, lpd-pwaic


def HDX_process_warp(kwargs):
    point_estimate_state, co_CI, co_CI_row_coord, \
    co_CI_col_coord, co_CI_int_rep, appearance, my_base_split, \
    my_base_total_c, my_base_total_r, point_idx, unique_partition, \
    unique_row_coord, unique_col_coord, unique_int_rep = MCMC_processing(**kwargs)

    _, bbb = HDX_WAIC(**kwargs)

    return bbb




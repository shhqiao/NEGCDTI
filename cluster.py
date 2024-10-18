import numpy as np
import copy


def Cluster(sim,th,js,k):
    SR_old = np.where(sim >= th, 1, 0)
    SR_old = SR_old.astype(float)
    SR_d = np.sum(SR_old, axis=1)
    SR_D = np.diag(SR_d)
    SR_avg_D = np.mean(SR_d)
    SR_std = np.std(SR_d)
    threshold = SR_avg_D - (k * SR_std)
    indices = []
    for i in range(SR_D.shape[0]):
        if SR_D[i, i] > threshold:
            indices.append(i)
    SR_core = SR_old[indices]
    SR_temp = np.sum(SR_core, axis=0)
    ind = np.where(SR_temp > 1)[0].tolist()
    sim_core = sim[indices]
    max_indices = np.argmax(sim_core.T[ind], axis=1)
    cluster = []
    i = 0
    while i < len(indices):
        cluster1 = []
        neigh = np.where(SR_old[indices[i]] == 1)[0]
        cluster1.append(indices[i])
        for j in neigh:
            if j not in indices:
                if SR_temp[j] == 1:
                    cluster1.append(j)
                elif i == max_indices[ind.index(j)]:
                    cluster1.append(j)
        cluster.append(cluster1)
        i = i + 1
    unkown = []
    j = 0
    flag = 0
    while j < len(SR_old):
        for arr in cluster:
            if j in arr:
                flag = 1
        if flag == 0:
            unkown.append(j)
        flag = 0
        j = j + 1
    n = 0
    SR_j = SR_old.copy()
    qc_temp = []
    while n < js:
        SR_j = SR_j @ SR_old
        np.fill_diagonal(SR_j, 0)
        SR_j_copy = SR_j
        for i in range(len(SR_j_copy)):
            for j in range(len(SR_j_copy[i])):
                if SR_j_copy[i][j] > 1:
                    SR_j_copy[i][j] = 1
        SR_j_clu = SR_j_copy[indices]
        SR_j_temp = np.sum(SR_j_clu, axis=0)
        for k in unkown:
            if SR_j_temp[k] > 1:
                unkown_neigh = np.where(SR_old[k] == 1)[0]
                result = [item for sublist in cluster for item in sublist]
                result = np.array(result)
                intersection = np.intersect1d(unkown_neigh, result)
                col = sim[:, k]
                col_new = col[intersection]
                max_ind = np.argmax(col_new)
                select_nei = intersection[max_ind]

                for i, sublist in enumerate(cluster):
                    if select_nei in sublist:
                        row_index = i
                        cluster[row_index].append(k)
                        qc_temp.append(k)
            else:
                for g in SR_j_clu:
                    if g[k] == 1 and SR_j_temp[k] == 1:
                        col = SR_j_clu[:, k]
                        row_indices = np.where(col == 1)[0]
                        cluster[row_indices[0]].append(k)
                        qc_temp.append(k)
        new_unkown = [x for x in unkown if x not in qc_temp]
        unkown = new_unkown
        qc_temp = []
        n = n + 1

    if len(unkown) != 0:
        for k in unkown:
            col = sim[:, k]
            col_new = col[indices]
            max_ind = np.argmax(col_new)
            select_nei = indices[max_ind]
            for i, sublist in enumerate(cluster):
                if select_nei in sublist:
                    row_index = i
                    cluster[row_index].append(k)
                    qc_temp.append(k)
        new_unkown = [x for x in unkown if x not in qc_temp]
    l = sim.shape[0]+10
    new = copy.deepcopy(cluster)

    for sub in range(len(cluster)):
        if len(cluster[sub]) == 1:
            indices_f = indices.copy()
            only_core = cluster[sub][0]
            indices_f.remove(only_core)
            col = sim[:, only_core]
            col_new = col[indices_f]
            max_ind = np.argmax(col_new)
            select_nei = indices_f[max_ind]

            for y, sublt in enumerate(new):
                if select_nei in sublt:
                    c_ind = sublt
            if len(c_ind)==1:
                for i, sublist in enumerate(cluster):
                    if select_nei in sublist:
                        row_index = i
                        cluster[sub] = [l]
                        cluster[row_index].append(only_core)
                        new[sub] = [l]
                        new[row_index].append(only_core)

            elif len(c_ind)!=1:
                for i, sublist in enumerate(cluster):
                    if select_nei in sublist:
                        row_index = i
                        cluster[sub] = [l]
                        cluster[row_index].append(only_core)
                        new[sub] = [l]
                        new[row_index].append(only_core)
    new_cluster = [lst for lst in cluster if len(lst) > 1]
    new_indices = [sublist[0] for sublist in new_cluster]

    return new_cluster, new_indices


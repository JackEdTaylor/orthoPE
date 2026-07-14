import numpy as np
import scipy as sp
import ot

def get_jaccard(a, b):
    intersect = np.sum(np.minimum(a, b)) # area of intersection of A and B (element-wise minimum)
    area_a = np.sum(a)
    area_b = np.sum(b)
    union = (area_a + area_b - intersect)
    return 1 - intersect / union

def get_average_euclidean(a, b, squared_euclidean=False):
    # coordinates of mass
    xa = np.transpose(np.array(np.where(a!=0)))
    xb = np.transpose(np.array(np.where(b!=0)))
    
    if squared_euclidean:
        # squared Euclidean distance kernels for all pairs of coordinates between the arrays
        M = sp.spatial.distance.cdist(xa, xb) ** 2
    else:
        # Euclidean distance kernels for all pairs of coordinates between the arrays
        M = sp.spatial.distance.cdist(xa, xb)

    return np.mean(M)

def get_fgw(s, t, alpha_tradeoff=0.5, w_metric='Euclidean', gw_metric='Euclidean', scale_distance=True):
    # higher alpha puts more weight on Gromov-Wasserstein (structure), and less on Wasserstein (features)

    xs = np.transpose(np.array(np.where(s!=0)))
    xt = np.transpose(np.array(np.where(t!=0)))

    M = ot.dist(xs, xt, metric=w_metric)

    # rescale the between-distribution distance matrix(?)
    if scale_distance:
        M /= M.max()

    # distance kernels
    C1 = ot.dist(xs, metric=gw_metric)
    C2 = ot.dist(xt, metric=gw_metric)

    # rescale the within-distribution distance kernels(?)
    if scale_distance:
        C1 /= C1.max()
        C2 /= C2.max()

    s_hist = s.flatten()[s.flatten()!=0]
    t_hist = t.flatten()[t.flatten()!=0]

    # proportion-scale
    s_hist /= s_hist.sum()
    t_hist /= t_hist.sum()

    # get cost
    cost = ot.fused_gromov_wasserstein2(M, C1, C2, p=s_hist, q=t_hist, loss_fun='square_loss', symmetric=None, alpha=alpha_tradeoff, armijo=False, G0=None, max_iter=1e12, tol_rel=1e-09, tol_abs=1e-09)

    return cost

def get_gw(s, t, metric='Euclidean', scale_distance=True):
    # coordinates of mass
    xs = np.transpose(np.array(np.where(s!=0)))
    xt = np.transpose(np.array(np.where(t!=0)))

    # distance kernels
    C1 = ot.dist(xs, metric=metric)
    C2 = ot.dist(xt, metric=metric)

    # rescale the within-distribution distance kernels
    if scale_distance:
        C1 /= C1.max()
        C2 /= C2.max()

    s_hist = s.flatten()[s.flatten()!=0]
    t_hist = t.flatten()[t.flatten()!=0]

    # proportion-scale
    s_hist /= s_hist.sum()
    t_hist /= t_hist.sum()

    # get cost
    cost = ot.gromov.gromov_wasserstein2(C1, C2, p=s_hist, q=t_hist, loss_fun='square_loss', symmetric=None, armijo=False, G0=None, max_iter=1e12, tol=1e-09)

    return cost

def get_w(s, t, metric='Euclidean', scale_distance=False):

    xs = np.transpose(np.array(np.where(s!=0)))
    xt = np.transpose(np.array(np.where(t!=0)))

    M = ot.dist(xs, xt, metric=metric)

    # rescale the between-distribution distance matrix(?)
    if scale_distance:
        M /= M.max()

    s_hist = s.flatten()[s.flatten()!=0]
    t_hist = t.flatten()[t.flatten()!=0]

    # proportion-scale
    s_hist /= s_hist.sum()
    t_hist /= t_hist.sum()

    # get cost
    cost = ot.emd2(a=s_hist, b=t_hist, M=M)

    return cost

def get_fgw_from_hists(xs, xt, s_hist, t_hist, alpha_tradeoff=0.5, w_metric='Euclidean', gw_metric='Euclidean', scale_distance=True):
    # higher alpha puts more weight on Gromov-Wasserstein (structure), and less on Wasserstein (features)

    # between-distribution distances
    M = ot.dist(xs, xt, metric=w_metric)

    # rescale the between-distribution distance matrix(?)
    if scale_distance:
        M /= M.max()

    # within-distribution distance kernels
    C1 = ot.dist(xs, metric=gw_metric)
    C2 = ot.dist(xt, metric=gw_metric)

    # rescale the within-distribution distance kernels(?)
    if scale_distance:
        C1 /= C1.max()
        C2 /= C2.max()

    # proportion-scale mass
    s_hist /= s_hist.sum()
    t_hist /= t_hist.sum()

    # get cost
    cost = ot.fused_gromov_wasserstein2(M, C1, C2, p=s_hist, q=t_hist, loss_fun='square_loss', symmetric=None, alpha=alpha_tradeoff, armijo=False, G0=None, max_iter=1e12, tol_rel=1e-09, tol_abs=1e-09)

    return cost

# wrapper for get_w, get_gw, and get_fgw which functions on list of pairs (can be useful for parallelisation on chunks)
def get_ot_cost_list(s_t_list, fun=get_fgw, **kwargs):
    # s_t_list should be a list of pairs, of form [[s_1, t_1], [s_2, t_2], ...[s_i, t_i]]
    return [fun(a, b, **kwargs) for a, b in s_t_list]

# wrapper for get_fgw_from_hists which takes indices as inputs for a list of coordinates and a list of histograms (can be useful for parallelisation on chunks)
def get_fgw_cost_list2(pairs_idx, coords, hists, **kwargs):
    # idx should be indices for coords and hists for the pairs being compared, of form [[i1, j1], [i2, j2], ...]
    # coords should be a list of coords like xs and xt, of form [x_1, x_2, ...]
    # hists should be a list of histigrams of mass like s_hist and t_hist, of form [hist_1, hist_2, ...]
    return [get_fgw_from_hists(xs=coords[i], xt=coords[j], s_hist=hists[i], t_hist=hists[j], **kwargs) for i, j in pairs_idx]

def get_w_barycentre(arrs_3d, debias=True, **kwargs):
    # arrs_3d should be a N x w x h array

    # proportion-scale
    arrs_3d /= arrs_3d.sum(axis=(1, 2), keepdims=True)

    if debias:
        bc = ot.bregman.convolutional_barycenter2d_debiased(arrs_3d, **kwargs)
    else:
        bc = ot.bregman.convolutional_barycenter2d(arrs_3d, **kwargs)

    return bc

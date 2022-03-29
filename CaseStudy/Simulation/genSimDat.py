import numpy as np
from numpy.random.mtrand import multivariate_normal as mvr
from numpy import linalg as la
import matplotlib.pyplot as plt
def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    def isPD(B):
        try:
            _ = la.cholesky(B)
            return True
        except la.LinAlgError:
            return False

    B = (A + A.T) / 2
    _, s, V = la.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    return A3

def gendat_single(n_genes,n_samples,dc,de):
    """Single patten generates"""
    cov_mat = np.identity(n_genes)
    cov_mat[np.tril_indices_from(cov_mat,k=-1)] = dc
    cov_mat *= np.random.choice([-1,1],size=cov_mat.shape)
    cov_mat = cov_mat.T + cov_mat
    cov_mat[np.diag_indices_from(cov_mat)] = 1
    cov_mat = nearestPD(cov_mat)
    return mvr(
        mean=de*np.ones(n_genes),
        cov=cov_mat,
        size=n_samples,
        # check_valid='ignore'
    )

def gendat_mix(
    n_genes=5,n_samples=30,
    mix_patten={
        'de':[0.2,1.5],
        'dc':[0.2,0.8]
        },
    negtive_ratio = 10
    ):
    """ Generate mixing patten simulations.
    DE: strong / weak / None
    DC: strong / weak / None

    all_genes = n_genes*8*(negtive_ratio+1)
    """
    des = mix_patten['de'][1]
    dew = mix_patten['de'][0]
    dcs = mix_patten['dc'][1]
    dcw = mix_patten['dc'][0]

    # 1: de s + dc s
    gen1 = gendat_single(n_genes,n_samples,dcs,des)
    # 2: de s + dc w
    gen2 = gendat_single(n_genes,n_samples,dcw,des)
    # 3: de w + dc s
    gen3 = gendat_single(n_genes,n_samples,dcs,dew)
    # 4: de w + dc w
    gen4 = gendat_single(n_genes,n_samples,dcw,dew)
    # 5: de s + dc n
    gen5 = gendat_single(n_genes,n_samples,0,des)
    # 6: de w + dc n
    gen6 = gendat_single(n_genes,n_samples,0,dew)
    # 7: de n + dc s
    gen7 = gendat_single(n_genes,n_samples,dcs,0)
    # 8: de n + dc w
    gen8 = gendat_single(n_genes,n_samples,dcw,0)
    # 9: de n + dc n
    gen9 = gendat_single(n_genes*8*negtive_ratio,n_samples,0,0)

    gen = np.concatenate((gen1,gen2,gen3,gen4,gen5,gen6,gen7,gen8,gen9),axis=1)
    label = [1]*n_genes*8 + [0]* gen9.shape[1]
    negtive_gen = gendat_single(
        gen.shape[1],n_samples,0,0)
    return gen,label,negtive_gen


import numpy as np
import math
import warnings

from scipy.stats import spearmanr
from scipy.spatial.distance import pdist

from sklearn.preprocessing import StandardScaler

# In this code we used masked arrays when calculating correlations. We did that, because
# some constant rows/columns might occur in some of the biclusters found. In these cases,
# the correlations will return NaN values and the code might raise exceptions.
# NumPy implements masked arrays to deal with missing or invalid entries.

#warnings.filterwarnings('ignore')

__SCALER = StandardScaler()

def _means(bicluster_data):
    mean = np.mean(bicluster_data)
    row_means = np.mean(bicluster_data, axis=1)
    col_means = np.mean(bicluster_data, axis=0)
    return mean, row_means, col_means

def _pearson(X):
    c = np.ma.corrcoef(X)
    return c if not c.mask.all() else np.ma.zeros(c.shape)

def _abs_pearson(X):
    c = _pearson(X)
    return np.ma.abs(c)

def _spearman(X):
    s, _ = spearmanr(X, axis=1)

    if isinstance(s, np.ma.core.MaskedConstant):
        s = 0.0

    if isinstance(s, float):
        return s

    s = np.ma.array(s, mask=np.isnan(s))
    return s if not s.mask.all() else np.ma.zeros(s.shape)

def _abs_spearman(X):
    return np.ma.abs(_spearman(X))

def _mean_residue(bicluster_data, func=lambda x : x ** 2):
    mean, row_means, col_means = _means(bicluster_data)
    residues = func(bicluster_data - row_means[:, np.newaxis] - col_means + mean)
    return np.mean(residues)

def isclose(a, b, tol=1e-10):
    return abs(a-b) <= tol

# Variance (VAR)
def var(bicluster_data):
    mean = np.mean(bicluster_data)
    VAR = np.sum((bicluster_data - mean) ** 2)
    assert VAR >= 0.0
    return VAR

# Mean Squared Residue (MSR)
def msr(bicluster_data):
    MSR = _mean_residue(bicluster_data)
    assert MSR >= 0.0
    return MSR

# Mean Absolute Residue (MAR)
def mar(bicluster_data):
    MAR = _mean_residue(bicluster_data, func=np.abs)
    assert MAR >= 0.0
    return MAR

# Relevance Index (RI)
def ri(bicluster_data, bicluster_col_global_data):
    assert bicluster_data.shape[1] == bicluster_col_global_data.shape[1]
    global_var = np.var(bicluster_col_global_data, axis=0)
    local_var = np.var(bicluster_data, axis=0)
    relevance = 1.0 - local_var / global_var
    return np.sum(relevance)

# Constancy by rows (Cr)
def constancy_by_rows(bicluster_data):
    n, m = bicluster_data.shape
    dist = pdist(bicluster_data, metric='euclidean')
    Cr = np.sum(dist) / n
    assert len(dist) == n * (n - 1) // 2
    assert Cr >= 0.0
    return Cr

# Constancy by cols (Cc)
def constancy_by_cols(bicluster_data):
    return constancy_by_rows(bicluster_data.T)

# Overall Constancy (OC)
def oc(bicluster_data):
    n, m = bicluster_data.shape
    Cr = constancy_by_rows(bicluster_data)
    Cc = constancy_by_cols(bicluster_data.T)
    OC = (n * Cr + m * Cc) / (n + m)
    assert OC >= 0.0
    return OC

# Scaling Mean Squared Residue (SMSR)
def smsr(bicluster_data, eps=1e-50):
    mean, row_means, col_means = _means(bicluster_data)
    x = (np.outer(row_means, col_means) - bicluster_data * mean) ** 2
    y = np.outer(row_means ** 2, col_means ** 2) + eps # adding a small min value (eps) to the denominator to avoid division by zero
    scaling_residues = x / y
    SMSR = np.mean(scaling_residues)
    assert math.isfinite(SMSR) # if y is too small, SMSR might be infinite
    assert SMSR >= 0.0 or isclose(SMSR, 0.0)
    return SMSR

# Minimal Mean Squared Error (MMSE)
def mmse(bicluster_data):
    n, m = bicluster_data.shape
    row_means = np.mean(bicluster_data, axis=1)
    D = bicluster_data - row_means[:, np.newaxis]
    S = np.dot(D, D.T) if n < m else np.dot(D.T, D)
    abs_eigvals = np.abs(np.linalg.eigvals(S))
    MMSE = (np.sum(D ** 2) - np.max(abs_eigvals)) / (n * m)
    assert MMSE >= 0.0 or isclose(MMSE, 0.0)
    return MMSE

# Average Correlation (AC)
def ac(bicluster_data, corr=_pearson):
    n, m = bicluster_data.shape
    c = corr(bicluster_data)

    if isinstance(c, float): # needed if the bicluster has only 2 rows and using spearmanr
        return c

    diag = np.ma.diag(c)
    AC = (np.ma.sum(c) - np.ma.sum(diag)) / (n ** 2 - n)
    assert -1.0 <= AC <= 1.0 or isclose(AC, -1.0) or isclose(AC, 1.0)
    return AC

# Sub-Matrix Correlation Score (SCS)
def scs(bicluster_data):
    def score(bicluster_data):
        abs_corr = _abs_pearson(bicluster_data)

        if isinstance(abs_corr, float): # needed if the bicluster has only 2 rows and using spearmanr
            return 1 - abs_corr

        n, m = abs_corr.shape
        row_scores = 1 - (np.ma.sum(abs_corr, axis=1) - np.ma.diag(abs_corr)) / (n - 1)
        return np.ma.min(row_scores)

    row_score = score(bicluster_data)
    col_score = score(bicluster_data.T)
    SCS = min(row_score, col_score)
    assert 0.0 <= SCS <= 1.0 or isclose(SCS, 0.0) or isclose(SCS, 1.0)
    return SCS

# Average Correlation Value (ACV)
def acv(bicluster_data, corr=_abs_pearson):
    avg_row_corr = ac(bicluster_data, corr=corr)
    avg_col_corr = ac(bicluster_data.T, corr=corr)
    ACV = max(avg_row_corr, avg_col_corr)
    assert -1.0 <= ACV <= 1.0 or isclose(ACV, -1.0) or isclose(ACV, 1.0) # checking in [-1, 1] because asr (below) calls acv
    return ACV

# Average Spearman's Rho (ASR)
def asr(bicluster_data):
    ASR = acv(bicluster_data, corr=_spearman)
    assert -1.0 <= ASR <= 1.0 or isclose(ASR, -1.0) or isclose(ASR, 1.0)
    return ASR

# Spearman's Biclustering Measure (SBM)
def sbm(bicluster_data, full_data, alpha_thr=9, beta_reliability=1.0):
    avg_row_corr = ac(bicluster_data, corr=_abs_spearman)
    avg_col_corr = ac(bicluster_data.T, corr=_abs_spearman)

    n, m = bicluster_data.shape
    N, M = full_data.shape

    if m > alpha_thr:
        alpha_reliability = 1.0
    else:
        alpha_reliability = m / M

    SBM = alpha_reliability * avg_row_corr * beta_reliability * avg_col_corr
    assert SBM >= 0.0
    return SBM

# Maximal Standard Area (MSA)
def msa(bicluster_data):
    n, m = bicluster_data.shape
    scaled_bicluster_data = __SCALER.fit_transform(bicluster_data.T)
    upper = np.max(scaled_bicluster_data, axis=1)
    lower = np.min(scaled_bicluster_data, axis=1)
    assert len(upper) == len(lower) == m
    MSA = sum(abs(upper[j] - lower[j] + upper[j+1] - lower[j+1]) / 2 for j in range(m - 1))
    assert MSA >= 0.0
    return MSA

# Virtual Error (VE)
def virtual_error(bicluster_data):
    virtual_row = np.mean(bicluster_data, axis=0)
    scaled_virtual_row = __SCALER.fit_transform(virtual_row[:, np.newaxis])
    scaled_bicluster_data = __SCALER.fit_transform(bicluster_data.T)
    VE = np.mean(np.abs(scaled_bicluster_data - scaled_virtual_row))
    assert VE >= 0.0
    return VE

# Transposed Virtual Error (VEt)
def transposed_virtual_error(bicluster_data):
    return virtual_error(bicluster_data.T)

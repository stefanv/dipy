#from enthought.mayavi import mlab
import numpy as np
from scipy.special import sph_harm

def real_sph_harm(m, n, theta, phi):
    """
    Compute real spherical harmonics, where the real harmonic $Y^m_n$ is defined
    to be:
        Real($Y^m_n$) * sqrt(2) if m > 0
        $Y^m_n$                 if m == 0
        Imag($Y^m_n$) * sqrt(2) if m < 0
    
    This is a ufunc and may take scalar or array arguments like any other ufunc.
    The inputs will be broadcasted against each other.
    
    :Parameters:
      - `m` : int |m| <= n
        The order of the harmonic.
      - `n` : int >= 0
        The degree of the harmonic.
      - `theta` : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.
      - `phi` : float [0, pi]
        The polar (colatitudinal) coordinate.
    
    :Returns:
      - `y_mn` : real float
        The real harmonic $Y^m_n$ sampled at `theta` and `phi`.

    :See also:
        scipy.special.sph_harm
    """
    m = np.atleast_1d(m)
    m_eq0,junk,junk,junk = np.broadcast_arrays(m == 0, n, theta, phi)
    m_gt0,junk,junk,junk = np.broadcast_arrays(m > 0, n, theta, phi)
    m_lt0,junk,junk,junk = np.broadcast_arrays(m < 0, n, theta, phi)

    sh = sph_harm(m, n, theta, phi)
    real_sh = np.zeros(sh.shape, 'double')
    real_sh[m_eq0] = sh[m_eq0].real
    real_sh[m_gt0] = sh[m_gt0].real * np.sqrt(2)
    real_sh[m_lt0] = sh[m_lt0].imag * np.sqrt(2)
    return real_sh

def sph_harm_ind_list(sh_order):
    
    ncoef = (sh_order + 2)*(sh_order + 1)/2

    if sh_order % 2 != 0:
        raise ValueError('sh_order must be an even integer >= 0')
    
    n_range = np.arange(0, np.int(sh_order+1), 2)
    n_list = np.repeat(n_range, n_range*2+1)

    offset = 0
    m_list = np.zeros(ncoef, 'int')
    for ii in n_range:
        m_list[offset:offset+2*ii+1] = np.arange(-ii, ii+1)
        offset = offset + 2*ii + 1

    n_list = n_list[..., np.newaxis]
    m_list = m_list[..., np.newaxis]
    return (m_list, n_list)

class ODF():
    
    def __init__(self,data, sh_order, grad_table, b_values, keep_resid=False):
        from scipy.special import lpn

        if (sh_order % 2 != 0 or sh_order < 0 ):
            raise ValueError('sh_order must be an even integer >= 0')
        self.sh_order = sh_order
        dwi = b_values > 0
        self.n_grad = dwi.sum()

        theta = np.arctan2(grad_table[1, dwi], grad_table[0, dwi])
        phi = np.arccos(grad_table[2, dwi])

        m_list, n_list = sph_harm_ind_list(self.sh_order)
        comp_mat = real_sph_harm(m_list, n_list, theta, phi)

        #self.fit_matrix = np.dot(comp_mat.T, np.linalg.inv(np.dot(comp_mat, comp_mat.T)))
        self.fit_matrix = np.linalg.pinv(comp_mat)
        legendre0, junk = lpn(self.sh_order, 0)
        funk_radon = legendre0[n_list]
        self.fit_matrix *= funk_radon.T

        self.shape = data.shape[:-1]
        self.b0 = data[..., np.logical_not(dwi)]
        self.coef = np.dot(data[..., dwi], self.fit_matrix)

        if keep_resid:
            unfit = comp_mat / funk_radon
            self.resid = data[..., dwi] - np.dot(self.coef, unfit)
        else:
            self.resid = None

    def evaluate_at(self, theta_e, phi_e):
        
        m_list, n_list = sph_harm_ind_list(self.sh_order)
        comp_mat = real_sph_harm(m_list, n_list, theta_e.flat[:], phi_e.flat[:])
        values = np.dot(self.coef, comp_mat)
        values.shape = self.coef.shape[:-1] + np.broadcast(theta_e,phi_e).shape
        return values

    def evaluate_boot(self, theta_e, phi_e, permute=None):
        m_list, n_list = sph_harm_ind_list(self.sh_order)
        comp_mat = real_sph_harm(m_list, n_list, theta_e.flat[:], phi_e.flat[:])
        if permute == None:
            permute = np.random.permutation(self.n_grad)
        values = np.dot(self.coef + np.dot(self.resid[..., permute], self.fit_matrix), comp_mat)
        return values

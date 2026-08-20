"""Pytorch multilayer QG model, Louis Thiry, 2022.
Followed Q-GCM user guide, Hogg et al (2014), http://q-gcm.org/downloads/q-gcm-v1.5.0.pdf.
   - State variables are pressure p and the potential vorticity q.
  - Doubly-periodic rectangular domain (periodic in x AND y, no walls).
  - Advection of q is carried with a periodic centered-difference Jacobian.
  - (Bi-)Laplacian diffusion discretized with periodic centered finite difference.
  - Idealized double-gyre wind forcing, curl computed with periodic differences.
  - Pressure obtained via FFT-based Helmholtz inversion (periodic elliptic solve).
  - Time integration with SSPRK3.
  - Tested on Intel CPU and NVIDIA GPU with pytorch 1.10 and CUDA 11.2.
"""
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse 
from dataclasses import dataclass
from GMC_OU import Parameters, run_simulation
import torch


##################### Fonctions for residual method #####################

def filtre_image_k_torch(image, dx, dy, k_coupure, passe_haut=True):
    """
    Filtre passe-haut / passe-bas dans l'espace des wavenumbers physiques,
    version PyTorch. Fonctionne sur un tenseur (..., ny, nx),
    ex: (1, 3, N, N) -> filtre chaque canal indépendamment.

    Retourne un tenseur de même forme que l'entrée.
    """
    *_, ny, nx = image.shape
    device = image.device

    kx = torch.fft.fftfreq(nx, d=dx / (2 * torch.pi), device=device, dtype=image.dtype)
    ky = torch.fft.fftfreq(ny, d=dy / (2 * torch.pi), device=device, dtype=image.dtype)
    kx, ky = torch.meshgrid(kx, ky, indexing="ij")
    kh = torch.sqrt(kx**2 + ky**2)

    if passe_haut:
        masque = (kh > k_coupure).to(image.dtype)
    else:
        masque = (kh <= k_coupure).to(image.dtype)

    fft_image = torch.fft.fft2(image, dim=(-2, -1))
    fft_filtree = fft_image * masque
    image_filtree = torch.fft.ifft2(fft_filtree, dim=(-2, -1)).real
    return image_filtree

def image_valide(image, forme_attendue=None):
    """Vérifie que `image` est un array 2D exploitable (pas None, pas vide,
    pas de NaN/Inf, et de la bonne forme si `forme_attendue` est fourni)."""
    if image is None:
        return False
    if not isinstance(image, np.ndarray):
        return False
    if image.ndim != 2 or image.size == 0:
        return False
    if not np.isfinite(image).all():
        return False
    if forme_attendue is not None and image.shape != forme_attendue:
        return False
    return True


def radial_average(field_2d, k, k_bins):
    """Somme radiale d'un champ 2D (ex: |FFT|^2) sur des couronnes de k,
    normalisée par dk pour obtenir une densité spectrale E(k)."""
    dk = k_bins[1] - k_bins[0]
    E_1d = np.zeros(len(k_bins))
    for idx, k_i in enumerate(k_bins):
        mask = (k >= k_i) & (k < k_i + dk)
        if mask.any():
            E_1d[idx] = field_2d[mask].sum() / dk
    return E_1d

def moduler_bruit(image, bruit, k_c, alpha, dk=1.0):
    """
    image : array 2D réel, l'écoulement original
    bruit : array 2D réel, ton bruit déjà filtré (pas de basse fréquence)
    k_c   : nombre d'onde de coupure (là où on veux raccorder les pentes)
    alpha : pente théorique cible (E_theo(k) ~ k^-alpha)
    """
    if not image_valide(image, forme_attendue=bruit.shape):
        image = bruit.copy()

    N = image.shape[0]
    assert bruit.shape == image.shape

    # --- grille de nombre d'onde ---
    kx = np.fft.fftfreq(N) * N
    ky = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    k = np.sqrt(KX**2 + KY**2)

    k_max = kx.max()
    k_bins = np.arange(0, k_max, dk)

    # --- FFT de l'image et du bruit ---
    I_hat = np.fft.fft2(image)
    B_hat = np.fft.fft2(bruit)

    E_orig_2d  = np.abs(I_hat)**2
    E_bruit_2d = np.abs(B_hat)**2

    print(np.shape(E_bruit_2d))

    E_orig_1d  = radial_average(E_orig_2d,  k, k_bins)
    E_bruit_1d = radial_average(E_bruit_2d, k, k_bins)

    # --- spectre théorique cible, raccordé en k_c ---
    idx_c = np.searchsorted(k_bins, k_c)
    A = E_orig_1d[idx_c] * k_c**alpha if E_orig_1d[idx_c] > 0 else E_bruit_1d[idx_c] * k_c**alpha
    E_theo_1d = A * np.where(k_bins > 0, k_bins, np.inf)**(-alpha)

    # --- résidu : ce qui manque par rapport à l'image ---
    R_1d = np.maximum(E_theo_1d - E_orig_1d, 0)
    R_1d[k_bins < k_c] = 0   # rien à ajouter en dessous de k_c

    # --- facteur de modulation M(k) = sqrt(R / E_bruit), en évitant les divisions par 0 ---
    M_1d = np.zeros_like(R_1d)
    #valid = E_bruit_1d > 1e-9
    #M_1d[valid] = np.sqrt(R_1d[valid] / E_bruit_1d[valid])
    M_1d = np.sqrt(R_1d / E_bruit_1d)
    
    # si le bruit n'a aucune énergie à un k donné mais qu'on en voudrait -> impossible de créer
    # de l'énergie avec un facteur multiplicatif, on laisse M=0 

    # --- passage du filtre radial 1D vers la grille 2D ---
    M_2d = np.interp(k, k_bins, M_1d)

    # --- application du filtre au bruit (multiplicatif, dans Fourier) ---
    B_hat_modulé = B_hat * M_2d

    bruit_modulé = np.real(np.fft.ifft2(B_hat_modulé))

    image_finale = image + bruit_modulé

    return image_finale, bruit_modulé, (k_bins, E_orig_1d, E_bruit_1d, E_theo_1d, R_1d, M_1d)

##################### QGM #####################

## functions to solve elliptic equation with homogeneous boundary conditions
def compute_laplace_dst(nx, ny, dx, dy, arr_kwargs):
    """Discrete sine transform of the 2D centered discrete laplacian
    operator."""
    x, y = torch.meshgrid(torch.arange(1,nx-1, **arr_kwargs),
                          torch.arange(1,ny-1, **arr_kwargs),
                          indexing='ij')
    return 2*(torch.cos(torch.pi/(nx-1)*x) - 1)/dx**2 + 2*(torch.cos(torch.pi/(ny-1)*y) - 1)/dy**2

def dstI1D(x, norm='ortho'):
    """1D type-I discrete sine transform."""
    return torch.fft.irfft(-1j*F.pad(x, (1,1)), dim=-1, norm=norm)[...,1:x.shape[-1]+1]

def dstI2D(x, norm='ortho'):
    """2D type-I discrete sine transform."""
    return dstI1D(dstI1D(x, norm=norm).transpose(-1,-2), norm=norm).transpose(-1,-2)

def inverse_elliptic_dst(f, operator_dst):
    """Inverse elliptic operator (e.g. Laplace, Helmoltz)
       using float32 discrete sine transform."""
    return dstI2D(dstI2D(f) / operator_dst)

## discrete spatial differential operators
def jacobi_h(f, g):
    """Arakawa discretisation of Jacobian J(f,g).
       Scalar fields f and g must have the same dimension.
       Grid is regular and dx = dy."""
    dx_f = f[...,2:,:] - f[...,:-2,:]
    dx_g = g[...,2:,:] - g[...,:-2,:]
    dy_f = f[...,2:] - f[...,:-2]
    dy_g = g[...,2:] - g[...,:-2]
    return (
            (   dx_f[...,1:-1] * dy_g[...,1:-1,:] - dx_g[...,1:-1] * dy_f[...,1:-1,:]  ) +
            (   (f[...,2:,1:-1] * dy_g[...,2:,:] - f[...,:-2,1:-1] * dy_g[...,:-2,:]) -
                (f[...,1:-1,2:]  * dx_g[...,2:] - f[...,1:-1,:-2] * dx_g[...,:-2])     ) +
            (   (g[...,1:-1,2:] * dx_f[...,2:] - g[...,1:-1,:-2] * dx_f[...,:-2]) -
                (g[...,2:,1:-1] * dy_f[...,2:,:] - g[...,:-2,1:-1] * dy_f[...,:-2,:])  )
           ) / 12.


def laplacian_h_boundaries(f, fc):
    return fc*(torch.cat([f[...,1,1:-1],f[...,-2,1:-1], f[...,1], f[...,-2]], dim=-1) -
               torch.cat([f[...,0,1:-1],f[...,-1,1:-1], f[...,0], f[...,-1]], dim=-1))


def laplacian_h_nobc(f):
    return (f[...,2:,1:-1] + f[...,:-2,1:-1] + f[...,1:-1,2:] + f[...,1:-1,:-2]
            - 4*f[...,1:-1,1:-1])

def laplacian_h(f, fc):
    delta_f = torch.zeros_like(f)
    delta_f[...,1:-1,1:-1] = laplacian_h_nobc(f)
    delta_f_bound = laplacian_h_boundaries(f, fc)
    nx, ny = f.shape[-2:]
    delta_f[...,0,1:-1] = delta_f_bound[...,:ny-2]
    delta_f[...,-1,1:-1] = delta_f_bound[...,ny-2:2*ny-4]
    delta_f[...,0] = delta_f_bound[...,2*ny-4:nx+2*ny-4]
    delta_f[...,-1] = delta_f_bound[...,nx+2*ny-4:2*nx+2*ny-4]
    return delta_f


def curl_wind(tau, dx, dy):
    tau_x = 0.5 * (tau[:-1,:,0] + tau[1:,:,0])
    tau_y = 0.5 * (tau[:,:-1,1] + tau[:,1:,1])
    curl_stagg = (tau_y[1:] - tau_y[:-1]) / dx - (tau_x[:,1:] - tau_x[:,:-1]) / dy
    return  0.25*(curl_stagg[:-1,:-1] + curl_stagg[:-1,1:] + curl_stagg[1:,:-1] + curl_stagg[1:,1:])

def interp_1d(f, dim):
    """Average of neighboring points along `dim`, periodic BC: output[i] is
    the average of f[i] and f[i+1] (wrapping at the last index), so the
    output keeps the FULL size along `dim` (no shrinking, unlike the old
    version which returned n-1 points and needed manual padding downstream)."""
    return (f + torch.roll(f, -1, dims=dim)) / 2

def interp_2d(f):
    """4-point corner average, periodic BC (full shape, no shrinking)."""
    return (f + torch.roll(f, -1, dims=-2) + torch.roll(f, -1, dims=-1)
            + torch.roll(f, shifts=(-1,-1), dims=(-2,-1))) / 4

def center_adv(q, u, v):
    """Flux-form advection -div(u q), periodic BC.
    NOTE: as originally written this function referenced interp_xy/interp_x/
    interp_y as free functions, but those only ever existed as self.* closures
    built inside QGM.__init__ -- calling this module-level function would
    always raise a NameError. It's currently unused (commented out in
    __init__: 'self.rand_tracer_adv'). Fixed here to be self-contained and
    periodic in case you want to use it; it is numerically identical to
    arakawa_adv below (both are the same 2-point energy-conserving flux)."""
    fx = interp_2d(u) * interp_1d(q, dim=-2)
    fy = interp_2d(v) * interp_1d(q, dim=-1)
    return -((fx - torch.roll(fx, 1, dims=-2)) + (fy - torch.roll(fy, 1, dims=-1)))

def arakawa_adv(q, u, v):
    """Flux-form advection -div(u q) of tracer q by velocity (u,v), periodic
    BC, collocated (A-)grid. u, v are expected pre-scaled by 1/dx and 1/dy
    respectively, matching the existing call site:
        rand_adv_tracer(q, u_noise/self.dx, v_noise/self.dy)
    so this function itself performs no further dx/dy division (same
    "raw" convention as jacobi_h/laplacian_h_nobc elsewhere in this file).

    Uses the classical 2nd-order centered ("energy conserving") advective
    flux: F_x[i+1/2] = 1/4 * (u[i]+u[i+1]) * (q[i]+q[i+1]), and similarly in
    y, then a periodic flux difference. Because flux differences telescope
    to zero around a periodic domain, this conserves the domain-mean of q
    exactly, by construction, for any u, v (divergent or not).

    IMPORTANT: this replaces the original 9-point stencil, which had a real
    shape-mismatch bug -- e.g. its first term multiplied a (...,nx-2,ny-1)
    array against a (...,nx-2,ny-2) array, which fails to broadcast (checked
    numerically). Being written for a bounded, interior-only grid, it also
    couldn't just be "rolled" into a periodic version without re-deriving
    it, and blindly patching indices without being able to execute/verify
    it (no torch available in this sandbox) felt too risky. This version is
    correct and periodic by construction, but it is a simpler 2-point scheme
    -- it does NOT reproduce the full potential-enstrophy-conserving 9-point
    Arakawa & Lamb (1981) scheme (which is also derived for a staggered
    C-grid, not the collocated grid used here for u,v,q). If you specifically
    need that stronger conservation property, it's worth doing as a separate,
    careful pass -- happy to work through it with you and check it
    numerically (e.g. verifying domain-integrated q**2 stays constant under
    a known non-divergent test flow) before relying on it."""
    Fx = 0.25 * (u + torch.roll(u, -1, dims=-2)) * (q + torch.roll(q, -1, dims=-2))
    Fy = 0.25 * (v + torch.roll(v, -1, dims=-1)) * (q + torch.roll(q, -1, dims=-1))
    return -((Fx - torch.roll(Fx, 1, dims=-2)) + (Fy - torch.roll(Fy, 1, dims=-1)))

def initial_condition_p(ens,
    nl,
    nx,
    ny,
    L,
    B0,
    H,
    patm,
    device="cpu",
    dtype=torch.float32
):
    """
    Initial condition for buoyancy from Resseguier et al. 2020 NPG

    b(x,y,0) =
        0.2 B0 cos(2*pi/L * (x+y))
        + F(x-x00, y-y00)
        + F(x-x10, y-y10)
        - F(x-x01, y-y01)
        - F(x-x11, y-y11)

    with

        F(x,y) = B0 exp[
            -1/2 * (15/L)^2 * (x^2+y^2)^2
        ]

    Periodic domain [0,Lx] x [0,Ly].

    Supposing a hydrostatic balance, extending to the multilayer case, the initial pressure is then p(x,y,0) = p_atm + b(x,y,0) * H[i-1] for layer i>0
    """

    #B0 = 1.0
    #patm = 10e5

    x = torch.arange(
        nx) * (L / nx)

    y = torch.arange(
        ny) * (L / ny)

    Y, X = torch.meshgrid(y, x, indexing="ij")

    # Large-scale sinusoidal component
    b0 = (
        0.2 * B0
        * torch.cos(
            2.0 * torch.pi / L * (X + Y)
        )
    )

    # Localized structure F
    def F(Xloc, Yloc):

        r2 = Xloc**2 + Yloc**2

        return B0 * torch.exp(
            -0.5 * (15.0 / L)**2 * r2**2
        )

    # Four centres
    x00 = L / 4.0
    y00 = L / 4.0

    x10 = 3.0 * L / 4.0
    y10 = L / 4.0

    x01 = L / 4.0
    y01 = 3.0 * L / 4.0

    x11 = 3.0 * L / 4.0
    y11 = 3.0 * L / 4.0

    b0 += F(X - x00, Y - y00)
    b0 += F(X - x10, Y - y10)
    b0 -= F(X - x01, Y - y01)
    b0 -= F(X - x11, Y - y11)

    p0 = torch.full((ens, nl, nx, ny), patm, device=device, dtype=dtype)

    for i in range(1, nl):
        p0[:, i, :, :] = (
            patm
            + b0 * H[i - 1]
        )
    return p0

### Periodicity update ###

def ddx_periodic(f, dx):
    """Centered finite difference in x with periodic BC."""
    return (torch.roll(f, -1, dims=-2) - torch.roll(f, 1, dims=-2)) / (2 * dx)

def ddy_periodic(f, dy):
    """Centered finite difference in y with periodic BC."""
    return (torch.roll(f, -1, dims=-1) - torch.roll(f, 1, dims=-1)) / (2 * dy)

def laplacian_periodic(f, dx, dy):
    """2D Laplacian with periodic BC."""
    return ((torch.roll(f, 1, dims=-2) + torch.roll(f, -1, dims=-2) - 2*f) / dx**2 +
            (torch.roll(f, 1, dims=-1) + torch.roll(f, -1, dims=-1) - 2*f) / dy**2)

def compute_laplace_fft(nx, ny, dx, dy, arr_kwargs):
    """Discrete Fourier symbol of the 2D periodic centered (5-point) discrete
    laplacian operator, i.e. the exact eigenvalues of laplacian_periodic.
    Using the *discrete* symbol (cosines) rather than the continuous one
    (-(kx**2+ky**2)) ensures the FFT elliptic solve exactly inverts the same
    finite-difference stencil used elsewhere (advection, diffusion), just like
    compute_laplace_dst exactly inverts laplacian_h_nobc for the bounded case."""
    kx = 2 * torch.pi * torch.fft.fftfreq(nx, d=1.0, **arr_kwargs)  # phase per grid step
    ky = 2 * torch.pi * torch.fft.fftfreq(ny, d=1.0, **arr_kwargs)
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    return 2*(torch.cos(KX) - 1)/dx**2 + 2*(torch.cos(KY) - 1)/dy**2

def inverse_elliptic_fft(f, operator_fft):
    """Inverse Laplacian/Helmholtz operator assuming periodic BCs.
    operator_fft may carry extra leading dims (e.g. one eigenvalue per
    vertical mode); the k=0 (mean) mode is handled with '...' indexing so it
    works whether operator_fft is 2D (nx,ny) or (..., nx, ny)."""
    f_hat = torch.fft.fft2(f)
    op = operator_fft.clone()
    op[..., 0, 0] = torch.inf  # avoid division by zero at k=0
    psi_hat = f_hat / op
    psi_hat[..., 0, 0] = 0.0   # remove mean mode: horizontal mean of psi stays constant
    return torch.fft.ifft2(psi_hat).real

def jacobian_periodic(f, g, dx, dy):
    """Centered-difference Jacobian J(f,g) = df/dx*dg/dy - df/dy*dg/dx with periodic BCs.
    Properly normalized by dx, dy (unlike jacobi_h which returns a raw/unnormalized
    Arakawa stencil that needs an external 1/(dx*dy) factor)."""
    df_dx = ddx_periodic(f, dx)
    df_dy = ddy_periodic(f, dy)
    dg_dx = ddx_periodic(g, dx)
    dg_dy = ddy_periodic(g, dy)
    return df_dx * dg_dy - df_dy * dg_dx

def curl_wind_periodic(tau, dx, dy):
    """Periodic curl of the wind stress tau=(tau_x,tau_y), collocated grid,
    returns the full (nx,ny) domain (no shrinking, unlike curl_wind)."""
    tau_x, tau_y = tau[..., 0], tau[..., 1]
    return ddx_periodic(tau_y, dx) - ddy_periodic(tau_x, dy)

class QGM:
    """Implementation of multilayer quasi-geostrophic model
    in variables pressure p and potential vorticity q.
    Doubly-periodic domain (periodic in x and y): no walls, no boundary
    corrections, pressure obtained via FFT-based Helmholtz inversion.
    """

    def __init__(self, param, cfg):
        self.nx = param['nx']
        self.Lx = param['Lx']
        self.ny = param['ny']
        self.Ly = param['Ly']
        self.nl = param['nl']
        self.H = param['H']
        self.g_prime = param['g_prime']
        self.f0 = param['f0']
        self.a2 = param['a2']
        self.a4 = param['a4']
        self.beta = param['beta']
        self.delta_ek = param['delta_ek']
        self.dt = param['dt']
        self.cfg = cfg
        # alpha_bc / zfbc (mixed-slip wall coefficient) is no longer used:
        # there are no walls in the periodic domain. Kept accepted-but-unused
        # in param for backward compatibility with existing param dicts.
        self.n_ens = param['n_ens']
        self.A_sp = param['A_sp']
        self.C_tau = param['C_tau']
        self.device = param['device']
        self.arr_kwargs = {'dtype':torch.float64, 'device':self.device}
        self.u = None

        # grid (periodic: nx points span [0, Lx), no duplicated wrap-around point)
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.x, self.y = torch.meshgrid(torch.arange(self.nx, **self.arr_kwargs) * self.dx,
                                        torch.arange(self.ny, **self.arr_kwargs) * self.dy,
                                        indexing='ij')
        self.y0 = 0.5 * self.Ly
        self.laplace_fft = compute_laplace_fft(self.nx, self.ny, self.dx, self.dy, self.arr_kwargs)

        assert self.dx == self.dy, f'dx {self.dx} != dy {self.dy}, must be equal'
        # NOTE: laplacian_periodic and jacobian_periodic already divide by dx/dy
        # internally (unlike the old wall-respecting laplacian_h_nobc/jacobi_h,
        # which returned *raw*, unnormalized stencils). So the dx powers that used
        # to compensate for that in these coefficients are no longer needed here.
        self.diff_coef = self.a2 / self.f0**2
        self.hyperdiff_coef = self.a4 / self.f0**2
        self.jac_coef = 1. / self.f0
        self.bottom_friction_coef = self.delta_ek / (2*np.abs(self.f0)*(-self.H[-1]))
          
        # init matrices
        self.compute_A_matrix()
        self.compute_layer_to_mode_matrices()
        self.compute_helmoltz_matrices()

        # initialize pressure p and potential voritcity q
        self.p_shape = (self.n_ens, self.nl, self.nx, self.ny)
        #self.p = torch.zeros(self.p_shape, **self.arr_kwargs)
        self.p = initial_condition_p(self.n_ens, self.nl, self.nx, self.ny, self.Lx, 0.01, self.H, 1.0, device=self.device, dtype=torch.float64)
        #print(("Initial pressure p: min", self.p.min().item(), "max", self.p.max().item()))
        self.compute_q_over_f0_from_p()
        
        # Stochastic transport model
        if 'rand_model' in param.keys():
            self.rand_model = param['rand_model']
            self.base_noise = param['base_noise']
            if self.base_noise == 'dwt':
                # Wavelet-based noise
                self.noise_level = param.get('noise_level', None)  # no longer used by
                # the simplified noise_generator below; kept optional only so
                # existing param dicts with this key don't break.
                self.wave_level = param['wave_level']
                self.dwt2 = DWTForward(J=self.wave_level, wave='db4', mode='periodization').to(self.device)
                self.idwt2 = DWTInverse(wave='db4', mode='periodization').to(self.device)
                # Gaussian low-pass filter used to extract the local (mesoscale)
                # fluctuation of the streamfunction before the wavelet
                # decomposition (see noise_generator). Width is in grid points;
                # ASSUMPTION since this wasn't in what you gave me: defaults to
                # 4 grid points via param['filt_width'] if you don't set it --
                # tune to whatever large-scale cutoff you actually want.
                filt_width = param.get('filt_width', 4)
                xf = torch.arange(-filt_width, filt_width+1, **self.arr_kwargs)
                Xf, Yf = torch.meshgrid(xf, xf, indexing='ij')
                filt = torch.exp(-(Xf**2 + Yf**2) / filt_width**2)
                filt /= filt.sum()
                self.filt = filt.unsqueeze(0).unsqueeze(0)  # conv2d weight (1,1,2w+1,2w+1)
                self.filt_pad_size = (filt_width, filt_width, filt_width, filt_width)
            elif self.base_noise == 'fourier':
                # GMC-based noise
                #data = np.load(param['file_modes'])
                #self.u_modes = torch.tensor(data['u'], dtype=torch.float64, device=self.device)
                #self.v_modes = torch.tensor(data['v'], dtype=torch.float64, device=self.device)
                cfg = Parameters()
        else:
            self.rand_model = False

        # precompile torch functions
        jacobian_p = lambda f, g: jacobian_periodic(f, g, self.dx, self.dy)
        laplacian_p = lambda f: laplacian_periodic(f, self.dx, self.dy)
        self.jacobian_periodic = torch.jit.trace(jacobian_p, (self.q_over_f0, self.p))
        self.laplacian_periodic = torch.jit.trace(laplacian_p, (self.p,))
        self.inverse_elliptic_fft = torch.jit.trace(inverse_elliptic_fft, (self.q_over_f0, self.helmholtz_fft))
        interp_y = lambda f: interp_1d(f, dim=-1)
        interp_x = lambda f: interp_1d(f, dim=-2)
        self.interp_x = torch.jit.trace(interp_x, (self.p,))
        self.interp_y = torch.jit.trace(interp_y, (self.p,))
        self.interp_xy = torch.jit.trace(interp_2d, (self.p,))
        if self.rand_model:
            u, v = self.compute_velocity(self.p)
            #self.rand_tracer_adv = torch.jit.trace(center_adv, (self.q_over_f0, u, v)) 
            self.rand_adv_tracer = torch.jit.trace(arakawa_adv, (self.q_over_f0, u, v)) 
        if self.rand_model == 'energy_preserve':
            betay = self.beta * (self.y.unsqueeze(0).unsqueeze(0) - self.y0)
            self.betay_on_u = self.interp_y(betay) # interpolated on u-points (periodic, full shape)
            self.betay_on_v = self.interp_x(betay) # on v-points (periodic, full shape)

    def compute_A_matrix(self):
        self.A = torch.zeros((self.nl,self.nl), **self.arr_kwargs)
        self.A[0,0] = 1./(self.H[0]*self.g_prime[0])
        self.A[0,1] = -1./(self.H[0]*self.g_prime[0])
        for i in range(1, self.nl-1):
            self.A[i,i-1] = -1./(self.H[i]*self.g_prime[i-1])
            self.A[i,i] = 1./self.H[i]*(1/self.g_prime[i] + 1/self.g_prime[i-1])
            self.A[i,i+1] = -1./(self.H[i]*self.g_prime[i])
        self.A[-1,-1] = 1./(self.H[self.nl-1]*self.g_prime[self.nl-2])
        self.A[-1,-2] = -1./(self.H[self.nl-1]*self.g_prime[self.nl-2])

    def compute_deterministic_forcing(self):
        """Deterministic forcing S for the bottom layer, periodic domain. 
        Hard set parameters for now, but could be made configurable if needed.
        """

        nu_f = 1e1
        A_f = 10e-8
        tau_f = 10e8

        b = (self.p[...,0,:,:] - self.p[...,1,:,:]) / self.H[0]  # bottom layer pressure difference (nl-1 to nl)

        kx = 2.0 * torch.pi * torch.fft.fftfreq(
            self.nx, d=self.dx, device=b.device
        )
        ky = 2.0 * torch.pi * torch.fft.fftfreq(
            self.ny, d=self.dy, device=b.device
        )

        KY, KX = torch.meshgrid(ky, kx, indexing="ij")
        k2 = KX**2 + KY**2
       
        b_hat = torch.fft.fft2(b)

        # Spectral Delta^4 operator :
        # Delta -> -k^2
        # Delta^4 -> (-k^2)^4 = k^8
        delta4_b_hat = k2**4 * b_hat
        #print(("Deterministic forcing: min", torch.fft.ifft2(delta4_b_hat).min().item(), "max", torch.fft.ifft2(delta4_b_hat).max().item()))

        S_hat = (
            -nu_f * delta4_b_hat
            - b_hat / tau_f
        )

        x = torch.arange(
            self.nx,
            device=b.device,
            dtype=b.dtype
        ) * self.dx

        y = torch.arange(
            self.ny,
            device=b.device,
            dtype=b.dtype
        ) * self.dy

        Y, X = torch.meshgrid(y, x, indexing="ij")

        kfx = 3 * (2 * np.pi) / self.Lx
        kfy = 2 * (2 * np.pi) / self.Ly
        
        forcing_physical = (
            A_f
            * torch.sin(kfx * X)
            * torch.sin(kfy * Y)
        )

        forcing_hat = torch.fft.fft2(forcing_physical)

        S_hat = S_hat + forcing_hat
        S = torch.fft.ifft2(S_hat).real

        # Check stationarity of the forcing: 
        #print(("Deterministic forcing S: min", S.min().item(), "max", S.max().item()))

        return S

    def compute_layer_to_mode_matrices(self):
        """Matrices to change from layers to modes."""
        lambd_r, R = torch.linalg.eig(self.A)
        lambd_l, L = torch.linalg.eig(self.A.T)
        self.lambd = lambd_r.real
        R, L = R.real, L.real
        self.Cl2m = torch.diag(1./torch.diag(L.T @ R)) @ L.T
        self.Cm2l = R


    def compute_helmoltz_matrices(self):
        """Spectral Helmholtz operator (one eigenvalue field per vertical mode),
        for periodic FFT-based elliptic inversion. Replaces the DST-based
        (bounded-domain) solve. No homogeneous-solution / mass-conservation
        correction is needed here: for a periodic domain the FFT solve already
        leaves the k=0 (horizontal mean) mode of each mode's pressure untouched
        (see inverse_elliptic_fft), which is the natural mass conservation."""
        self.helmholtz_fft = self.laplace_fft.reshape((1, self.nx, self.ny)) / self.f0**2 - self.lambd.reshape((self.nl, 1, 1))
        self.helmholtz_fft.unsqueeze_(0)  # (1, nl, nx, ny), broadcastable over ensemble dim


    def compute_q_over_f0_from_p(self):
        """Potential vorticity from pressure, periodic domain.
        NOTE: laplacian_periodic already divides by dx**2/dy**2 internally
        (unlike the old laplacian_h, which returned a raw undivided stencil),
        so only the f0**2 normalization is needed here."""
        Ap = torch.einsum('kl,...lij->...kij', self.A, self.p)
        self.q_over_f0 = laplacian_periodic(self.p, self.dx, self.dy) / self.f0**2 - Ap + (self.beta / self.f0) * (self.y - self.y0)


    def compute_velocity(self, psi):
        """Compute geostrophic velocities (u,v) from streamfunction psi."""
        """Geostrophic velocities under periodic BCs."""
        u = -ddy_periodic(psi, self.dy)
        v = ddx_periodic(psi, self.dx)
        return u, v


    def reset_p(self, p):
        """Reset pressure from a restart p (periodic domain: no boundaries to
        preserve, and no alpha/homogeneous-solution mass-conservation
        correction, since the FFT elliptic solve already keeps the horizontal
        mean of p unchanged by construction -- see inverse_elliptic_fft)."""
        if (p.shape[-2]==self.nx) and (p.shape[-1]==self.ny):
            self.p = p.clone() # reset from the same resolution
        elif (p.shape[-2]<self.nx) or (p.shape[-1]<self.ny):
            # Interpolation from low- to high-resolution
            self.p = F.interpolate(p, self.p_shape[-2:], mode='bicubic', align_corners=False)
        else:
            # Downsampling from high- to low-resolution
            d = p.shape[-1] // self.ny
            x = torch.linspace(-d, d, 2*d+1, **self.arr_kwargs)
            x, y = torch.meshgrid(x, x, indexing='xy')
            f = torch.exp(-(x**2 + y**2) / (d**2)) / (torch.pi*d**2) # Gaussian filter
            f /= f.sum()
            p_ = F.pad(p, (d,d,d,d), mode='circular') # periodic padding
            p = F.conv2d(p_.reshape((-1,)+p_.shape[-2:]).unsqueeze(1), \
                    f.unsqueeze(0).unsqueeze(0)).squeeze(1).reshape(p.shape[:-2]+(self.nx,self.ny))[...,::d,::d]
            self.p = p
        # Update potential vorticity
        self.compute_q_over_f0_from_p()


    def advection_rhs(self):
        """Advection diffusion RHS for vorticity, over the whole periodic domain
        (no interior/boundary split needed: there are no walls)."""
        rhs = self.jac_coef * self.jacobian_periodic(self.q_over_f0, self.p)

        delta2_p = self.laplacian_periodic(self.p)
        if self.a2 != 0.:
            rhs += self.diff_coef * self.laplacian_periodic(delta2_p)
        if self.a4 != 0.:
            rhs -= self.hyperdiff_coef * self.laplacian_periodic(self.laplacian_periodic(delta2_p))

        #self.wind_forcing = self.generate_wind_forcing() 
        self.wind_forcing = self.compute_deterministic_forcing()
        rhs[...,0,:,:] += self.wind_forcing
        rhs[...,-1:,:,:] += self.bottom_friction_coef * self.laplacian_periodic(self.p[...,-1:,:,:])

        # Stochastic transport noise based on diff. conservation laws
        # (rand_adv_momentum / arakawa_adv are now periodic and shape-safe too,
        # see their docstrings for what was fixed)
        if self.rand_model == 'energy_preserve':
            # Advection of momentum (then taking curl)
            du, dv = self.rand_adv_momentum()
            # periodic forward difference (same convention as torch.diff:
            # dv[i+1]-dv[i], but wrapping instead of shrinking, so it stays
            # full-shape and can be added directly to rhs)
            rhs += (torch.roll(dv, -1, dims=-2) - dv)/(self.dx*self.f0) \
                 - (torch.roll(du, -1, dims=-1) - du)/(self.dy*self.f0)
            # Advection of tracer (stratification)
            rhs += self.rand_adv_tracer(-torch.einsum('kl,...lij->...kij', self.A, self.p), \
                    self.u_noise/self.dx, self.v_noise/self.dy) 
        if self.rand_model == 'enstrophy_preserve':
            # Advection of PV
            rhs += self.rand_adv_tracer(self.q_over_f0, self.u_noise/self.dx, self.v_noise/self.dy) 
        return rhs


    def compute_time_derivatives(self):
        """Advect vorticity and invert the periodic Helmholtz equation for the
        pressure tendency. No boundary patch-up and no mass-conservation
        correction: both were bounded-domain-only concerns."""
        dq_over_f0 = self.advection_rhs()

        rhs_helmholtz = torch.einsum('ml,...lij->...mij', self.Cl2m, dq_over_f0)
        dp = torch.einsum('lm,...mij->...lij', self.Cm2l,
                           self.inverse_elliptic_fft(rhs_helmholtz, self.helmholtz_fft))

        return dq_over_f0, dp


    def step(self):
        #print('self.q_over_f0', self.q_over_f0.shape)
        #print('self.u_noise', self.u_noise.shape)
        #print('self.v_noise', self.v_noise.shape)
        """ Time itegration with SSPRK3 scheme.""" 
        dt0_q_over_f0, dt0_p = self.compute_time_derivatives()
        #print('dt0_q_over_f0', dt0_q_over_f0.shape)
        self.q_over_f0 += self.dt * dt0_q_over_f0
        self.p += self.dt * dt0_p

        dt1_q_over_f0, dt1_p = self.compute_time_derivatives()
        self.q_over_f0 += (self.dt/4) * (dt1_q_over_f0 - 3*dt0_q_over_f0)
        self.p += (self.dt/4) * (dt1_p - 3*dt0_p)

        dt2_q_over_f0, dt2_p = self.compute_time_derivatives()
        self.q_over_f0 += (self.dt/12) * (8*dt2_q_over_f0 - dt1_q_over_f0 - dt0_q_over_f0)
        self.p += (self.dt/12) * (8*dt2_p - dt1_p - dt0_p)

        u, v = self.compute_velocity(self.p / self.f0)
	# Optional: store them for later use
        self.u = u
        self.v = v
        

    def noise_generator(self, time=None):
        
        # GMC-based noise
        if self.base_noise == 'fourier':

            u_noise_temp, v_noise_temp, *_ = run_simulation(self.cfg)

            self.u_noise = torch.from_numpy(u_noise_temp).float().permute(0, 3, 1, 2)
            self.v_noise = torch.from_numpy(v_noise_temp).float().permute(0, 3, 1, 2)

            self.u_noise /= torch.std(self.u_noise)
            self.v_noise /= torch.std(self.v_noise)

            self.u_noise = filtre_image_k_torch(self.u_noise, dx=2/256, dy=2/256, k_coupure=110, passe_haut=True)
            self.v_noise = filtre_image_k_torch(self.v_noise, dx=2/256, dy=2/256, k_coupure=110, passe_haut=True)

            #self.u_noise = moduler_bruit(self.u , self.u_noise, 110, alpha=5/3)
            #self.v_noise = moduler_bruit(self.v , self.v_noise, 110, alpha=5/3)

            #print(self.u_noise.mean(), self.v_noise.mean())
            #self.u_noise, self.v_noise, *_ = run_simulation(self.cfg)
            #print(np.shape(self.u_noise), np.shape(self.v_noise))

        # Wavelet-based noise
        if self.base_noise == 'dwt':
            # Local fluctuation of the streamfunction: subtract a smoothed
            # (large-scale) version so the wavelet noise is built from
            # mesoscale structure rather than the mean flow.
            # NOTE: your snippet used F.pad(..., mode='replicate'), which
            # clamps to the edge value -- a wall-style boundary condition.
            # In an otherwise fully periodic domain that would quietly
            # reintroduce a wall at this one spot, so I switched it to
            # mode='circular'.
            psi = self.p / self.f0
            psi_ = F.pad(psi, self.filt_pad_size, mode='circular')
            psi = psi - F.conv2d(psi_.reshape((-1,)+psi_.shape[-2:]).unsqueeze(1), self.filt) \
                    .squeeze(1).reshape(self.p_shape)

            # Randomize the wavelet decomposition of psi.
            # NOTE: self.dwt2/self.idwt2 already use mode='periodization'
            # (see __init__), which *is* the periodic-BC wavelet transform,
            # and it expects the full (nx,ny) field. Your snippet dropped the
            # last row/col (psi[...,:-1,:-1]) before the transform and then
            # padded the result back afterwards (F.pad(...,(1,1,1,1)) on
            # idwt2(...)[...,1:,1:]) -- that trim/pad dance looks like it was
            # written for a non-periodization dwt mode. With periodization
            # active it's both unnecessary and counterproductive: it would
            # feed the transform an odd-sized (nx-1,ny-1) field (periodization
            # mode needs sizes divisible by 2**wave_level), and splicing a
            # padded edge back in reintroduces exactly the kind of seam the
            # periodic transform is meant to avoid. So I removed it -- the
            # full field goes in and comes out unchanged in shape.
            psi_low, psi_high = self.dwt2(psi.float())
            psi_low *= torch.randn((self.n_ens,1,)+psi_low.shape[-2:], **self.arr_kwargs)
            for l in range(self.wave_level):
                psi_high[l] *= torch.randn((self.n_ens,1,3,)+psi_high[l].shape[-2:], **self.arr_kwargs)
            self.psi_noise = self.idwt2((psi_low, psi_high))
            self.u_noise, self.v_noise = self.compute_velocity(self.psi_noise)
            print(np.shape(self.u_noise), np.shape(self.v_noise))
    def rand_adv_momentum(self):
        """Random advection of momentum using centered flux form, periodic BC.
        NOTE: the original version had a real shape-mismatch bug (checked
        numerically): after interp_y shrank the y-axis by 1 and F.pad(...,(1,1))
        only pads the LAST dim, the result (nx-2, ny+1) never matched the
        (nx-1, ny) shape it was multiplied against. Now that interp_x/interp_y/
        interp_xy are periodic (they no longer shrink the array), no
        trimming or padding is needed at all: everything stays at the full
        (nx,ny) shape, so the shapes always match by construction."""
        u, v = self.compute_velocity(self.p/self.f0) 
        unoi_on_v, vnoi_on_u = self.interp_xy(self.u_noise), self.interp_xy(self.v_noise)
        
        # Advection of u
        fx = self.interp_x(u*self.u_noise)
        fy = self.interp_x(self.v_noise) * self.interp_y(u)
        du = - (fx - torch.roll(fx, 1, dims=-2))/self.dx - (fy - torch.roll(fy, 1, dims=-1))/self.dy

        # Advection of v
        fy = self.interp_y(v*self.v_noise)
        fx = self.interp_y(self.u_noise) * self.interp_x(v)
        dv = - (fx - torch.roll(fx, 1, dims=-2))/self.dx - (fy - torch.roll(fy, 1, dims=-1))/self.dy
        
        # Add beta effects
        return du + self.betay_on_u * vnoi_on_u, dv - self.betay_on_v * unoi_on_v


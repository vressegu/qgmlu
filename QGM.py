"""Pytorch multilayer QG model, Louis Thiry, 2022.
Followed Q-GCM user guide, Hogg et al (2014), http://q-gcm.org/downloads/q-gcm-v1.5.0.pdf.
   - State variables are pressure p and the potential vorticity q.
  - Rectangular domain, mixed slip boundary condition for velocity.
  - Advection of q is carried with the conservative Arakawa-Lamb scheme.
  - (Bi-)Laplacian diffusion discretized with centered finite difference.
  - Idealized double-gyre wind forcing
  - Time integration with SSPRK3.
  - Tested on Intel CPU and NVIDIA GPU with pytorch 1.10 and CUDA 11.2.
"""
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse 


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
    n = f.shape[dim]
    return (f.narrow(dim, 0, n-1) + f.narrow(dim, 1, n-1))/2


def interp_2d(f):
    return (f[...,:-1,:-1] + f[...,:-1,1:] + f[...,1:,:-1] + f[...,1:,1:])/4


def center_adv(q, u, v):
    fx = interp_xy(u) * interp_x(q[...,1:-1])
    fy = interp_xy(v) * interp_y(q[...,1:-1,:])
    return -(torch.diff(fx, dim=-2) + torch.diff(fy, dim=-1))


def arakawa_adv(q, u, v): 
    fxp = (u[...,1:-1,1:] + u[...,2:,1:] + u[...,1:-1,:-1] + u[...,2:,:-1]) * \
          (q[...,1:-1,1:-1] + q[...,2:,1:-1]) + 0.5*( (u[...,1:-1,1:] + u[...,2:,1:])* \
          (q[...,1:-1,1:-1] + q[...,2:,2:]) + (u[...,1:-1,:-1] + u[...,2:,:-1])* \
          (q[...,1:-1,1:-1] + q[...,2:,:-2]) )
    fxm = (u[...,:-2,1:] + u[...,:-2,:-1] + u[...,1:-1,:-1] + u[...,1:-1,1:]) * \
          (q[...,1:-1,1:-1] + q[...,:-2,1:-1]) + 0.5*( (u[...,1:-1,1:] + u[...,:-2,1:])* \
          (q[...,1:-1,1:-1] + q[...,:-2,2:]) + (u[...,:-2,:-1] + u[...,1:-1,:-1])* \
          (q[...,1:-1,1:-1] + q[...,:-2,:-2]) )
    fyp = (v[...,:-1,2:] + v[...,:-1,1:-1] + v[...,1:,1:-1] + v[...,1:,2:]) * \
          (q[...,1:-1,1:-1] + q[...,1:-1,2:]) + 0.5*( (v[...,1:,1:-1] + v[...,1:,2:])* \
          (q[...,1:-1,1:-1] + q[...,2:,2:]) + (v[...,:-1,2:] + v[...,:-1,1:-1])* \
          (q[...,1:-1,1:-1] + q[...,:-2,2:]) )
    fym = (v[...,:-1,1:-1] + v[...,:-1,:-2] + v[...,1:,:-2] + v[...,1:,1:-1]) * \
          (q[...,1:-1,1:-1] + q[...,1:-1,:-2]) + 0.5*( (v[...,1:,1:-1] + v[...,1:,:-2])* \
          (q[...,1:-1,1:-1] + q[...,2:,:-2]) + (v[...,:-1,1:-1] + v[...,:-1,:-2])* \
          (q[...,1:-1,1:-1] + q[...,:-2,:-2]) )
    return -(fxp - fxm + fyp - fym)/12


class QGM:
    """Implementation of multilayer quasi-geostrophic model
    in variables pressure p and potential vorticity q.
    """

    def __init__(self, param):
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
        self.alpha_bc = param['alpha_bc']
        self.n_ens = param['n_ens']
        self.zfbc = self.alpha_bc / (1. + 0.5*self.alpha_bc)
        self.device = param['device']
        self.arr_kwargs = {'dtype':torch.float64, 'device':self.device}

        # grid
        self.x, self.y = torch.meshgrid(torch.linspace(0, self.Lx, self.nx, **self.arr_kwargs),
                                        torch.linspace(0, self.Ly, self.ny, **self.arr_kwargs),
                                        indexing='ij')
        self.y0 = 0.5 * self.Ly
        self.dx = self.Lx / (self.nx-1)
        self.dy = self.Ly / (self.ny-1)

        assert self.dx == self.dy, f'dx {self.dx} != dy {self.dy}, must be equal'
        self.diff_coef = self.a2 / self.f0**2 / self.dx**4
        self.hyperdiff_coef = (self.a4 / self.f0**2) / self.dx**6
        self.jac_coef = 1. / (self.f0 * self.dx * self.dy)
        self.bottom_friction_coef = self.delta_ek / (2*np.abs(self.f0)*self.dx**2*(-self.H[-1]))

        tau = torch.zeros((self.nx, self.ny, 2), **self.arr_kwargs)
        tau[:,:,0] = - param['tau0'] * torch.cos(2*torch.pi*(torch.arange(self.ny, **self.arr_kwargs)+0.5)/self.ny).reshape((1, self.ny))
        self.wind_forcing = (curl_wind(tau, self.dx, self.dy) / (self.f0 * self.H[0])).unsqueeze(0)

        # init matrices
        self.compute_A_matrix()
        self.compute_layer_to_mode_matrices()
        self.compute_helmoltz_matrices()
        self.compute_alpha_matrix()

        # initialize pressure p and potential voritcity q
        self.p_shape = (self.n_ens, self.nl, self.nx, self.ny)
        self.p = torch.zeros(self.p_shape, **self.arr_kwargs)
        self.compute_q_over_f0_from_p()

        # Stochastic transport model
        if 'rand_model' in param.keys():
            self.rand_model = param['rand_model']
            self.base_noise = param['base_noise']
            if self.base_noise == 'dwt':
                # Wavelet-based noise
                self.filt_width = param['filt_width']
                self.wave_level = param['wave_level']
                self.filt = torch.ones((1,1,self.filt_width,self.filt_width), **self.arr_kwargs) / self.filt_width**2
                n = self.filt_width // 2
                self.filt_pad_size = (n,n,n,n)
                torch.set_default_dtype(torch.float64)
                self.dwt2 = DWTForward(J=self.wave_level, mode='symmetric', wave='db5').to(self.device) 
                self.idwt2 = DWTInverse(mode='symmetric', wave='db5').to(self.device)
            elif self.base_noise == 'eof':
                # EOF-based noise
                data = np.load(param['file_modes'])
                self.u_modes = torch.from_numpy(data['u']).type(torch.float64).to(self.device)
                self.v_modes = torch.from_numpy(data['v']).type(torch.float64).to(self.device)
                data.close()
            elif self.base_noise == 'dmd':
                # DMD-based noise
                data = np.load(param['file_modes'])
                self.omega = torch.from_numpy(data['omega']).type(torch.float64).to(self.device) # freq.
                self.u_modes = torch.from_numpy(data['u']).type(torch.complex128).to(self.device)
                self.v_modes = torch.from_numpy(data['v']).type(torch.complex128).to(self.device)
                data.close()
            # Include Girsanov drift (nudging)    
            if 'file_drift' in param.keys():
                self.mean_drift = True
                data = np.load(param['file_drift'])
                # Read steady drift component
                self.u_drift = torch.from_numpy(data['um']).type(torch.float64).to(self.device).unsqueeze(0)
                self.v_drift = torch.from_numpy(data['vm']).type(torch.float64).to(self.device).unsqueeze(0)
                if ('u' in data) and ('v' in data):
                    # Read cyclic drift component (DMD modes)
                    self.cyclic_drift = True
                    self.omega_drift = torch.from_numpy(data['omega']).type(torch.float64).to(self.device)
                    self.u_modes_drift = torch.from_numpy(data['u']).type(torch.complex128).to(self.device)
                    self.v_modes_drift = torch.from_numpy(data['v']).type(torch.complex128).to(self.device)
                else:
                    self.cyclic_drift = False
                data.close()
            else:
                self.mean_drift = False
        else:
            self.rand_model = False

        # precompile torch functions
        self.zfbc = torch.tensor(self.zfbc, **self.arr_kwargs) # convert to Tensor for tracing
        self.inverse_elliptic_dst = torch.jit.trace(inverse_elliptic_dst, (self.q_over_f0[...,1:-1,1:-1], self.helmoltz_dst))
        self.jacobi_h = torch.jit.trace(jacobi_h, (self.q_over_f0, self.p))
        self.laplacian_h = torch.jit.trace(laplacian_h, (self.p, self.zfbc))
        self.laplacian_h_boundaries = torch.jit.trace(laplacian_h_boundaries, (self.p, self.zfbc))
        self.laplacian_h_nobc = torch.jit.trace(laplacian_h_nobc, (self.p,))
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
            self.betay_on_u = self.interp_y(betay[...,1:-1,:]) # interoplated on u-points
            self.betay_on_v = self.interp_x(betay[...,1:-1]) # on v-points


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


    def compute_layer_to_mode_matrices(self):
        """Matrices to change from layers to modes."""
        lambd_r, R = torch.linalg.eig(self.A)
        lambd_l, L = torch.linalg.eig(self.A.T)
        self.lambd = lambd_r.real
        R, L = R.real, L.real
        self.Cl2m = torch.diag(1./torch.diag(L.T @ R)) @ L.T
        self.Cm2l = R


    def compute_helmoltz_matrices(self):
        self.helmoltz_dst = compute_laplace_dst(self.nx, self.ny, self.dx, self.dy, self.arr_kwargs).reshape((1, self.nx-2, self.ny-2)) / self.f0**2 - self.lambd.reshape((self.nl , 1, 1))
        constant_field = torch.ones((self.nl, self.nx, self.ny), **self.arr_kwargs) / (self.nx * self.ny)
        s_solutions = torch.zeros_like(constant_field)
        s_solutions[:,1:-1,1:-1] = inverse_elliptic_dst(constant_field[:,1:-1,1:-1], self.helmoltz_dst)
        self.homogeneous_sol = (constant_field +  s_solutions*self.lambd.reshape((self.nl, 1, 1)))[:-1] # ignore last solution correponding to lambd = 0, i.e. Laplace equation
        self.helmoltz_dst.unsqueeze_(0), self.homogeneous_sol.unsqueeze_(0)


    def compute_alpha_matrix(self):
        M = (self.Cm2l[1:] - self.Cm2l[:-1])[:self.nl-1,:self.nl-1] * self.homogeneous_sol[0].mean((1,2)).reshape((1, self.nl-1))
        M_inv = torch.linalg.inv(M)
        self.alpha_matrix = -M_inv @ (self.Cm2l[1:,:-1] - self.Cm2l[:-1,:-1])
        self.alpha_matrix.unsqueeze_(0)


    def compute_q_over_f0_from_p(self):
        Ap = torch.einsum('kl,...lij->...kij', self.A, self.p)
        self.q_over_f0 = laplacian_h(self.p, self.zfbc) / (self.f0*self.dx)**2 - Ap + (self.beta / self.f0) * (self.y - self.y0)


    def compute_velocity(self, psi):
        """Compute geostrophic velocities (u,v) from streamfunction psi."""
        return -torch.diff(psi, dim=-1) / self.dy, torch.diff(psi, dim=-2) / self.dx


    def reset_p(self, p):
        """Reset pressure from a restart p."""
        if (p.shape[-2]==self.nx) and (p.shape[-1]==self.ny):
            self.p = p.clone() # reset from the same resolution
        if (p.shape[-2]<self.nx) or (p.shape[-1]<self.ny): 
            # Interpolation from low- to high-resolution
            self.p = F.interpolate(p, self.p_shape[-2:], mode='bicubic', align_corners=False)
        if (p.shape[-2]>self.nx) or (p.shape[-1]>self.ny): 
            # Downsamping from high- to low-resolution
            d = (p.shape[-1]-1) // (self.ny-1) 
            x = torch.linspace(-d, d, 2*d+1, **self.arr_kwargs)
            x, y = torch.meshgrid(x, x, indexing='xy')
            f = torch.exp(-(x**2 + y**2) / (d**2)) / (torch.pi*d**2) # Gaussian filter
            f /= f.sum()
            p_ = F.pad(p, (d,d,d,d), mode='replicate')
            p = F.conv2d(p_.reshape((-1,)+p_.shape[-2:]).unsqueeze(1), \
                    f.unsqueeze(0).unsqueeze(0)).squeeze(1).reshape(p.shape)[...,::d,::d]
            self.p[...,1:-1,1:-1] = p[...,1:-1,1:-1] # no boundaries
        # Ensure mass conservation 
        p_modes = F.pad(torch.einsum('ml,...lij->...mij', self.Cl2m, self.p[...,1:-1,1:-1]), (1,1,1,1))
        alpha =  (self.alpha_matrix @ p_modes[...,:-1,:,:].mean((-2,-1)).unsqueeze(-1)).unsqueeze(-1)
        p_modes[...,:-1,:,:] += alpha * self.homogeneous_sol
        self.p = torch.einsum('lm,...mij->...lij', self.Cm2l, p_modes)
        # Update potential vortocity
        self.compute_q_over_f0_from_p()


    def advection_rhs(self):
        """Advection diffusion RHS for vorticity, only inside domain"""
        rhs = self.jac_coef * self.jacobi_h(self.q_over_f0, self.p)

        delta2_p = self.laplacian_h(self.p, self.zfbc)
        if self.a2 != 0.:
            rhs += self.diff_coef * self.laplacian_h_nobc(delta2_p)
        if self.a4 != 0.:
            rhs -= self.hyperdiff_coef * self.laplacian_h_nobc(self.laplacian_h(delta2_p, self.zfbc))

        rhs[...,0,:,:] += self.wind_forcing
        rhs[...,-1:,:,:] += self.bottom_friction_coef * self.laplacian_h_nobc(self.p[...,-1:,:,:])
        
        # Stochastic transport noise based on diff. conservation laws
        if self.rand_model == 'energy_preserve':
            # Advection of momentum (then taking curl)
            du, dv = self.rand_adv_momentum()
            rhs += torch.diff(dv, dim=-2)/(self.dx*self.f0) - torch.diff(du, dim=-1)/(self.dy*self.f0)
            # Advection of tracer (stratification)
            rhs += self.rand_adv_tracer(-torch.einsum('kl,...lij->...kij', self.A, self.p), \
                    self.u_noise/self.dx, self.v_noise/self.dy) 
        if self.rand_model == 'enstrophy_preserve':
            # Advection of PV
            rhs += self.rand_adv_tracer(self.q_over_f0, self.u_noise/self.dx, self.v_noise/self.dy) 
        return rhs


    def compute_time_derivatives(self):
        # advect vorticity inside of the domain
        dq_over_f0 = F.pad(self.advection_rhs(), (1,1,1,1))

        # Solve helmoltz eq for pressure
        rhs_helmoltz = torch.einsum('ml,...lij->...mij', self.Cl2m, dq_over_f0[...,1:-1,1:-1])
        dp_modes = F.pad(self.inverse_elliptic_dst(rhs_helmoltz, self.helmoltz_dst), (1,1,1,1))

        # Ensure mass conservation
        dalpha =  (self.alpha_matrix @ dp_modes[...,:-1,:,:].mean((-2,-1)).unsqueeze(-1)).unsqueeze(-1)
        dp_modes[...,:-1,:,:] += dalpha * self.homogeneous_sol
        dp = torch.einsum('lm,...mij->...lij', self.Cm2l, dp_modes)

        # update voriticity on the boundaries
        dp_bound = torch.cat([dp[...,0,1:-1], dp[...,-1,1:-1], dp[...,:,0], dp[...,:,-1]], dim=-1)
        delta_p_bound = self.laplacian_h_boundaries(dp, self.zfbc) / (self.f0*self.dx)**2
        dq_over_f0_bound = delta_p_bound - torch.einsum('kl,...lj->...kj', self.A, dp_bound)
        dq_over_f0[...,0,1:-1] = dq_over_f0_bound[...,:self.ny-2]
        dq_over_f0[...,-1,1:-1] = dq_over_f0_bound[...,self.ny-2:2*self.ny-4]
        dq_over_f0[...,0] = dq_over_f0_bound[...,2*self.ny-4:self.nx+2*self.ny-4]
        dq_over_f0[...,-1] = dq_over_f0_bound[...,self.nx+2*self.ny-4:2*self.nx+2*self.ny-4]

        return dq_over_f0, dp


    def step(self):
        """ Time itegration with SSPRK3 scheme.""" 
        dt0_q_over_f0, dt0_p = self.compute_time_derivatives()
        self.q_over_f0 += self.dt * dt0_q_over_f0
        self.p += self.dt * dt0_p

        dt1_q_over_f0, dt1_p = self.compute_time_derivatives()
        self.q_over_f0 += (self.dt/4) * (dt1_q_over_f0 - 3*dt0_q_over_f0)
        self.p += (self.dt/4) * (dt1_p - 3*dt0_p)

        dt2_q_over_f0, dt2_p = self.compute_time_derivatives()
        self.q_over_f0 += (self.dt/12) * (8*dt2_q_over_f0 - dt1_q_over_f0 - dt0_q_over_f0)
        self.p += (self.dt/12) * (8*dt2_p - dt1_p - dt0_p)
        

    def noise_generator(self, time=None):
        
        # EOF-based noise
        if self.base_noise == 'eof':
            rn = torch.randn((self.n_ens,self.u_modes.shape[0]), **self.arr_kwargs)
            self.u_noise = torch.einsum('rm,mkij->rkij', rn, self.u_modes)
            self.v_noise = torch.einsum('rm,mkij->rkij', rn, self.v_modes)
        
        # DMD-based noise
        if self.base_noise == 'dmd':
            rn = torch.randn((self.n_ens,len(self.omega)), dtype=torch.complex128, device=self.device)
            rn *= torch.exp(1j*self.omega*time) 
            self.u_noise = torch.einsum('rm,mkij->rkij', rn, self.u_modes).real
            self.v_noise = torch.einsum('rm,mkij->rkij', rn, self.v_modes).real
        
        # Wavelet-based noise
        if self.base_noise == 'dwt':
            # Local fluct. of streamfunction
            psi = self.p / self.f0
            psi_ = F.pad(psi, self.filt_pad_size, mode='replicate')
            psi -= F.conv2d(psi_.reshape((-1,)+psi_.shape[-2:]).unsqueeze(1), self.filt).squeeze(1).reshape(self.p_shape)
            # Randomize wavelet decomp. of psi
            psi_low, psi_high = self.dwt2(psi[...,:-1,:-1])
            psi_low *= torch.randn((self.n_ens,1,)+psi_low.shape[-2:], **self.arr_kwargs)
            for l in range(self.wave_level):
                psi_high[l] *= torch.randn((self.n_ens,1,3,)+psi_high[l].shape[-2:], **self.arr_kwargs) 
            # Build random psi then derive nondivergent velocity noise
            psi_noise = F.pad(self.idwt2((psi_low, psi_high))[...,1:,1:], (1,1,1,1))
            self.u_noise, self.v_noise = self.compute_velocity(psi_noise)  
        
        # Include Girsanov drift
        if self.mean_drift:
            self.u_noise -= self.u_drift 
            self.v_noise -= self.v_drift
            if self.cyclic_drift:
                rn = torch.exp(1j*self.omega_drift*time).unsqueeze(0)
                self.u_noise -= torch.einsum('rm,mkij->rkij', rn, self.u_modes_drift).real
                self.v_noise -= torch.einsum('rm,mkij->rkij', rn, self.v_modes_drift).real


    def rand_adv_momentum(self):
        """Random advection of momentum using centered flux form."""
        u, v = self.compute_velocity(self.p/self.f0) 
        unoi_on_v, vnoi_on_u = self.interp_xy(self.u_noise), self.interp_xy(self.v_noise)
        
        # Adevction of u
        fx = self.interp_x(u*self.u_noise)
        fy = self.interp_x(self.v_noise) * F.pad(self.interp_y(u[...,1:-1,:]), (1,1))
        du = - torch.diff(fx, dim=-2)/self.dx - torch.diff(fy, dim=-1)/self.dy

        # Advection of v
        fy = self.interp_y(v*self.v_noise)
        fx = self.interp_y(self.u_noise) * F.pad(self.interp_x(v[...,1:-1]), (0,0,1,1))
        dv = - torch.diff(fx, dim=-2)/self.dx - torch.diff(fy, dim=-1)/self.dy
        
        # Add beta effects
        return du + self.betay_on_u * vnoi_on_u, dv - self.betay_on_v * unoi_on_v


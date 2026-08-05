"""
RUN STOCHASTIC MODEL
"""
import numpy as np
import torch
import torch.nn.functional as F
from QGM import QGM, Parameters
torch.backends.cudnn.deterministic = True

################################################################
# SET RUN PARAMETERS
################################################################
#
regime = 'eddy_permitting' # 'eddy_resolving' or 'eddy_permitting' or 'non_eddy_resolving'
#restart_file = 'lastday_5km.npz' # set '' if start from zero states
restart_file = ''
run_years = 10 # run length (years)
freq_plot_days = 15 # plotting frequency (days)
freq_save_days = 15 # data saving frequency (days)
freq_log_days = 5 # logout frequency (days)
output_dir = './output' # output directory

bulk_frc = 'bulk_frc'

# Random param. for noise
rand_model = 'enstrophy_preserve' # diff. random conservation: 'energy_preserve' or 'enstrophy_preserve'
#base_noise = 'dwt' # type of noise: 'fourier' (GMC)
base_noise = 'fourier' # type of noise: 'fourier' (GMC)

################################################################
# SET MODEL (DOES NOT REQUIRE ALTERING IN THE FOLLOWING)
################################################################
#clea
if regime == 'eddy_resolving':
    # High-resolution model which fully resolved mesoscale eddies
    nx, ny = 768+1, 960+1 # corresponding to a 5km horizontal grid spacing
    dt = 1800. # timestep (s)
    a4 = 2e9 # biharmonic viscosity (m^4/s)
             # Here, nondim. Munk scale (a4/beta)^(1/5)/dx = 2
if regime == 'eddy_permitting':
    # Mid-resolution model which partially resolved mesoscale eddies
    #nx, ny = 240+1, 240+1 # dx = dy = 20km
    nx, ny = 256, 256 # dx = dy = 20km
    dx = dy = 20e3 # horizontal grid spacing (km)
    dt = 7200. 
    a4 = 5e10 # Munk scale = 1 
if regime == 'non_eddy_resolving':
    # Low-resolution model which didn't resolve mesoscale eddies
    nx, ny = 96+1, 120+1 # 40 km
    dt = 14400.
    a4 = 5e11 # Munk scale = 0.8

if base_noise in ('eof','dmd'):
    file_modes = base_noise + f'.npz'
else:
    file_modes = None

Lx, Ly = 4800e3, 4800e3 # domain size (km)
A_sp = 0.024
#C_tau = 1 / (np.sqrt(2) * (10 ** 3 / max(Lx,Ly)) ** (1 / 3))
C_tau = 1
print('The value of Ctau is', C_tau)

param = {
    'n_ens': 1, # ensemble size
    'nl': 3, # number of layers
    'Lx': Lx, # zonal length (m)
    'Ly': Ly, # meridional length (m)
    'H': [350., 750., 2900.], # layer thickness (m)
    'g_prime': [0.025, 0.0125], # reduced gravities (m/s^2)
    'f0': 9.37456e-5, # Coriolis param. (s^-1) at 40deg (north) lat.
    'beta': 0, # Coriolis gradient (m^-1/s)
    'delta_ek': 2.0, # bottom Ekman layer thickness (m)
    'tau0': 2.0e-4, # wind stress magnitude (m/s^2)
    'a2': 0., # laplacian diffusion coef (m^2/s)
    'alpha_bc': 0.2, # boundary condition coef. (non-dim.)
    'nx': nx, # zonal pressure grid points
    'ny': ny, # meridional pressure grid points
    'dt': dt, # timestep (s) 
    'a4': a4, # biharmonic viscosity (m^4/s)
    'A_sp' : A_sp, # amplitude of tau (k) spectra
    'C_tau' : C_tau, # tau (k) amplitude 
    'device': 'cuda' if torch.cuda.is_available() else 'cpu', # 'cuda' or 'cpu'
}

param.update({
        'rand_model': rand_model, 
        'base_noise': base_noise, 
        'file_modes': file_modes, # Noise modes (used for EOF/DMD noise)
        'wave_level': 4, # wavelet decomposition levels (only used for DWT nosie)
        'noise_level': 500, # energy injected to fluctuations (only used for DWT nosie)
})

print('==================================================')
print(f'run stochastic {regime} model'.upper())
print('==================================================\n')
print(param)
cfg = Parameters()
qgm = QGM(param, cfg)

################################################################
# INITIALIZATION
################################################################

# Set time parameters
n_steps = int(run_years*365*24*3600/dt)+1 # simulation steps
freq_log = int(freq_log_days*24*3600/dt) 
freq_plot = int(freq_plot_days*24*3600/dt) 
freq_save = int(freq_save_days*24*3600/dt)
freq_checknan = 1000
n_steps_save = 0
t = 0.

# Input restart file
if restart_file:
    data = np.load(restart_file)
    t = float(data['t'])
    p_restart = torch.from_numpy(data['p']).type(torch.float64).to(qgm.device)
    data.close()
    if len(p_restart.shape)<4:
        p_restart = p_restart.tile(qgm.n_ens,1,1,1)
    qgm.reset_p(p_restart)

# Initialize outputs
if freq_save > 0:
    import os
    os.makedirs(output_dir) if not os.path.isdir(output_dir) else None
    filename = os.path.join(output_dir, 'param.pth')
    torch.save(param, filename)
    filename = os.path.join(output_dir, f'pq_{n_steps_save}.npz')
    # Write out time, dynamic pressure and (normalized) potential vorticity in single precision
    np.savez(filename, t=t/(365*86400), p=qgm.p.cpu().numpy().astype('float32'), \
                       q=qgm.q_over_f0.cpu().numpy().astype('float32'))
    n_steps_save += 1

# Initialize figures
if freq_plot > 0:
    import matplotlib.pyplot as plt
     

    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    plt.ion()
    f,a = plt.subplots(1,2)
    a[0].set_title('Surface dynamic pressure ($p$)')
    a[1].set_title('Surface potential vorticity ($q/f_0$)')
    n_years, n_days = int(t//(365*24*3600)), int(t%(365*24*3600)//(24*3600))
    p = (qgm.p[0,0]).cpu().numpy()
    q = (qgm.q_over_f0[0,0]).cpu().numpy()
    a[0].imshow(p.T, cmap='RdBu_r', origin='lower', vmin=-5, vmax=5, animated=True)
    a[1].imshow(q.T, cmap='RdBu_r', origin='lower', vmin=-.5, vmax=.5, animated=True)
    plt.suptitle(f't = {n_years:03d} yrs {n_days:03d} days')
    plt.tight_layout()
    plt.pause(0.1)

################################################################
# TIME-STEPPING
################################################################
#

for n in range(1,n_steps+1):
    qgm.noise_generator(t)
    qgm.step()
    t += dt

    if n % freq_checknan == 0 and torch.isnan(qgm.p).any():
        raise ValueError('Stopping, NAN number in p at iteration {n}.')

    if freq_plot > 0 and n % freq_plot == 0:
        n_years, n_days = int(t//(365*24*3600)), int(t%(365*24*3600)//(24*3600))
        p = (qgm.p[0,0]).cpu().numpy()
        q = (qgm.q_over_f0[0,0]).cpu().numpy()
        a[0].imshow(p.T, cmap='RdBu_r', origin='lower', vmin=-5, vmax=5, animated=True)
        a[1].imshow(q.T, cmap='RdBu_r', origin='lower', vmin=-.5, vmax=.5, animated=True)
        plt.suptitle(f't = {n_years:03d} yrs {n_days:03d} days')
        plt.pause(0.5)

    if freq_log > 0 and n % freq_log == 0:
        p, q = qgm.p.cpu().numpy(), qgm.q_over_f0.cpu().numpy()
        log_str = f'{n=:06d}, t={t/(365*24*60**2):.3f} yr, ' \
                  f'p: ({p.mean():+.1E}, {np.abs(p).max():.2E}) m^2/s^2, ' \
                  f'q: ({q.mean():+.1E}, {np.abs(q).max():.2E}) f0.'
        print(log_str)
    
    if freq_save > 0 and n > n_steps_save and n % freq_save == 0:
        filename = os.path.join(output_dir, f'pq_{n_steps_save}.npz')
        np.savez(filename, t=t/(86400*365), p=qgm.p.cpu().numpy().astype('float32'), \
                           q=qgm.q_over_f0.cpu().numpy().astype('float32'),		 ut=qgm.u.cpu().numpy().astype('float32'),vt=qgm.v.cpu().numpy().astype('float32'), un=qgm.u_noise.cpu().numpy().astype('float32'),vn=qgm.v_noise.cpu().numpy().astype('float32'))        

        n_steps_save += 1
        if n % (10*freq_save) == 0:
            print(f'saved p to {filename}')

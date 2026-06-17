#!/usr/bin/env python3

# Load modules
# ------------
import numpy as np
from numpy import linalg as LA
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
from netCDF4 import Dataset
plt.ion()

# Set parameters
mode = 'o' # 'o' or 'a'
dx = 20. # (km)
dx = float(input("Choose grid spacing dx:\n"))
iodir = '/media/loli/Long/q-gcm/double_gyre_ocean_only/REF5/data1d/'
tau = 1800. if mode=='o' else 600. # (s)
infile = r'ocdmd%d_f200.nc'%(int(dx)) if mode=='o' else r'atdmd%d.nc'%(int(dx))
outfile = r'ocludat%d_dmd.nc'%(int(dx)) if mode=='o' else r'atludat%d_dmd.nc'%(int(dx))
freq_cut = np.pi/(30.*86400.) # cutoff correlated/uncorrelated (s) 
freq_cut = 5e-6
rescale = True
npzdir = '../'

print('Read (full) DMD data')
fin = Dataset(iodir + infile, 'r')
t = fin.variables['time'][:].data
z = fin.variables['z'][:].data
hw = fin.variables['hco'][:].data
xp = fin.variables['xp'][:].data 
yp = fin.variables['yp'][:].data
xt = fin.variables['xt'][:].data 
yt = fin.variables['yt'][:].data
um = fin.variables['umean'][...].data
vm = fin.variables['vmean'][...].data
u0 = fin.variables['uinit'][...].data
v0 = fin.variables['vinit'][...].data
re = fin.variables['lambda_real'][:].data
im = fin.variables['lambda_imag'][:].data
lam = re + 1j*im
re = fin.variables['omega_real'][:].data 
im = fin.variables['omega_imag'][:].data 
omg = re + 1j*im
re = fin.variables['b_real'][:].data 
im = fin.variables['b_imag'][:].data 
amp = re + 1j*im
dt = (t[1]-t[0])*365.*86400.
nx, ny, nz = len(xp), len(yp), len(z)

print('Plot (origin) eigenvalues and amptitudes')
plt.figure(figsize=(5,5))
plt.plot(lam.real, lam.imag, '.')
plt.grid(which='both', axis='both')
plt.xlabel(r'Re ($\lambda$)')
plt.ylabel(r'Im ($\lambda$)')
plt.savefig('eval.pdf', bbox_inches='tight', pad_inches=0)

amp_mod = np.abs(amp)
plt.figure(figsize=(5,5))
plt.plot(1.0e5*omg.imag, amp_mod, '+')
plt.grid(which='both', axis='both')
plt.xlabel(r'Im ($\omega$) $\times 10^{-5}$')
plt.ylabel(r'amplitude')
plt.savefig('amp.pdf', bbox_inches='tight', pad_inches=0)

print('Filter eigenvalues (kill those not on unit cercle)')
lam_mod = np.abs(lam)
idm = np.where((lam_mod > 1.0 - 1.0e-2) & (lam_mod <= 1.0 + 1.0e-3))
mask1 = np.full(len(lam_mod), False, dtype=bool)
mask1[idm] = True
lam, omg, amp, amp_mod = lam[mask1], omg[mask1], amp[mask1], amp_mod[mask1]
del lam_mod

plt.figure(figsize=(5,5))
plt.plot(lam.real, lam.imag, '.')
plt.grid(which='both', axis='both')
plt.xlabel(r'Re ($\lambda$)')
plt.ylabel(r'Im ($\lambda$)')
plt.savefig('eval_filt.pdf', bbox_inches='tight', pad_inches=0)

plt.figure(figsize=(5,5))
plt.plot(1.0e5*omg.imag, amp_mod, '+')
plt.grid(which='both', axis='both')
plt.xlabel(r'Im ($\omega$) $\times 10^{-5}$')
plt.ylabel(r'amplitude')
plt.savefig('amp_filt.pdf', bbox_inches='tight', pad_inches=0)

print('Truncate modes by energy')
amp_cut = float(input("Set trucation amplitude:\n"))
idm = np.where((amp_mod >= amp_cut) & (lam.imag != 0.))
mask2 = np.full(len(amp_mod), False, dtype=bool)
mask2[idm] = True
lam, omg, amp, amp_mod = lam[mask2], omg[mask2], amp[mask2], amp_mod[mask2]
nm = len(amp_mod)
print(f'Number of modes used = {nm}')

plt.figure(figsize=(5,5))
plt.plot(lam.real, lam.imag, '.')
plt.xlim([-1.1,1.1])
plt.ylim([-1.1,1.1])
plt.grid(which='both', axis='both')
plt.xlabel(r'Re ($\lambda$)')
plt.ylabel(r'Im ($\lambda$)')
plt.savefig('eval_filt_cut.pdf', bbox_inches='tight', pad_inches=0)

plt.figure(figsize=(5,5))
plt.plot(1.0e5*omg.imag, amp_mod, '+')
plt.grid(which='both', axis='both')
plt.xlabel(r'Im ($\omega$) $\times 10^{-5}$')
plt.ylabel(r'amplitude')
plt.savefig('amp_filt_cut.pdf', bbox_inches='tight', pad_inches=0)

nu, nv = nz*(ny-1)*nx, nz*ny*(nx-1)
re = np.zeros((nu+nv,nm), dtype=np.float64)
re[:nu,:] = fin.variables['umode_real'][mask1][mask2].data.reshape((nm,nu)).T
re[nu:,:] = fin.variables['vmode_real'][mask1][mask2].data.reshape((nm,nv)).T
im = np.zeros((nu+nv,nm), dtype=np.float64)
im[:nu,:] = fin.variables['umode_imag'][mask1][mask2].data.reshape((nm,nu)).T
im[nu:,:] = fin.variables['vmode_imag'][mask1][mask2].data.reshape((nm,nv)).T
fin.close()
Phi = re + 1j*im
del re,im
if rescale:
    print('Rescale truncated amplitudes')
    X0 = u0.flatten() 
    X0 = np.append(X0, v0.flatten())
    del u0,v0
    Psi = Phi @ LA.pinv(Phi.conj().T @ Phi)
    amp = Psi.conj().T @ X0
    amp_mod = np.abs(amp)
Phi *= amp[None]

print('Seperate temporal scales')
freq_cut = float(input("Set seperation scale:\n"))
idm = np.where(np.abs(omg.imag) <= freq_cut)
mask3 = np.full(len(omg.imag), False, dtype=bool)
mask3[idm] = True
Phi_corr, Phi_rand = Phi[:,mask3], Phi[:,~mask3]
lam_corr, lam_rand = lam[mask3], lam[~mask3]
omg_corr, omg_rand = omg[mask3], omg[~mask3]
amp_corr, amp_rand = amp_mod[mask3], amp_mod[~mask3]
del Phi,lam,omg,amp_mod

plt.figure(figsize=(5,5))
plt.plot(lam_corr.real, lam_corr.imag, '.', label='correlated')
plt.plot(lam_rand.real, lam_rand.imag, '.', label='uncorrelated')
plt.xlim([-1.1,1.1])
plt.ylim([-1.1,1.1])
plt.grid(which='both', axis='both')
plt.legend(loc='best')
plt.xlabel(r'Re ($\lambda$)')
plt.ylabel(r'Im ($\lambda$)')
plt.savefig('eval_filt_cut_res_sep.pdf', bbox_inches='tight', pad_inches=0)

plt.figure(figsize=(5,5))
plt.plot(1.0e5*omg_corr.imag, amp_corr, '+', label='correlated')
plt.plot(1.0e5*omg_rand.imag, amp_rand, '+', label='uncorrelated')
plt.grid(which='both', axis='both')
plt.legend(loc='best')
plt.xlabel(r'Im ($\omega$) $\times 10^{-5}$')
plt.ylabel(r'amplitude')
plt.savefig('amp_filt_cut_res_sep.pdf', bbox_inches='tight', pad_inches=0)

print('Vertical normalization')
nmc, nmr = len(omg_corr), len(omg_rand)
umo_corr = ( Phi_corr[:nu,:].T.reshape((nmc,nz,ny-1,nx)) )/( np.sqrt(hw)[None,:,None,None] ) 
vmo_corr = ( Phi_corr[nu:,:].T.reshape((nmc,nz,ny,nx-1)) )/( np.sqrt(hw)[None,:,None,None] ) 
umo_rand = ( Phi_rand[:nu,:].T.reshape((nmr,nz,ny-1,nx)) )/( np.sqrt(hw)[None,:,None,None] ) 
vmo_rand = ( Phi_rand[nu:,:].T.reshape((nmr,nz,ny,nx-1)) )/( np.sqrt(hw)[None,:,None,None] ) 
del Phi_corr,Phi_rand

print('Write outputs')
fout = Dataset(iodir + outfile, 'w', format='NETCDF4')
# Create dimensions
fout.createDimension('mode_c', nmc)
fout.createDimension('mode_r', nmr)
fout.createDimension('z', nz)
fout.createDimension('yp', ny)
fout.createDimension('xp', nx)
fout.createDimension('yt', ny-1)
fout.createDimension('xt', nx-1)
# Create variables
fout.createVariable('mode_c', 'i4', ('mode_c',))
fout.createVariable('mode_r', 'i4', ('mode_r',))
fout.createVariable('z', 'f8', ('z',))
fout.createVariable('yp', 'f8', ('yp',))
fout.createVariable('xp', 'f8', ('xp',))
fout.createVariable('yt', 'f8', ('yt',))
fout.createVariable('xt', 'f8', ('xt',))
fout.createVariable(r'uc%s'%(mode), 'f8', ('z','yt','xp',))
fout.createVariable(r'vc%s'%(mode), 'f8', ('z','yp','xt',))
fout.createVariable('omega_c', 'f8', ('mode_c',))
fout.createVariable('omega_r', 'f8', ('mode_r',))
fout.createVariable('umode_real_c', 'f8', ('mode_c','z','yt','xp',))
fout.createVariable('umode_imag_c', 'f8', ('mode_c','z','yt','xp',))
fout.createVariable('vmode_real_c', 'f8', ('mode_c','z','yp','xt',))
fout.createVariable('vmode_imag_c', 'f8', ('mode_c','z','yp','xt',))
fout.createVariable('umode_real_r', 'f8', ('mode_r','z','yt','xp',))
fout.createVariable('umode_imag_r', 'f8', ('mode_r','z','yt','xp',))
fout.createVariable('vmode_real_r', 'f8', ('mode_r','z','yp','xt',))
fout.createVariable('vmode_imag_r', 'f8', ('mode_r','z','yp','xt',))
fout.createVariable('axx', 'f8', ('z','yt','xp',))
fout.createVariable('ayy', 'f8', ('z','yp','xt',))
fout.createVariable('axy', 'f8', ('z','yp','xp',))
# Add attributes
fout.variables['mode_c'].long_name = 'Mode index (for correction drift)'
fout.variables['mode_r'].long_name = 'Mode index (for random noise)'
fout.variables['z'].long_name = 'Mid-layer axis'
fout.variables['z'].units = 'km'
fout.variables['yp'].long_name = 'Y axis (p-grid)'
fout.variables['yp'].units = 'km'
fout.variables['xp'].long_name = 'X axis (p-grid)'
fout.variables['xp'].units = 'km'
fout.variables['yt'].long_name = 'Y axis (T-grid)'
fout.variables['yt'].units = 'km'
fout.variables['xt'].long_name = 'X axis (T-grid)'
fout.variables['xt'].units = 'km'
fout.variables['omega_c'].long_name = 'Continuous eigenvalues for correction drift'
fout.variables['omega_c'].units = 's^-1'
fout.variables['omega_r'].long_name = 'Continuous eigenvalues for random noise'
fout.variables['omega_r'].units = 's^-1'
fout.variables[r'uc%s'%(mode)].long_name = 'Mean of zonal correction drift'
fout.variables[r'uc%s'%(mode)].units = 'm/s'
fout.variables[r'vc%s'%(mode)].long_name = 'Mean of meridional correction drift'
fout.variables[r'vc%s'%(mode)].units = 'm/s'
fout.variables['umode_real_c'].long_name = 'Zonal DMD modes for correction drift (real part)'
fout.variables['umode_real_c'].units = 'm/s'
fout.variables['vmode_real_c'].long_name = 'Meridional DMD modes for correction drift (real part)'
fout.variables['vmode_real_c'].units = 'm/s'
fout.variables['umode_imag_c'].long_name = 'Zonal DMD modes for correction drift (imag part)'
fout.variables['umode_imag_c'].units = 'm/s'
fout.variables['vmode_imag_c'].long_name = 'Meridional DMD modes for correction drift (imag part)'
fout.variables['vmode_imag_c'].units = 'm/s'
fout.variables['umode_real_r'].long_name = 'Zonal DMD modes for random noise (real part)'
fout.variables['umode_real_r'].units = 'm/s'
fout.variables['vmode_real_r'].long_name = 'Meridional DMD modes for random noise (real part)'
fout.variables['vmode_real_r'].units = 'm/s'
fout.variables['umode_imag_r'].long_name = 'Zonal DMD modes for random noise (imag part)'
fout.variables['umode_imag_r'].units = 'm/s'
fout.variables['vmode_imag_r'].long_name = 'Meridional DMD modes for random noise (imag part)'
fout.variables['vmode_imag_r'].units = 'm/s'
fout.variables['axx'].long_name = 'Variance of zonal noise'
fout.variables['axx'].units = 'm^2/s'
fout.variables['ayy'].long_name = 'Variance of meridional noise'
fout.variables['ayy'].units = 'm^2/s'
fout.variables['axy'].long_name = 'Cross variance of noise'
fout.variables['axy'].units = 'm^2/s'
# Write data
fout.variables['mode_c'][:] = range(nmc)
fout.variables['mode_r'][:] = range(nmr)
fout.variables['z'][:] = z
fout.variables['yp'][:] = yp
fout.variables['xp'][:] = xp
fout.variables['yt'][:] = yt
fout.variables['xt'][:] = xt
fout.variables['omega_c'][:] = omg_corr.imag
fout.variables['omega_r'][:] = omg_rand.imag
fout.variables[r'uc%s'%(mode)][...] = um 
fout.variables[r'vc%s'%(mode)][...] = vm 
fout.variables['umode_real_c'][...] = umo_corr.real 
fout.variables['vmode_real_c'][...] = vmo_corr.real
fout.variables['umode_imag_c'][...] = umo_corr.imag
fout.variables['vmode_imag_c'][...] = vmo_corr.imag
fout.variables['umode_real_r'][...] = umo_rand.real 
fout.variables['vmode_real_r'][...] = vmo_rand.real
fout.variables['umode_imag_r'][...] = umo_rand.imag
fout.variables['vmode_imag_r'][...] = vmo_rand.imag
fout.variables['axx'][...] = tau*np.sum((umo_rand.conj()*umo_rand).real.astype('float64'), axis=0)
fout.variables['ayy'][...] = tau*np.sum((vmo_rand.conj()*vmo_rand).real.astype('float64'), axis=0)
fout.variables['axy'][...] = np.zeros((nz,ny,nx), dtype=np.float64)
ctmp = (umo_rand[...,:-1,1:-1] + umo_rand[...,1:,1:-1]).conj() \
      *(vmo_rand[...,1:-1,:-1] + vmo_rand[...,1:-1,1:])
fout.variables['axy'][:,1:-1,1:-1] = 0.25*tau*np.sum( ctmp.real.astype('float64'), axis=0 )
# Close file
fout.close()

filename = os.path.join(npzdir, f'dmd_{int(dx)}km.npz')
np.savez(filename, omega=omg_rand.imag, u=umo_rand.transpose(0,1,3,2), v=vmo_rand.transpose(0,1,3,2))

filename = os.path.join(npzdir, f'cyclic_drift_{int(dx)}km.npz')
np.savez(filename, um=um.transpose(0,2,1), vm=vm.transpose(0,2,1), omega=omg_corr.imag, \
                   u=umo_corr.transpose(0,1,3,2), v=vmo_corr.transpose(0,1,3,2))

print('Program terminates')


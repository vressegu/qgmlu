# Load modules
# ------------
import numpy as np
from numpy import linalg as LA
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Set parammeters
mode = 'oc' # 'oc' or 'at'
dx = 20.0 # (km)
mdir = '/media/loli/Long/q-gcm/double_gyre_ocean_only/REF5/data1d/'
sdir = ['yrs105-106/','yrs106-107/','yrs107-108/','yrs108-109/','yrs109-110/',
        'yrs110-111/','yrs111-112/','yrs112-113/','yrs113-114/','yrs114-115/']
infile = r'%sfluc%d_f200.nc'%(mode,int(dx))
outfile = r'%sdmd%d_f200.nc'%(mode,int(dx))
h = [350.0,750.0,2900.0] if mode=='oc' else [2.e3,3.e3,4.e3] # Layer thickness (m)

# Read axis
fin = Dataset(mdir + sdir[0] + infile, 'r')
xp = fin.variables['xp'][:].data # P-grid axis (km)
yp = fin.variables['yp'][:].data
xt = fin.variables['xt'][:].data # T-grid axis (km)
yt = fin.variables['yt'][:].data
z  = fin.variables['z'][:].data # vertical axis (km)
fin.close()
nx, ny, nz = len(xp), len(yp), len(z)
hfc = h/np.sum(h)

print('Collect snapshots')
n = 1
t = np.empty(0) 
u = np.empty((0,nz,ny-1,nx))
v = np.empty((0,nz,ny,nx-1))
for s in sdir:
    file1 = mdir + s + infile 
    print(f'Opening file: {file1}')
    fin = Dataset(file1,'r')
    nn = np.minimum(n,2) - 1
    tmp = fin.variables['time'][nn:].copy()
    t = np.append(t, tmp, axis=0) 
    tmp = fin.variables['ur'][nn:,...].copy()
    u = np.append(u, tmp, axis=0)
    tmp = fin.variables['vr'][nn:,...].copy()
    v = np.append(v, tmp, axis=0)
    del tmp
    n += 1
    fin.close() 
nt, dt = len(t), (t[1]-t[0])*365.*86400. # (s)

# Init. output file
file1 = mdir + outfile 
print(f'Create file: {file1}')
fout = Dataset(file1, 'w', format='NETCDF4')
# Create dimensions
fout.createDimension('time', nt)
fout.createDimension('mode', nt-1)
fout.createDimension('yt', ny-1)
fout.createDimension('xt', nx-1)
fout.createDimension('yp', ny)
fout.createDimension('xp', nx)
fout.createDimension('z', nz)
# Create variables
fout.createVariable('time', 'f8', ('time',))
fout.createVariable('yt', 'f8', ('yt',))
fout.createVariable('xt', 'f8', ('xt',))
fout.createVariable('yp', 'f8', ('yp',))
fout.createVariable('xp', 'f8', ('xp',))
fout.createVariable('z', 'f8', ('z',))
fout.createVariable('hco', 'f8', ('z',))
fout.createVariable('lambda_real', 'f8', ('mode',))
fout.createVariable('lambda_imag', 'f8', ('mode',))
fout.createVariable('omega_real', 'f8', ('mode',))
fout.createVariable('omega_imag', 'f8', ('mode',))
fout.createVariable('b_real', 'f8', ('mode',))
fout.createVariable('b_imag', 'f8', ('mode',))
fout.createVariable('umean', 'f8', ('z','yt','xp',))
fout.createVariable('vmean', 'f8', ('z','yp','xt',))
fout.createVariable('uinit', 'f8', ('z','yt','xp',))
fout.createVariable('vinit', 'f8', ('z','yp','xt',))
fout.createVariable('umode_real', 'f8', ('mode','z','yt','xp',))
fout.createVariable('umode_imag', 'f8', ('mode','z','yt','xp',))
fout.createVariable('vmode_real', 'f8', ('mode','z','yp','xt',))
fout.createVariable('vmode_imag', 'f8', ('mode','z','yp','xt',))
# Add attributes
fout.variables['time'].long_name = 'Time axis'
fout.variables['time'].units = 'years'
fout.variables['yt'].long_name = 'Y axis (T-grid)'
fout.variables['yt'].units = 'km'
fout.variables['xt'].long_name = 'X axis (T-grid)'
fout.variables['xt'].units = 'km'
fout.variables['yp'].long_name = 'Y axis (p-grid)'
fout.variables['yp'].units = 'km'
fout.variables['xp'].long_name = 'X axis (p-grid)'
fout.variables['xp'].units = 'km'
fout.variables['z'].long_name = 'Mid-layer axis'
fout.variables['z'].units = 'km'
fout.variables['hco'].long_name = 'Layer weight'
fout.variables['umean'].long_name = 'Mean of zonal velocity'
fout.variables['umean'].units = 'm/s'
fout.variables['vmean'].long_name = 'Mean of meridional velocity'
fout.variables['vmean'].units = 'm/s'
fout.variables['uinit'].long_name = 'Initial zonal velocity'
fout.variables['uinit'].units = 'm/s'
fout.variables['vinit'].long_name = 'Initial meridional velocity'
fout.variables['vinit'].units = 'm/s'
fout.variables['lambda_real'].long_name = 'DMD eigenvalues (real part)'
fout.variables['lambda_imag'].long_name = 'DMD eigenvalues (imag part)'
fout.variables['omega_real'].long_name = 'Continuous eigenvalues (real part)'
fout.variables['omega_imag'].long_name = 'Continuous eigenvalues (imag part)'
fout.variables['b_real'].long_name = 'DMD amplitudes (real part)'
fout.variables['b_imag'].long_name = 'DMD amplitudes (imag part)'
fout.variables['umode_real'].long_name = 'Zonal DMD modes (real part)'
fout.variables['vmode_real'].long_name = 'Meridional DMD modes (real part)'
fout.variables['umode_imag'].long_name = 'Zonal DMD modes (imag part)'
fout.variables['vmode_imag'].long_name = 'Meridional DMD modes (imag part)'
# Write data
fout.variables['time'][:] = t
fout.variables['yt'][:] = yt
fout.variables['xt'][:] = xt
fout.variables['yp'][:] = yp
fout.variables['xp'][:] = xp
fout.variables['z'][:] = z
fout.variables['hco'][:] = hfc

print('Build data matrix')
um = np.mean(u, axis=0, dtype=np.float64)
vm = np.mean(v, axis=0, dtype=np.float64)
u -= um[np.newaxis] 
v -= vm[np.newaxis] 
fout.variables['umean'][...] = um
fout.variables['vmean'][...] = vm
del um,vm
for k in range(nz):
    u[:,k,...] *= np.sqrt(hfc[k])
    v[:,k,...] *= np.sqrt(hfc[k])
nu, nv = nz*(ny-1)*nx, nz*ny*(nx-1)
X1 = np.zeros((nu+nv,nt), dtype=np.float64)
X1[:nu,:] = u.reshape((nt,nu)).T 
X1[nu:,:] = v.reshape((nt,nv)).T
fout.variables['uinit'][...] = u[0]
fout.variables['vinit'][...] = v[0]
del u,v

print('Perform DMD procedure')
U1, S1, V1 = LA.svd(X1[:,:-1], full_matrices=False) 
A1 = ( (U1.T @ X1[:,1:]) @ V1 ) @ np.diag(1./S1)
lamb, W1 = LA.eig(A1) # 'lamb' is discrete-time eigenvalues
Phi = ( ( X1[:,1:] @ V1) @ np.diag(1./S1) ) @ W1 # DMD modes
omega = np.log(lamb)/dt # continuous-time eigenvalues
b = LA.pinv(Phi) @ X1[:,0] # DMD amptitude
del X1,U1,S1,V1,A1

fout.variables['lambda_real'][:] = lamb.real
fout.variables['lambda_imag'][:] = lamb.imag
fout.variables['omega_real'][:] = omega.real
fout.variables['omega_imag'][:] = omega.imag
fout.variables['b_real'][:] = b.real
fout.variables['b_imag'][:] = b.imag
fout.variables['umode_real'][...] = (Phi[:nu,:].real.T).reshape((nt-1,nz,ny-1,nx))
fout.variables['vmode_real'][...] = (Phi[nu:,:].real.T).reshape((nt-1,nz,ny,nx-1))
fout.variables['umode_imag'][...] = (Phi[:nu,:].imag.T).reshape((nt-1,nz,ny-1,nx))
fout.variables['vmode_imag'][...] = (Phi[nu:,:].imag.T).reshape((nt-1,nz,ny,nx-1))
del Phi,b,omega,lamb

# Close file
fout.close()

print('Program terminates')

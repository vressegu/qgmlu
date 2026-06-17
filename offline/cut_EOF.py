import os
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Set parammeters
iodir = '/media/loli/Long/q-gcm/double_gyre_ocean_only/REF5/data1d/'
dx = 40.0 # (km)
dt = 1800.0 # (s)
infile = r'ocueof%d_f200.nc'%(int(dx))
outfile = r'ocludat%d_eof.nc'%(int(dx))
proj_mean = False
plot_opt = True
rescaling = True
eporp = 0.9
npzdir = '../'

# Read dimensions and eigenvalues
file1 = iodir + infile
print(f'Open file: {file1}')
fin = Dataset(file1, 'r')
nz = fin.dimensions['z'].size
ny = fin.dimensions['yp'].size
nx = fin.dimensions['xp'].size
lam = fin.variables['eigval'][:].data
nu, nv = nz*(ny-1)*nx, nz*ny*(nx-1)

print('Truncate EOFs of noise')
ric = np.cumsum(lam)/np.sum(lam)
plt.figure()
plt.subplot(211)
plt.plot(lam[:500],'+')
plt.grid(which='both', axis='both')
plt.xlabel('Modes')
plt.ylabel('Eigenvalues')
plt.subplot(212)
plt.plot(ric[:1000])
plt.grid(which='both', axis='both')
plt.xlabel('Modes')
plt.ylabel('Energy proportion')
plt.show()
nm = int(input("Set number of modes (better in even):\n"))
efc = eporp*np.sum(lam)/np.sum(lam[:nm]) if rescaling else 1.
lam = lam[:nm]
lam *= efc

# Create output file
file1 = iodir + outfile 
print(f'Create file: {file1}')
fout = Dataset(file1, 'w', format='NETCDF4')
# Create dimensions
fout.createDimension('mode', nm)
fout.createDimension('yt', ny-1)
fout.createDimension('xt', nx-1)
fout.createDimension('yp', ny)
fout.createDimension('xp', nx)
fout.createDimension('z', nz)
fout.createDimension('uv', nu+nv)
# Create variables
fout.createVariable('yt', 'f8', ('yt',))
fout.createVariable('xt', 'f8', ('xt',))
fout.createVariable('yp', 'f8', ('yp',))
fout.createVariable('xp', 'f8', ('xp',))
fout.createVariable('z', 'f8', ('z',))
fout.createVariable('axxo', 'f8', ('z','yt','xp',))
fout.createVariable('ayyo', 'f8', ('z','yp','xt',))
fout.createVariable('axyo', 'f8', ('z','yp','xp',))
#fout.createVariable('umode', 'f8', ('mode','z','yt','xp',))
#fout.createVariable('vmode', 'f8', ('mode','z','yp','xt',))
fout.createVariable('eofo', 'f8', ('mode','uv',))
#fout.createVariable('ucorr', 'f8', ('z','yt','xp',))
#fout.createVariable('vcorr', 'f8', ('z','yp','xt',))
fout.createVariable('uco', 'f8', ('z','yt','xp',))
fout.createVariable('vco', 'f8', ('z','yp','xt',))
# Add attributes
fout.variables['yt'].long_name = 'Ocean Y axis (T-grid)'
fout.variables['yt'].units = 'km'
fout.variables['xt'].long_name = 'Ocean X axis (T-grid)'
fout.variables['xt'].units = 'km'
fout.variables['yp'].long_name = 'Ocean Y axis (p-grid)'
fout.variables['yp'].units = 'km'
fout.variables['xp'].long_name = 'Ocean X axis (p-grid)'
fout.variables['xp'].units = 'km'
fout.variables['z'].long_name = 'Ocean mid-layer axis'
fout.variables['z'].units = 'km'
fout.variables['axxo'].long_name = 'Variance of zonal velocity'
fout.variables['axxo'].units = 'm^2/s^2'
fout.variables['ayyo'].long_name = 'Variance of meridional velocity'
fout.variables['ayyo'].units = 'm^2/s^2'
fout.variables['axyo'].long_name = 'Covariance of velocity components'
fout.variables['axyo'].units = 'm^2/s^2'
#fout.variables['umode'].long_name = 'Zonal velocity modes'
#fout.variables['umode'].units = 'm/s'
#fout.variables['vmode'].long_name = 'Meridional velocity modes'
#fout.variables['vmode'].units = 'm/s'
fout.variables['eofo'].long_name = 'Eddy velocity modes'
fout.variables['eofo'].units = 'm/s'
fout.variables['uco'].long_name = 'Zonal correlated drift'
fout.variables['uco'].units = 'm/s'
fout.variables['vco'].long_name = 'Meridional correlated drift'
fout.variables['vco'].units = 'm/s'
# Write data
fout.variables['z'][:] = fin.variables['z'][:].data
fout.variables['yp'][:] = fin.variables['yp'][:].data
fout.variables['xp'][:] = fin.variables['xp'][:].data
fout.variables['yt'][:] = fin.variables['yt'][:].data
fout.variables['xt'][:] = fin.variables['xt'][:].data
if proj_mean:
    fout.variables['uco'][...] = -fin.variables['ucorr'][...].data
    fout.variables['vco'][...] = -fin.variables['vcorr'][...].data
else:
    fout.variables['uco'][...] = -fin.variables['umean'][...].data
    fout.variables['vco'][...] = -fin.variables['vmean'][...].data

print('Rescale EOFs')
umo = fin.variables['umode'][:nm,...].data
vmo = fin.variables['vmode'][:nm,...].data
filename = os.path.join(npzdir, f'mean_drift_{int(dx)}km.npz')
np.savez(filename, um=fin.variables['umean'][...].data.transpose(0,2,1), \
                   vm=fin.variables['vmean'][...].data.transpose(0,2,1))
fin.close()
for m in range(nm):
    umo[m,...] *= np.sqrt(lam[m])
    vmo[m,...] *= np.sqrt(lam[m])
filename = os.path.join(npzdir, f'eof_{int(dx)}km.npz')
np.savez(filename, u=umo.transpose(0,1,3,2), v=vmo.transpose(0,1,3,2))
#fout.variables['umode'][...] = umo
#fout.variables['vmode'][...] = vmo
fout.variables['eofo'][:,:nu] = umo.reshape((nm,nu))
fout.variables['eofo'][:,nu:] = vmo.reshape((nm,nv))

print('Build variance tensor')
fout.variables['axxo'][...] = dt*np.sum(umo**2, axis=0)
fout.variables['ayyo'][...] = dt*np.sum(vmo**2, axis=0)
fout.variables['axyo'][...] = np.zeros((nz,ny,nx), dtype=np.float64)
fout.variables['axyo'][:,1:-1,1:-1] = 0.25*dt*np.sum( 
        (umo[...,:-1,1:-1] + umo[...,1:,1:-1])*
        (vmo[...,1:-1,:-1] + vmo[...,1:-1,1:]), axis=0 )
fout.close()

if plot_opt:
    print('Generate EOF noise')
    tco = np.random.randn(nm)
    unoi = (umo.reshape((nm,nu)).T @ tco).reshape((nz,ny-1,nx)) 
    vnoi = (vmo.reshape((nm,nv)).T @ tco).reshape((nz,ny,nx-1)) 
    for k in range(nz):
        print(f'Layer = {k+1}')
        plt.figure()
        plt.subplot(121)
        plt.imshow(unoi[k],origin='lower',cmap='RdBu_r',vmin=-.3,vmax=.3)
        plt.title('Unoise')
        plt.subplot(122)
        plt.imshow(vnoi[k],origin='lower',cmap='RdBu_r',vmin=-.3,vmax=.3)
        plt.title('Vnoise')
        plt.show()
del umo, vmo

print('Program terminates')

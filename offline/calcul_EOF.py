import numpy as np
from numpy import linalg
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def inner_prod(U, V, W):
    '''Inner product of U and V associated with weights W'''
    return (U * W[None]) @ (V.T)

def spot_POD(U, V, wU, wV, nm, check_opt):
    '''Snapshot POD of 2D velocity data'''
    [nt, nu] = np.shape(U)
    nv = np.shape(V[0,:])
    # Detrending
    Um, Vm = np.mean(U, axis=0), np.mean(V, axis=0)
    Uf, Vf = U-Um, V-Vm
    # Build temporal correlation matrix
    C = ( inner_prod(Uf, Uf, wU) + inner_prod(Vf, Vf, wV) )/nt
    # Solve eigen problem
    eigval, eigvec = linalg.eigh(C)
    ids = np.argsort(eigval)[::-1] # sort eigvals in descend order
    PC = eigval[ids]
    Tmod = eigvec[:,ids].T # sort temporal modes
    del C, eigval, eigvec
    # Scaling temporal modes
    PC, Tmod = PC[:nm], Tmod[:nm,:]
    for i in range(nm):
        Tmod[i,:] *= np.sqrt( nt*np.maximum(PC[i], 0.) ) 
    # Build spatial modes
    Umod, Vmod = Tmod @ Uf, Tmod @ Vf
    for i in range(nm):
        Umod[i,:] /= (nt*PC[i]) 
        Vmod[i,:] /= (nt*PC[i]) 
    if check_opt:
        print('Check if spatial modes are orthonormal')
        Cx = inner_prod(Umod, Umod, wU) + inner_prod(Vmod, Vmod, wV) 
        plt.figure()
        plt.imshow(Cx, vmin=0., vmax=1.)
        plt.colorbar()
        plt.xlabel('modes')
        plt.ylabel('modes')
        plt.title('Orthogonality of spatial modes')
        plt.show()
        print('Check if temporal modes are orthogonal')
        Ct = Tmod @ (Tmod.T)
        for m in range(nm):
            Ct[m,m] /= (nt*PC[m])
        plt.figure()
        plt.imshow(Ct, vmin=0., vmax=1.)
        plt.colorbar()
        plt.xlabel('modes')
        plt.ylabel('modes')
        plt.title('Orthogonality of temporal modes')
        plt.show()
        print('Check reconstruction of EOF')
        Uerr, Verr = (Tmod.T) @ Umod, (Tmod.T) @ Vmod
        Uerr -= Uf
        Verr -= Vf
        print(f'Max. error = {np.maximum(abs(Uerr).max(), abs(Verr).max())}')
    return PC, Um, Vm, Umod, Vmod, Tmod 


# Set parammeters
mdir = '/media/loli/Long/q-gcm/double_gyre_ocean_only/REF5/data1d/'
sdir = ['yrs105-106/','yrs106-107/','yrs107-108/','yrs108-109/','yrs109-110/',
        'yrs110-111/','yrs111-112/','yrs112-113/','yrs113-114/','yrs114-115/']
odir = mdir
dx = 40.0 # (km)
infile = r'ocfluc%d_f200.nc'%(int(dx))
outfile = r'ocueof%d_f200.nc'%(int(dx))
f0 = 9.37456e-5 # Coriolis param. (s^-1)
h = [350.0,750.0,2900.0] # Layer thickness (m)
check_opt = False

# Read axis
file1 = mdir + sdir[0] + infile 
fin = Dataset(file1, 'r')
z = fin.variables['z'][:].data
yp = fin.variables['yp'][:].data
xp = fin.variables['xp'][:].data
yt = fin.variables['yt'][:].data
xt = fin.variables['xt'][:].data
nz, ny, nx = len(z), len(yp), len(xp)
fin.close()

print('Collect snapshots')
n = 1
t = np.empty(0) 
u = np.empty((0,nz,ny-1,nx))
v = np.empty((0,nz,ny,nx-1))
for s in sdir:
    file1 = mdir + s + infile 
    print(f'Read file: {file1}')
    fin = Dataset(file1, 'r')
    nn = np.minimum(n,2) - 1
    tmp = fin.variables['time'][nn:].data
    t = np.append(t, tmp, axis=0) 
    tmp = fin.variables['ur'][nn:,...].data
    u = np.append(u, tmp, axis=0)
    tmp = fin.variables['vr'][nn:,...].data
    v = np.append(v, tmp, axis=0)
    del tmp
    n += 1
    fin.close() 
nt = len(t)
nm = nt - 1

# Initialize output file
file1 = odir + outfile 
print(f'Create file: {file1}')
fout = Dataset(file1, 'w', format='NETCDF4')
# Create dimensions
fout.createDimension('time', nt)
fout.createDimension('mode', nm)
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
fout.createVariable('eigval', 'f8', ('mode',))
fout.createVariable('pcoef', 'f8', ('mode',))
fout.createVariable('tmode', 'f8', ('mode','time',))
fout.createVariable('umean', 'f8', ('z','yt','xp',))
fout.createVariable('vmean', 'f8', ('z','yp','xt',))
fout.createVariable('ucorr', 'f8', ('z','yt','xp',))
fout.createVariable('vcorr', 'f8', ('z','yp','xt',))
fout.createVariable('umode', 'f8', ('mode','z','yt','xp',))
fout.createVariable('vmode', 'f8', ('mode','z','yp','xt',))
# Add attributes
fout.variables['time'].long_name = 'Time axis'
fout.variables['time'].units = 'years'
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
fout.variables['eigval'].long_name = 'Eigenvalues of covariance (EKE)'
fout.variables['eigval'].units = 'm^2/s^2'
fout.variables['pcoef'].long_name = 'Projection coefficient'
fout.variables['pcoef'].units = 'm/s'
fout.variables['umean'].long_name = 'Mean zonal velocity'
fout.variables['umean'].units = 'm/s'
fout.variables['vmean'].long_name = 'Mean meridional velocity'
fout.variables['vmean'].units = 'm/s'
fout.variables['ucorr'].long_name = 'Zonal correlated drift'
fout.variables['ucorr'].units = 'm/s'
fout.variables['vcorr'].long_name = 'Meridional correlated drift'
fout.variables['vcorr'].units = 'm/s'
fout.variables['umode'].long_name = 'Zonal velocity modes'
fout.variables['vmode'].long_name = 'Meridional velocity modes'
fout.variables['tmode'].long_name = 'Temporal modes'
# Write data
fout.variables['time'][:] = fin.variables['time'][:].data 
fout.variables['yt'][:] = yt
fout.variables['xt'][:] = xt
fout.variables['yp'][:] = yp
fout.variables['xp'][:] = xp
fout.variables['z'][:] = z

print('Define inner product')
afc = 1./(nx-1)/(ny-1)
hfc = h/np.sum(h)
# For zonal component (U)
nv = ny*(nx-1)
wv = np.ones(nv, dtype=np.float64)
wv[:nx-1] = 0.5
wv[-(nx-1):] = 0.5
wv = np.tile(wv*afc, nz)
for k in range(nz):
    wv[nv*k:nv*(k+1)] *= hfc[k]
nv *= nz    
# For meridional component (V)
nu = (ny-1)*nx
wu = np.ones(nx)
wu[0] = 0.5
wu[-1] = 0.5
wu = np.tile(afc*np.tile(wu, ny-1), nz)
for k in range(nz):
    wu[nu*k:nu*(k+1)] *= hfc[k]
nu *= nz    

print('Perform EOF procedure')
fout.variables['eigval'][...], ume, vme, umo, vmo, fout.variables['tmode'][...] = \
        spot_POD(u.reshape((nt,nu)), v.reshape((nt,nv)), wu, wv, nm, check_opt)
del u, v
fout.variables['umean'][...] = ume.reshape((nz,ny-1,nx))
fout.variables['vmean'][...] = vme.reshape((nz,ny,nx-1))
fout.variables['umode'][...] = umo.reshape((nm,nz,ny-1,nx))
fout.variables['vmode'][...] = vmo.reshape((nm,nz,ny,nx-1))

print('Build correlated drift')
mu = inner_prod(umo, ume, wu) + inner_prod(vmo, vme, wv) # project mean on modes 
del ume, vme, wu, wv
fout.variables['pcoef'][:] = mu
fout.variables['ucorr'][...] = (umo.T @ mu).reshape((nz,ny-1,nx))
fout.variables['vcorr'][...] = (vmo.T @ mu).reshape((nz,ny,nx-1))
del umo, vmo, mu
fout.close()

print('Program terminates')

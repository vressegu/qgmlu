import numpy as np
from pytorch_wavelets import DWTForward, DWTInverse 
from dataclasses import dataclass

@dataclass
class SimulationConfig:
    # --------- Numerical ---------
    n_order: int = 8              # N = 2^n_order = number of space steps along x
    N_steps: int = 1              # number of time steps
    n_ens: int = 3                # ensemble size    ATTENTION ATM lié au nombre de layers du QG model, à changer si on veut faire du QG 1 couche
    #method: str = 'Wavelet'      # Choice of method :'Fourier' or 'Wavelet'
    method: str = 'Fourier'       # Choice of method :'Fourier' or 'Wavelet'
    wave: str = 'db4'             # choice of the wavelet
    J: int = 9                    # Number of level of decomposition for wavelet method
    beta: float = 1.              # Energy injected in wavelet spectrum
    mode: str = 'periodization'
    new_spTimeGMC = False          # Wandrille's method for spatiotemporal GMC

    # --------- Physical ---------
    Lx: float = 2                  # domain size
    L_LS: float = 1.0 * Lx         # large spatial scale

    dim: float = 2.0               # spatial dimension
    c2: float = 0.025              # intermittency coefficient (must be < c1 /2, ]0,0.5[ in 2D)
    rho_s: float = 1.66            # spatial spectrum slope (]0,3[ in 2D; ]0,2[ in 3D)
    gamma_t: float = rho_s         # temporal spectrum slope
    c1 = (rho_s - 1) / 2.          # fractal exponent

    tau_0B: float = 1.0
    #tau_0e_factor: float = np.inf # steady intermittency
    tau_0e_factor: float = 1.0  # unsteady intermittency
    z_eps: float = 1.0

    # --------- Matérn parameters ---------
    nu_m: float = c1
    l_m: float = (np.sqrt(2 * nu_m) * L_LS) / (2 * np.pi)

class Parameters(SimulationConfig): 
    
    def __init__(self, **overrides):
        super().__init__(**overrides) 
    
        #Checking core dynamical parameters (2D)
        if (self.c2 >= self.c1 / 2) or (self.c1 >= 1.0):
            raise ValueError("Parameters must repect : c2 < c1/2 and c1 < 1.0 for 2D simulations. " \
                             + "Here : c1 = " + str(self.c1) + " and c2 = " + str(self.c2))
        else:
            print(f"Using consistent parameters: " \
                  + f"dim = {self.dim}, c1 = {self.c1}, c2 = {self.c2}, rho_s = {self.rho_s}, gamma_t = {self.gamma_t}")

        rho = self.rho_s
        N = 2 ** self.n_order
        self.PN = N//2                                 # aliased wavevector index
        self.dx = self.Lx / N

        self.wavelet = self.method == 'Wavelet'
        z = (self.rho_s - 1) / (self.gamma_t - 1)      # scaling slope for time scale
        theta_1 = 2 * self.nu_m + self.dim + 2         # spectrum slope of the bidirectionnal spatial spectrum  
        K_Lx = 2 * np.pi / self.Lx                     # largest-scale wavenumber
        K_LS = 2 * np.pi / self.L_LS                   # large-scale wavenumber
        self.K_LS = K_LS

        np.random.seed(0)

        # Wavenumbers
        kx = np.fft.fftfreq(N, d=self.dx) * 2 * np.pi
        if self.dim == 2 :
            KX, KY = np.meshgrid(kx, kx, indexing="ij")
            KX = KX[:,:,np.newaxis]
            KY = KY[:,:,np.newaxis]
            K = np.sqrt(KX**2 + KY**2)
            Kv = np.concat((KX, KY), axis= 2)
            if self.wavelet:
                KX = KX[:,:,0]
                KY = KY[:,:,0]
                K = K[:,:,0]
                Kv = Kv[:,:,0]
            self.KX = KX
            self.KY = KY
        elif self.dim == 3 :
            KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
            KX = KX[:,:,:,np.newaxis]
            KY = KY[:,:,:,np.newaxis]
            KZ = KZ[:,:,:,np.newaxis]
            K = np.sqrt(KX**2 + KY**2 + KZ**2)
            Kv = np.concat((KX, KY, KZ), axis= 3)
            Kv = Kv[:,:,:,:,np.newaxis]
            if self.wavelet:
                KX = KX[:,:,:,0]
                KY = KY[:,:,:,0]
                KZ = KZ[:,:,:,0]
                K = K[:,:,:,0]
                Kv = Kv[:,:,:,0]
            self.KX = KX
            self.KY = KY
            self.KZ = KZ
            self.Kv = Kv
        else:
            raise ValueError("Incorrect dim")

        # Spectrum
        S1 = 1 / (1 + (K / K_LS)**2 ) ** (theta_1 / 2)
        self.cleanAliasing(S1)
        A = np.sqrt(S1)
        self.N = N

        if (self.method == 'Wavelet'): # Wavelet method
            print(f"Using wavelet method with wavelet {self.wave} and J={self.J}")
            self.Jmax = np.log2(N)
            self.dwt = DWTForward(J=self.J, wave=self.wave, mode=self.mode)
            self.idwt = DWTInverse(wave=self.wave, mode=self.mode)
            if self.J > self.Jmax:
                raise ValueError("ACHTUNG : Requested J exceeds maximum wavelet scale !!!")
            
            # j = np.arange(1, self.J + 1)
            # Wavelet characteristic wavenumbers
            self.j = np.arange(self.J)
            self.k_j = K_Lx * (2.0 ** self.j)

            # ---------- OU time scales per scales ----------
            tau_B = self.tau_0B * (1 + (self.k_j / K_LS)**2) ** (-z / 2)
            tau_e = (self.tau_0e_factor * self.tau_0B * (1 + (self.k_j / K_LS)**2) ** (-self.z_eps / 2))

        elif (self.method == 'Fourier'): # Fourier method
            print("Using Fourier method")

            # OU timescales
            tau_B = self.tau_0B * (1 + (K / K_LS)**2 ) ** ( - z/2)
            tau_e = (self.tau_0e_factor * self.tau_0B) * (1 + (K / K_LS)**2 ) ** ( - self.z_eps/2)

            np.seterr(divide = 'ignore') 
            S2 = ( K / K_Lx ) ** (-self.dim) # cf eq (34) of Ruffenach et al. (2026)
            np.seterr(divide = 'warn') 
            self.cleanAliasing(S2)
            self.S2 = S2
        
        else:
            raise ValueError("Invalid method. Choose 'Fourier' or 'Wavelet'.")

        self.Delta_t = min(tau_B.min(), tau_e.min()) * 0.5
        tau_B_filtered = tau_B[~np.isinf(tau_B)]
        self.Big_t = np.max(tau_B_filtered) * 0.5
        print('my two small times are: ', tau_B.min() * 0.5, tau_e.min() * 0.5)
        print("The time step is : dt = " + str(self.Delta_t))
        print(2 * np.pi /(tau_B.min()), 2 * np.pi /(tau_B.max()))
        exp_B = np.exp(-self.Delta_t / tau_B)
        noise_B = np.sqrt(1 - exp_B**2)
        exp_e = np.exp(-self.Delta_t / tau_e)
        noise_e = np.sqrt(1 - exp_e**2)

        if self.dim == 3 and not self.wavelet:
            K = K[:,:,:,:,np.newaxis]
            A = A[:,:,:,:,np.newaxis]
            exp_B = exp_B[:,:,:,:,np.newaxis]
            noise_B = noise_B[:,:,:,:,np.newaxis]

        self.c2_eff = self.c2 * np.log( self.L_LS / (self.dx) )
        self.N = N
        self.K = K
        self.A = A
        self.exp_B = exp_B
        self.noise_B = noise_B
        self.exp_e = exp_e
        self.noise_e = noise_e
    
    def cleanAliasing(self,S1):
        if self.dim == 2 :
            S1[0, 0] = 0
            S1[self.PN, :] = 0
            S1[:, self.PN] = 0
        elif self.dim == 3 :
            S1[0, 0, 0] = 0
            S1[self.PN, :, :] = 0
            S1[:, self.PN, :] = 0
            S1[:, :, self.PN] = 0
        else:
            raise ValueError("Incorrect dim")

    def cleanAliasing_W(self,S1):
        S1[0, 0, 0] = 0
        S1[0, self.PN, :] = 0
        S1[0, :, self.PN] = 0

    def zeroField(self):
        if self.dim == 2 :
            zeroF = np.zeros((self.N_steps, self.N, self.N, self.n_ens))
        elif self.dim == 3 :
            zeroF = np.zeros((self.N_steps, self.N, self.N, self.N, self.n_ens))
        else:
            raise ValueError("Incorrect dim")
        return zeroF

    def zeroSpatialField(self):
        if self.dim == 2 :
            zeroF = np.zeros((self.N, self.N, self.n_ens))
        elif self.dim == 3 :
            zeroF = np.zeros((self.N, self.N, self.N, self.n_ens))
        else:
            raise ValueError("Incorrect dim")
        return zeroF
    
    def whiteNoise(self, scalar = False):
        if self.dim == 2 :
            xi = np.random.randn(self.N, self.N, self.n_ens)
        elif self.dim == 3 :
            xi = np.random.randn(self.N, self.N, self.N, 3*(not scalar) + scalar, self.n_ens)
            if scalar:
                xi = xi[:,:,:,0,:]
        else:
            raise ValueError("Incorrect dim")
        return xi

def run_simulation(cfg):

    # ---- init ----
    V_x = cfg.zeroField()
    V_y = cfg.zeroField()
    #V_z = cfg.zeroField()
    #stream_z = cfg.zeroField()
    #epsi = cfg.zeroField()

    lepsi_hat =  np.sqrt(cfg.S2) * fftnoise(cfg, scalar = True)
    if cfg.new_spTimeGMC:
        B_hat = CMG_noise(lepsi_hat,cfg)
    else:
        B_hat = fftnoise(cfg)
    V_x[0,:], V_y[0,:], V_z, stream_z, epsi\
        = CMG_velocity(B_hat,lepsi_hat,cfg, intermittency = (not cfg.new_spTimeGMC) )

    for t in range(cfg.N_steps-1):
        # ---- OU process for B ----
        if cfg.new_spTimeGMC:
            B_hat = cfg.exp_B * B_hat + cfg.noise_B * CMG_noise(lepsi_hat,cfg)
        else:
            B_hat = cfg.exp_B * B_hat + cfg.noise_B * fftnoise(cfg)
        # ---- OU process for epsilon ----
        lepsi_hat = cfg.exp_e  * lepsi_hat + cfg.noise_e * np.sqrt(cfg.S2) * fftnoise(cfg, scalar = True)

        V_x[t+1,:], V_y[t+1,:], V_z, stream_z, epsi \
            = CMG_velocity(B_hat,lepsi_hat,cfg, intermittency = (not cfg.new_spTimeGMC) )

    return V_x, V_y, V_z, stream_z, epsi



def fftnoise(cfg, scalar = False):
    xi = cfg.whiteNoise(scalar)
    B_hat = np.fft.fft2(xi, axes=(0,1)) 
    if cfg.dim == 3:
        B_hat = np.fft.fft(B_hat, axis=2) 

    cfg.cleanAliasing(B_hat)

    return B_hat

def CMG_noise(lepsi_hat,cfg):
    Bepsi_hat,epsi = CMG_from_noise(cfg.whiteNoise(),lepsi_hat,cfg)
    return Bepsi_hat

def CMG_velocity(B_hat,lepsi_hat,cfg, intermittency = True):

    # ---- Real space ----
    B_real = np.fft.ifft2(B_hat, axes=(0,1)).real
    if cfg.dim == 3:
        B_real = np.fft.ifft(B_real, axis=2).real
    # B_real = (B_real - B_real.mean()) / B_real.std() # needed ???

    Bepsi_hat,epsi = CMG_from_noise(B_real,lepsi_hat,cfg, intermittency)

    return velocity_from_Streamfunction(Bepsi_hat,epsi,cfg)

def CMG_from_noise(B_real,lepsi_hat,cfg, intermittency = True):

    if intermittency:
        lepsi_real = np.fft.ifft2(lepsi_hat, axes=(0,1)).real
        if cfg.dim == 3:
            lepsi_real = np.fft.ifft(lepsi_real, axis=2).real
            lepsi_real = lepsi_real[:,:,:,:,np.newaxis]
        # Set the correct variance (related to c2)
        lepsi_real /= lepsi_real.std() 
        lepsi_real *= np.sqrt( cfg.c2_eff )
        lepsi_real -= lepsi_real.var() 
    else:
        lepsi_real = cfg.zeroSpatialField()
    epsi = np.exp(lepsi_real)

    Bepsi_hat = np.fft.fft2(B_real * epsi, axes=(0,1))
    if cfg.dim == 3:
        Bepsi_hat = np.fft.fft(Bepsi_hat, axis=2) 
    cfg.cleanAliasing(Bepsi_hat)

    return Bepsi_hat,epsi

def velocity_from_Streamfunction(Bepsi_hat,epsi,cfg):

    # ---- Velocity ----
    stream_hat = cfg.A * Bepsi_hat
    if cfg.dim == 2:
        v_hat_x = 1j * cfg.KY * stream_hat
        v_hat_y = -1j * cfg.KX * stream_hat
    elif cfg.dim == 3:
        v_hat = np.cross( 1j * cfg.Kv, stream_hat, axisa=3, axisb=3, axisc=3)

    stream = np.fft.ifft2(stream_hat, axes=(0,1)).real
    if cfg.dim == 2:
        V_x = np.fft.ifft2(v_hat_x, axes=(0,1)).real
        V_y = np.fft.ifft2(v_hat_y, axes=(0,1)).real
        V_z = cfg.zeroSpatialField()
        stream_z = stream
    if cfg.dim == 3:
        V = np.fft.ifft2(v_hat, axes=(0,1)).real
        V = np.fft.ifft(V, axis=2).real

        #epsi = epsi[:,:,:,0,:]
        stream_z = stream[:,:,:,2,:]
        V_x = V[:,:,:,0,:]
        V_y = V[:,:,:,1,:]
        V_z = V[:,:,:,2,:]

    # Set amplitude
    ampli = cfg.K_LS * np.sqrt((stream_z[:]**2).mean())
    stream_z /= ampli
    V_x /= ampli
    V_y /= ampli
    V_z /= ampli

    return V_x, V_y, V_z, stream_z, epsi
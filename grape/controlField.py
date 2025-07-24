import numpy as np
from plotly import express as px
from scipy.integrate import simpson  # Simpson's rule for numerical integration
import pywt

class ControlField:
    "Define the control field properties"
    scale: float =1
    
    def __init__(self, global_parameters, initial_temp_value, optimize=False, parametrization={}):
        self.__N                        = global_parameters.N
        self.__t_s                      = global_parameters.t_s
        self.__dt                       = self.__t_s[1] - self.__t_s[0]
        self.__gamma                    = {'H': 267.513e6, 'Na': 70.8013e6, 'P': 108.291e6, 'C':67.262e6}.get(global_parameters.nucleus, 267.513e6)
        self.__optimize                 = optimize                  #Boolean. True if field to be optimized
        self.__initial_temp_value       = initial_temp_value
        self.__parametrization          = {'name':'temp',
                                            'fourier_fond':1/global_parameters.T_s,
                                            'fourier_harmNb':9,
                                            'wavelet_type':'db16',
                                            'wavelet_nbLevels':np.max([2,np.fix(np.log2(self.__N))-2])
                                            } # default parametrization values
        
        self.__parametrization.update(parametrization) # update parametrization
        # print(self.__parametrization)
        if self.__parametrization['name']=='wavelet':
            self.__Wmatrix, self.__Wmatrix_inv = self.compute_waveletMatrix()
            if np.log2(self.__N) != np.floor(np.log2(self.__N)): # Check if N is a power of 2
                raise ValueError("N must be a power of 2 for wavelet parametrization")
        elif self.__parametrization['name']=='fourier':
            self.__Fmatrix, self.__Fmatrix_inv = self.compute_FourierMatrix()
        elif self.__parametrization['name']=='fourier_even':
            self.__Fmatrix_even, self.__Fmatrix_even_inv = self.compute_FourierMatrix_even()
        elif self.__parametrization['name']=='fourier_odd':
            self.__Fmatrix_odd, self.__Fmatrix_odd_inv = self.compute_FourierMatrix_odd()

        self.__controlField           = self.compute_paramShape(initial_temp_value)
    
    
    
    
    # Define getters
    @property # to define getters
    def controlField(self):
        return self.__controlField
    
    @controlField.setter
    def controlField(self, newCF):
        """Setter for the private attribute"""
        # if value < 0:
        #     raise ValueError("Value cannot be negative")
        self.__controlField = newCF
    
    @property # to define getters
    def optimize(self):
        return self.__optimize




    # Define methods
    def compute_tempShape(self):
        cF = self.__controlField
        if (self.__parametrization['name'] == 'fourier'):
            # fourier_fond     = self.__parametrization['fourier_fond']
            # fourier_harmNb   = self.__parametrization['fourier_harmNb']
            # t   = self.__t_s
            # a0  = cF[0]
            # an  = cF[1:fourier_harmNb+1]
            # bn  = cF[fourier_harmNb+1:]
            # f2   = a0*np.ones(self.__N)  # Initialize with DC component (a0)            
            # # Add cosine and sine terms
            # for n in range(1, len(an) + 1):
            #     f2 += an[n-1] * np.cos(2 * np.pi * n * fourier_fond * t)  # Cosine term
            #     f2 += bn[n-1] * np.sin(2 * np.pi * n * fourier_fond * t)  # Sine term
            f = self.__Fmatrix_inv@cF.reshape((cF.shape[0],1))
        elif (self.__parametrization['name'] == 'fourier_even'):
            # fourier_fond     = self.__parametrization['fourier_fond']
            # fourier_harmNb   = self.__parametrization['fourier_harmNb']
            # t   = self.__t_s
            # a0  = cF[0]
            # an  = cF[1:]
            # f2   = a0*np.ones(self.__N)  # Initialize with DC component (a0)            
            # # Add cosine and sine terms
            # for n in range(1, len(an) + 1):
            #     f2 += an[n-1] * np.cos(2 * np.pi * n * fourier_fond * t)  # Cosine term   
            f = self.__Fmatrix_even_inv@cF.reshape((cF.shape[0],1))
        elif (self.__parametrization['name'] == 'fourier_odd'):
            # fourier_fond     = self.__parametrization['fourier_fond']
            # fourier_harmNb   = self.__parametrization['fourier_harmNb']
            # t   = self.__t_s
            # bn  = cF
            # f2   = np.zeros(self.__N)  # Initialize with DC component (a0)            
            # # Add cosine and sine terms
            # for n in range(1, len(bn) + 1):
            #     f2 += bn[n-1] * np.sin(2 * np.pi * n * fourier_fond * t)  # Sine term      
            f = self.__Fmatrix_odd_inv@cF.reshape((cF.shape[0],1))
        elif (self.__parametrization['name'] == 'wavelet'):   
            f = self.__Wmatrix_inv@cF.reshape((cF.shape[0],1))
        elif (self.__parametrization['name'] == 'bezier'):
            raise NameError('Bezier parametrization not done')
        elif (self.__parametrization['name'] == 'temp'):
            f = cF
        return np.real(f.flatten())

    def compute_paramShape(self, f):
        # Convert f from temporal domain to parameter space
        if (self.__parametrization['name'] == 'fourier'):
            # fourier_fond     = self.__parametrization['fourier_fond']
            # fourier_harmNb   = self.__parametrization['fourier_harmNb']
            # t   = self.__t_s
            # a0  = fourier_fond * simpson(f,x=t)
            # # Initialize coefficients
            # an = np.zeros(fourier_harmNb)  # Cosine coefficients
            # bn = np.zeros(fourier_harmNb)  # Sine coefficients
            # # Compute the Fourier coefficients
            # for n in range(1, fourier_harmNb + 1):
            #     an[n-1] = 2*fourier_fond * simpson(f * np.cos(2 * np.pi * n * t *fourier_fond), x=t)  # Cosine coefficients
            #     bn[n-1] = 2*fourier_fond * simpson(f * np.sin(2 * np.pi * n * t *fourier_fond), x=t)  # Sine coefficients
            # x2 = np.hstack((a0,an,bn))
            x = self.__Fmatrix@f.reshape(f.shape[0], 1)
        if (self.__parametrization['name'] == 'fourier_even'):
            # fourier_fond     = self.__parametrization['fourier_fond']
            # fourier_harmNb   = self.__parametrization['fourier_harmNb']
            # t   = self.__t_s
            # a0  = fourier_fond * simpson(f, x=t)
            # # Initialize coefficients
            # an = np.zeros(fourier_harmNb)  # Cosine coefficients
            # # Compute the Fourier coefficients
            # for n in range(1, fourier_harmNb + 1):
            #     an[n-1] = 2*fourier_fond * simpson(f * np.cos(2 * np.pi * n * t *fourier_fond), x=t)  # Cosine coefficients
            # x2 = np.hstack((a0,an))  
            x = self.__Fmatrix_even@f.reshape(f.shape[0], 1)
        if (self.__parametrization['name'] == 'fourier_odd'):
            # fourier_fond     = self.__parametrization['fourier_fond']
            # fourier_harmNb   = self.__parametrization['fourier_harmNb']
            # t   = self.__t_s
            # # Initialize coefficients
            # bn = np.zeros(fourier_harmNb)  # Sine coefficients
            # # Compute the Fourier coefficients
            # for n in range(1, fourier_harmNb + 1):
            #     bn[n-1] = 2*fourier_fond * simpson(f * np.sin(2 * np.pi * n * t *fourier_fond), x=t)  # Sine coefficients
            # x2 = bn
            x = self.__Fmatrix_odd@f.reshape(f.shape[0], 1)
        if (self.__parametrization['name'] == 'wavelet'):  
            x = self.__Wmatrix@f.reshape(f.shape[0], 1)
        elif (self.__parametrization['name'] == 'bezier'):
            raise NameError('Bezier parametrization not done')
            x = 0 # to do
        elif (self.__parametrization['name'] == 'temp'):
            x = f
        return x.flatten()
    
    def compute_paramGrad(self, f):
        # Convert gradient from temporal domain to parameter space
        if (self.__parametrization['name'] == 'fourier'): #TODO
           x = self.__Fmatrix@f.reshape(f.shape[0], 1)
        elif (self.__parametrization['name'] == 'fourier_even'): #TODO
            x = self.__Fmatrix_even@f.reshape(f.shape[0], 1)
        elif (self.__parametrization['name'] == 'fourier_odd'): #TODO
            x = self.__Fmatrix_odd@f.reshape(f.shape[0], 1)
        elif (self.__parametrization['name'] == 'wavelet'):  
            x = self.__Wmatrix@f.reshape(f.shape[0], 1)
        elif (self.__parametrization['name'] == 'bezier'):
            raise NameError('Bezier parametrization not done')
            x = 0 # to do
        elif (self.__parametrization['name'] == 'temp'):
            x = f
        return x
    
    def plot_tempControl(self, plot=True):
        tempC = self.compute_tempShape()
        if plot:
            fig = px.line(x=self.__t_s*1000, y = tempC * 1e6 / self.__gamma, labels={'x': 'Time (ms)', 'y':'(µT)'}, title='Temporal shape')
            fig.show()
        return self.__t_s, tempC * 1e6 / self.__gamma
    
    def compute_FourierMatrix(self):
        N = self.__N
        fourier_fond = self.__parametrization['fourier_fond']
        fourier_harmNb = self.__parametrization['fourier_harmNb']
        t = self.__t_s
        F           = np.zeros((2 * fourier_harmNb + 1, N))
        F_inv       = np.zeros((N , 2 * fourier_harmNb + 1))
        F[0,:]      = np.ones(N)*fourier_fond
        F_inv[:,0]  = np.ones(N)
        for n in range(1, fourier_harmNb + 1):
            F[n,:] = 2 * fourier_fond * np.cos(2 * np.pi * n * fourier_fond * t) # Cosine term in the fourier_harmNb first lines
            F[fourier_harmNb + n,:] = 2 * fourier_fond * np.sin(2 * np.pi * n * fourier_fond * t) # Sine term in the fourier_harmNb last lines
            F_inv[:,n] = np.cos(2 * np.pi * n * fourier_fond * t) # Cosine term in the fourier_harmNb first columns  
            F_inv[:,fourier_harmNb + n] = np.sin(2 * np.pi * n * fourier_fond * t) # Sine term in the fourier_harmNb last columns
        return F*self.__dt,F_inv
    
    def compute_FourierMatrix_even(self):
        N = self.__N
        fourier_fond = self.__parametrization['fourier_fond']
        fourier_harmNb = self.__parametrization['fourier_harmNb']
        t = self.__t_s
        F = np.zeros((fourier_harmNb + 1, N))
        F_inv = np.zeros((N, fourier_harmNb + 1))
        F[0,:] = np.ones(N)*fourier_fond
        F_inv[:,0] = np.ones(N)
        for n in range(1, fourier_harmNb + 1):
            F[n,:] = 2 * fourier_fond * np.cos(2 * np.pi * n * fourier_fond * t) # Cosine term in the fourier_harmNb first lines
            F_inv[:,n] = np.cos(2 * np.pi * n * fourier_fond * t) # Cosine term in the fourier_harmNb first lines
        return F*self.__dt, F_inv
    
    def compute_FourierMatrix_odd(self):
        N = self.__N
        fourier_fond = self.__parametrization['fourier_fond']
        fourier_harmNb = self.__parametrization['fourier_harmNb']
        t = self.__t_s
        F = np.zeros((fourier_harmNb, N))
        F_inv = np.zeros((N, fourier_harmNb))
        for n in range(1, fourier_harmNb + 1):
            F[n-1,:] = 2 * fourier_fond * np.sin(2 * np.pi * n * fourier_fond * t) # Cosine term in the fourier_harmNb first lines
            F_inv[:,n-1] = np.sin(2 * np.pi * n * fourier_fond * t) # Cosine term in the fourier_harmNb first lines
        return F*self.__dt, F_inv

    def compute_waveletMatrix(self):
        N = self.__N
        wavelet_type = self.__parametrization['wavelet_type']
        J = np.fix(np.log2(N)).astype(np.int32)  # Maximum level of wavelet decomposition
        wavelet_nbLevels = self.__parametrization['wavelet_nbLevels']   # Number of levels for wavelet decomposition

        # Create the wavelet object for Daubechies16
        wavelet_obj = pywt.Wavelet(wavelet_type)

        # Get the decomposition low-pass and high-pass filter coefficients
        LoD = wavelet_obj.dec_lo  # Low-pass filter coefficients
        HiD = wavelet_obj.dec_hi  # High-pass filter coefficients

        # Initialize h and g with the filter coefficients
        h = LoD
        g = HiD

        G = np.zeros((N, N))
        H = np.zeros((N, N))
        gshift = np.zeros(N)
        gshift[:len(g)] = np.flip(g)
        hshift = np.zeros(N)
        hshift[:len(h)] = np.flip(h)

        # Create matrices H and G containing the filter coefficients for all spatial shifts
        for k in range(N):
            G[k, :] = gshift
            H[k, :] = hshift
            gshift = np.roll(gshift, 2)
            hshift = np.roll(hshift, 2)

        W = np.zeros((N, N))
        H_up = np.eye(N)

        # Loop on scales from the highest to the lowest resolution
        for c in range(1, J + 1):
            tmp = G @ H_up  # Matrix multiplication
            W[N // (2**c):N // (2**(c-1)), :] = tmp[N // (2**c):N // (2**(c-1)), :]
            H_up = H_up @ H  # Matrix multiplication

        W[:N // (2**c), :] = H_up[:N // (2**c), :]
        W = W[:2**(int(np.log2(N)) - J + wavelet_nbLevels), :]

        Winv = W.T  # Transpose of W
        return W, Winv

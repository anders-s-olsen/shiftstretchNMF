import torch
import torch.nn as nn
import math
import numpy as np
import time
from scipy.interpolate import interp1d
from CGD.tac_clustering import kmeans_plusplus, kshape, kmeans
from sklearn.decomposition._nmf import _initialize_nmf
torch.set_default_dtype(torch.float64)
import matplotlib.pyplot as plt
eps = 1e-16

# Inverse softplus function for parameter initialization
def inv_softplus(x):
    if torch.any(x<=0):
        raise ValueError('Input to inv_softplus must be greater than 0')
    return x + torch.log(-torch.expm1(-x))

class ShiftNMF(nn.Module):
    def __init__(self, K:int, P:int, N:int, init_params:list=None, 
                 integer_shift:bool=False, non_integer_shift:bool=False, 
                 stretch:bool=False,A_graph:bool=False,
                 zeropad_fraction:float=None, keep_S_fixed:bool=False):
        """
        ShiftNMF model for non-negative matrix factorization with optional shifting.

        Args:
            K: Number of components
            P: Number of channels
            N: Number of time points
            init_params: List of initial values for A and S and potentially tau. 
            integer_shift: If True, estimate integer shifts
            non_integer_shift: If True, estimate non-integer shifts
            zeropad_fraction: Fraction of zeropadding to add to the data
            keep_S_fixed: If True, S is not updated during training
        """
        super(ShiftNMF, self).__init__()
        self.K = K  # Number of components 
        self.P = P  # Number of channels
        self.N_init = N  # Number of time points before zeropadding

        # Handle zeropadding if specified
        if zeropad_fraction is None or zeropad_fraction == 0:
            self.N = N  # Number of time points
        else:
            self.N = int(N*(1+zeropad_fraction))
        self.N_fft = self.N//2+1 

        # Shift and padding options
        self.integer_shift = integer_shift  
        self.non_integer_shift = non_integer_shift
        self.stretch = stretch
        self.zeropad_fraction = zeropad_fraction
        self.analytical_A = False  # Whether to compute A analytically from S and tau
        self.A_graph = A_graph  # Whether to enforce computational graph on A

        if self.stretch:
            self.pad_grid = torch.arange(-self.N_fft//2+1,self.N_fft//2-1)
            # self.pad_grid = torch.arange(-self.N//4+1,self.N//4-1)
            # self.pad_grid = torch.arange(0,self.N-1)

        # Whether to keep S fixed during training - e.g., when testing
        self.keep_S_fixed = keep_S_fixed

        self.X_f = None  # Cached FFT of input
        self.inv_sqrt2 = torch.rsqrt(torch.tensor(2.0))

        # Non-negativity transform (Softplus for smooth non-negativity)
        self.non_negative_transform = nn.Softplus()

        # Only one shift mode can be enabled
        if self.integer_shift and self.non_integer_shift:
            raise ValueError("Only one of integer_shift and non_integer_shift can be True. "
            "Start by estimating a model with integer shift and then finetune with non-integer shift")
        
        if not self.integer_shift and not self.A_graph:
            raise ValueError("A_graph must be True if not using an integer-shift model, otherwise A will be unconstrained")
        
        if self.stretch and not self.integer_shift:
            raise ValueError("Stretch can only be used in combination with integer_shift=True")
        
        # Parameter initialization
        if init_params is not None:
            if self.keep_S_fixed:
                if len(init_params) == 1:
                    self.S_raw = init_params[0] # Only S provided
                else:
                    self.A_raw = init_params[0]
                    self.S_raw = init_params[1]
            else:
                self.A_raw = init_params[0]
                self.S_raw = init_params[1]
            if self.integer_shift or self.non_integer_shift:
                try:
                    self.tau = init_params[2]
                except:
                    self.tau = torch.zeros(self.P, self.K)
            if self.stretch:
                try:
                    # self.a = init_params[3]
                    self.a_idx = init_params[3]
                except:
                    # self.a = torch.ones(self.P, self.K)
                    self.a_idx = torch.ones(self.P,self.K,dtype=int)*self.pad_grid.shape[0]//2+1
        else:
            if self.keep_S_fixed:
                raise ValueError("If keep_S_fixed is True, init_params:list must be provided with S either in the first or second position")
            self.A_raw = torch.randn(self.P, self.K)
            self.S_raw = torch.randn(self.K, self.N)
            if self.integer_shift or self.non_integer_shift:
                self.tau = torch.zeros(self.P, self.K)
            if self.stretch:
                # self.a = torch.ones(self.P, self.K)  # Stretch factor
                self.a_idx = torch.ones(self.P,self.K,dtype=int)*self.pad_grid.shape[0]//2+1
        
        # Frequency grid for phase shifting (used in shift models)
        if self.integer_shift or self.non_integer_shift or self.stretch:
            self.f = (torch.arange(self.N)/self.N).view(1, -1, 1)  # Shape: (1, N, 1)
            self.f = self.f[:,:self.N_fft,:] # Shape: (1, N_fft, 1)
        
        # Register parameters as nn.Parameter for optimization
        if self.A_graph:
            self.A_raw = nn.Parameter(self.A_raw)
        if not self.keep_S_fixed:
            self.S_raw = nn.Parameter(self.S_raw)
        if self.non_integer_shift: # tau is learnable for non-integer shifts
            self.tau = nn.Parameter(self.tau.to(torch.float64))

    # Estimate tau (shifts) from S and X using cross-covariance in frequency domain
    def guess_tau_from_S(self,X, S=None):
        if S is None:
            S = self.non_negative_transform(self.S_raw)
        S_f = torch.fft.rfft(S,dim=-1)
        X_f = torch.fft.rfft(X,dim=-1)

        cross_spec = X_f[:,None,:] * torch.conj(S_f[None,:,:])
        cross_cov = torch.fft.irfft(cross_spec, n=self.N, dim=-1)
        delay = torch.argmax(cross_cov, dim=-1) - self.N
        if self.non_integer_shift:
            self.tau = nn.Parameter(delay.to(torch.float64))
        else:
            self.tau = delay

    # Estimate A given tau and S (solves least squares in frequency domain)
    def guess_A_from_tau_S(self, X=None, S=None, X_f=None):
        if S is None:
            S = self.non_negative_transform(self.S_raw)
        S_f = torch.fft.rfft(S,dim=-1)
        if X_f is None:
            X_f = torch.fft.rfft(X,dim=-1)

        if self.integer_shift or self.non_integer_shift:
            phase_shift = torch.exp(-1j*2*math.pi*self.f * self.tau.unsqueeze(1))
            A = torch.linalg.lstsq(S_f.T[None] * phase_shift,X_f[:,:,None])[0][:,:,0]
        else:
            A = torch.linalg.lstsq(S_f.T[None],X_f[:,:,None])[0][:,:,0]
        return torch.abs(A)

    # Initialization using various methods (random, NNDSVD, k-means, etc.)
    def initialize(self, X, init_method:str='nndsvda', repeats:int=1):
        if self.zeropad_fraction is not None and self.zeropad_fraction != 0:
            X = torch.cat((X, torch.zeros((self.P, self.N-self.N_init))), dim=1)
        if self.keep_S_fixed:
            if self.integer_shift or self.non_integer_shift:
                self.guess_tau_from_S(X=X)
            self.A_raw = nn.Parameter(inv_softplus(self.guess_A_from_tau_S(X=X)+eps))
            return
        if init_method.lower() == 'random':
            return
        elif init_method.lower() in ['nndsvd', 'nndsvda', 'nndsvdar']:
            A,S = _initialize_nmf(X.numpy(), self.K, init=init_method)
        elif init_method.lower() in ['k-shape','kshape']:
            _,S,_ = kshape(X.numpy(), self.K, num_repl=repeats)
            S[S<0] = eps
        elif init_method.lower() in ['k-means','kmeans']:
            _,S,_ = kmeans(X.numpy(), self.K, num_repl=repeats)
            S[S<0] = eps
        elif init_method.lower() in ['++','plusplus']:
            _,S = kmeans_plusplus(X.numpy(), self.K,dist='crosscorr')
        else:
            raise ValueError("Invalid init_method. Choose from 'random', 'nndsvd', "
            "'nndsvda', 'nndsvdar', 'kshape', 'k-means, '++'. Alternatively, provide S and set keep_S_fixed=True")
        
        if self.integer_shift or self.non_integer_shift:
            self.guess_tau_from_S(X=X, S=torch.tensor(S))
        A = self.guess_A_from_tau_S(X=X, S=torch.tensor(S))
        
        # Use inv_softplus for initialization if using Softplus non-negativity
        if isinstance(self.non_negative_transform, nn.Softplus):
            if self.A_graph:
                self.A_raw = nn.Parameter(inv_softplus(A+eps))
            else:
                self.A_raw = A
            self.S_raw = nn.Parameter(inv_softplus(torch.tensor(S)+eps))
        else:
            if self.A_graph:
                self.A_raw = nn.Parameter(A)
            else:
                self.A_raw = A
            self.S_raw = nn.Parameter(torch.tensor(S))

    # Cross-correlation computation for updating tau (shifts)
    def cross(self, A, S_f, return_A=False):
        A_f = A.unsqueeze(1) * self.phase_shift
        R_f = self.X_f - torch.sum(A_f*S_f.T[None],-1)
        for k in range(self.K):
            R_f_without_k = R_f + A_f[:,:,k]*S_f[None,k,:]
            C_f = R_f_without_k*torch.conj(S_f[None,k,:])
            C = torch.fft.irfft(C_f, dim=-1, n=self.N)
            maxval, ind = torch.max(C, dim=-1)
            self.tau[:,k] = ind - self.N
            
            if return_A:
                S_f_corrected = S_f[k].clone()
                S_f_corrected[[0, -1]] *= self.inv_sqrt2  
                S_norm = 2/self.N*torch.norm(S_f_corrected, dim=-1)**2
                A[:,k] = maxval / S_norm[None]  # Normalize A by S
                A[:,k][A[:,k]<0] = eps
                phase_shift_new = torch.exp(-1j*2*math.pi*self.f[:,:,0] * self.tau[:,k].unsqueeze(1))
                A_f[:,:,k] = A[:,k].unsqueeze(-1) * phase_shift_new
                R_f = R_f_without_k - A_f[:,:,k] * S_f[None,k,:]
        if not self.A_graph and return_A:
            return A

    def crossstretch(self, A, S_f_stretched, return_A=False):
        
        # apply phase shift to A
        A_f = A.unsqueeze(1) * self.phase_shift

        # reconstruction with the current stretch and shift for each voxel applied 
        S_f_actual_stretched = torch.zeros(self.P, self.K, self.N_fft, dtype=torch.complex128)
        for k in range(self.K):
            S_f_actual_stretched[:,k] = S_f_stretched[k,self.a_idx[:,k]]
        
        recon = torch.sum(A_f*torch.swapaxes(S_f_actual_stretched,-2,-1),-1)
        R_f = self.X_f - recon
        
        # reconstruction without the kth component
        for k in range(self.K):
            R_f_without_k = R_f + A_f[:,:,k]*S_f_actual_stretched[:,k,:] 
        
            # cross-spectrum and cross-correlation
            C_f = R_f_without_k[:,None,:]*torch.conj(S_f_stretched[k,None,:,:])
            C = torch.fft.irfft(C_f, dim=-1, n=self.N)
        
            # flatten stretch and lag dimensions to find argmax over both simultaneously
            C_flat = C.view(self.P, self.pad_grid.shape[0] * self.N)

            # Index of max in flattened space
            flat_value,flat_idx = torch.max(C_flat, dim=-1)

            # Step 3: Convert back to (stretch_idx, lag_idx)
            best_stretch_idx = flat_idx // self.N  # integer division
            # self.a_idx[:,k] = self.pad_grid[best_stretch_idx]
            self.a_idx[:,k] = best_stretch_idx

            best_lag_idx = flat_idx % self.N       # remainder
            self.tau[:,k] = best_lag_idx - self.N

            if return_A:
                S_f_corrected = S_f_stretched[k,self.a_idx[:,k]].clone()
                S_f_corrected[:, [0, -1]] *= self.inv_sqrt2  
                S_norm = 2/self.N*torch.norm(S_f_corrected, dim=-1)**2
                A[:,k] = flat_value / S_norm  # Normalize A by S
                # A[:,k] = torch.real(torch.max(C_flat[:,best_stretch_idx*self.N + best_lag_idx],dim=-1).values) / S_norm  # Normalize A by S
                A[:,k][A[:,k]<0] = 0
                phase_shift_new = torch.exp(-1j*2*math.pi*self.f[:,:,0] * self.tau[:,k].unsqueeze(1))
                A_f[:,:,k] = A[:,k].unsqueeze(-1) * phase_shift_new
                R_f = R_f_without_k - A_f[:,:,k] * S_f_stretched[k,self.a_idx[:,k],:]
        if not self.A_graph and return_A:
            return A

    def candidate_S_stretch(self,S):
        K,N = S.shape
        original_energy = torch.linalg.norm(S, dim=-1)**2
        S_f = torch.fft.rfft(S, dim=-1)
        S_stretch_all = torch.zeros(K,len(self.pad_grid),N)
        for i,a in enumerate(self.pad_grid):
            if a>0:
                S_f_padded = torch.hstack([S_f,torch.zeros(K,a)])
                S_stretched = torch.fft.irfft(S_f_padded,dim=-1)
                S_stretched_truncated = S_stretched[:,:N]
                S_stretch_all[:,i,:] = S_stretched_truncated * torch.sqrt(original_energy/torch.linalg.norm(S_stretched_truncated, dim=-1)**2)[:,None]
            elif a<0:
                S_f_padded = S_f[:,:a]
                S_f_padded = S_f_padded * torch.sqrt(original_energy/torch.linalg.norm(torch.fft.irfft(S_f_padded,dim=-1), dim=-1)**2)[:,None]
                tmp = torch.fft.irfft(S_f_padded,dim=-1)
                S_stretch_all[:,i,:] = torch.hstack([tmp,torch.zeros(K,N-tmp.shape[1])])
            elif a==0:
                S_stretch_all[:,i,:] = torch.fft.irfft(S_f,n=N,dim=-1)
        return S_stretch_all
            
    def forward(self, X):
        
        # Compute FFT of input data if not already cached
        if self.X_f is None:
            if self.zeropad_fraction is not None and self.zeropad_fraction != 0:
                X = torch.cat((X, torch.zeros((self.P, self.N-self.N_init))), dim=1)
            self.X = X
            self.X_f = torch.fft.rfft(X, dim=1)
            if self.integer_shift or self.non_integer_shift:
                self.phase_shift = torch.exp(-1j*2*math.pi*self.f * self.tau.unsqueeze(1))
        
        # Apply non-negativity constraint to parameters
        if self.A_graph:
            A = self.non_negative_transform(self.A_raw)
        else:
            A = self.A_raw
        S = self.non_negative_transform(self.S_raw)
        if self.stretch:
            S_stretched = self.candidate_S_stretch(S)
            S_stretched[S_stretched<0] = 0
            S_f_stretched = torch.fft.rfft(S_stretched, dim=-1)
        else:
            S_f = torch.fft.rfft(S, dim=1)

        # Compute phase-shift if using shift model
        if self.integer_shift or self.non_integer_shift:
            if self.integer_shift:
                with torch.no_grad():
                    if self.A_graph:
                        if self.stretch:
                            self.crossstretch(A=A, S_f_stretched=S_f_stretched)
                        else:
                            self.cross(A=A, S_f=S_f, return_A=False)
                    else:
                        if self.stretch:
                            A = self.crossstretch(A=A, S_f_stretched=S_f_stretched, return_A=True)
                        else:
                            A = self.cross(A=A, S_f=S_f, return_A=True)
                        self.A_raw = A
            self.phase_shift = torch.exp(-1j*2*math.pi*self.f * self.tau.unsqueeze(1))
            A_f = A.unsqueeze(1) * self.phase_shift
            if self.stretch:
                S_f_actual_stretched = torch.zeros(self.P, self.K, self.N_fft, dtype=torch.complex128)
                for k in range(self.K):
                    S_f_actual_stretched[:,k,:] = S_f_stretched[k,self.a_idx[:,k]]
        else:
            A_f = A.unsqueeze(1)
        
        # Compute reconstruction and loss
        if self.stretch:
            recon = torch.sum(A_f*torch.swapaxes(S_f_actual_stretched,-2,-1),-1)
        else:
            recon = torch.sum(A_f*S_f.T[None],-1)
        X_f_minus_reconstruction = self.X_f - recon

        # correct DC and Nyquist frequency components to compare loss directly to the time domain equivalent
        X_f_minus_reconstruction[:, [0, -1]] *= self.inv_sqrt2  

        # least squares loss in frequency domain
        ls_loss = 1/(self.N)*torch.linalg.matrix_norm(X_f_minus_reconstruction, ord='fro')**2

        return ls_loss# + 1e6*((S_norm-1)**2).sum()

    # Get raw model parameters (optionally including tau)
    def get_model_params(self):
        if self.stretch:
            return [self.A_raw.detach(), self.S_raw.detach(), self.tau.detach(), self.a_idx.detach()]
        elif self.integer_shift or self.non_integer_shift:
            return [self.A_raw.detach(), self.S_raw.detach(), self.tau.detach()]
        else:
            return [self.A_raw.detach(), self.S_raw.detach()]
    
    # Get transformed (non-negative) model parameters, and shifted S if using shift model
    def get_transformed_model_params(self):
        if self.A_graph:
            A = self.non_negative_transform(self.A_raw).detach()    
        else:
            A = self.A_raw
        S = self.non_negative_transform(self.S_raw).detach()

        if (self.integer_shift or self.non_integer_shift):
            A_max = torch.argmax(A, dim=0)  # (K,)
            tau_max = self.tau.detach()[A_max, torch.arange(self.K)]  # (K,)
            S_f_shifted =  torch.fft.rfft(S, dim=1) * torch.exp(-1j*2*math.pi*self.f[:,:,0] * tau_max.unsqueeze(1))  # (K, N_fft)
            S_shifted = torch.fft.irfft(S_f_shifted, dim=1, n=self.N)  # (K, N)
            tau = self.tau.detach() - tau_max.unsqueeze(0) - self.N  # (P, K)
            # return A.numpy(), S_shifted.numpy()[:,:self.N_init], tau.numpy()
            if self.stretch:
                return A.numpy(),S_shifted.numpy(),tau, self.a_idx.numpy()    
            else:
                return A.numpy(),S_shifted.numpy(),tau
        else:
            return A.numpy(), S.numpy()
        
        
    # Set model parameters (raw, not transformed)
    def set_model_params(self, params):
        if self.integer_shift or self.non_integer_shift:
            self.A_raw.data = params[0].clone()
            self.S_raw.data = params[1].clone()
            self.tau.data = params[2].clone()
        else:
            self.A_raw.data = params[0].clone()
            self.S_raw.data = params[1].clone()


    # # Cross-correlation computation for updating tau (shifts)
    # def crossstretch(self, A, S_f):
    #     # t1 = time.time()
    #     A_f = A.unsqueeze(1) * self.phase_shift
    #     actual_stretch_frequencies = self.f[None,:,:,0] * self.a.unsqueeze(-1)
    #     potential_stretch_frequencies = self.f[:,:,0] * self.pad_grid.unsqueeze(-1)
    #     S_f_actual_stretched = torch.zeros(self.P, self.K, self.N_fft, dtype=torch.complex128)
    #     S_f_potential_stretched = torch.zeros(self.K, self.pad_grid.shape[0], self.N_fft, dtype=torch.complex128)
    #     for k in range(self.K):
    #         interp = interp1d(self.f[0,:,0].numpy(), S_f[k].numpy(), bounds_error=False, fill_value=0.0)
    #         S_f_actual_stretched[:,k,:] = torch.from_numpy(interp(actual_stretch_frequencies[:,k,:].numpy())) * self.a[:,k].unsqueeze(-1)
    #         S_f_potential_stretched[k] = torch.from_numpy(interp(potential_stretch_frequencies.numpy())) * self.pad_grid.unsqueeze(-1)
    #     # t2 = time.time()
        
    #     X_f_minus_reconstruction = self.X_f - torch.sum(A_f*torch.swapaxes(S_f_actual_stretched,-2,-1),-1)
    #     R_f = torch.zeros(self.P, self.K, self.N_fft, dtype=torch.complex128)
    #     for k in range(self.K):
    #         R_f[:,k,:] = X_f_minus_reconstruction + A_f[:,:,k]*S_f_actual_stretched[:,k,:]
    #     # t3 = time.time()

    #     C_f = R_f[:,:,None,:]*torch.conj(S_f_potential_stretched[None,:,:,:])
    #     C = torch.fft.irfft(C_f, dim=-1, n=self.N)
    #     # t4 = time.time()

    #     C_flat = C.view(self.P, self.K, self.pad_grid.shape[0] * self.N)

    #     # Index of max in flattened space
    #     flat_idx = torch.argmax(C_flat, dim=-1)  # shape (P, K)

    #     # Step 3: Convert back to (stretch_idx, lag_idx)
    #     best_stretch_idx = flat_idx // self.N  # integer division
    #     self.a = self.pad_grid[best_stretch_idx]

    #     best_lag_idx = flat_idx % self.N       # remainder
    #     self.tau = best_lag_idx - self.N
    #     # t5 = time.time()
    #     # print(t5-t4,t4-t3,t3-t2,t2-t1)

    
                # actual_stretch_frequencies = self.f[None,:,:,0] * self.a.unsqueeze(-1)
                # S_f_actual_stretched = torch.zeros(self.P, self.K, self.N_fft, dtype=torch.complex128)
                # for k in range(self.K):
                #     interp = interp1d(self.f[0,:,0].numpy(), S_f[k].detach().numpy(), bounds_error=False, fill_value=0.0)
                #     S_f_actual_stretched[:,k,:] = torch.from_numpy(interp(actual_stretch_frequencies[:,k,:].numpy()))



                
# compress S
# S_compressed = torch.zeros_like(S)
# for k in range(self.K):
#     original_energy = torch.linalg.norm(S[k])**2
#     S_f_compressed = torch.fft.rfft(S[k], dim=-1)
#     S_f_compressed = S_f_compressed[:S_f_compressed.shape[0]//2+1]
#     S_compressed_k = torch.fft.irfft(S_f_compressed, dim=-1)
#     S_compressed_k = S_compressed_k * torch.sqrt(original_energy/torch.linalg.norm(S_compressed_k)**2)
#     S_compressed[k] = torch.hstack([S_compressed_k,torch.zeros(self.N-S_compressed_k.shape[0])])
# S_stretched = self.candidate_S_stretch(S_compressed)

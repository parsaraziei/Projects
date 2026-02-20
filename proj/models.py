
import numpy as np
import utils 
import pandas as pd
from scipy.special import gamma
from scipy.signal import fftconvolve
from fbm import FBM
from concurrent.futures import ThreadPoolExecutor
from numba import njit, prange
from stochastic.processes.continuous import FractionalBrownianMotion


class models:
    dt = 1/365
    
    def __init__(self,S0,K,IR,ta,sig,sym,beg_date,fin_date,_seed,num_sims,_AV):
        self.S0 = S0
        self.IR = IR
        self.K = K
        self.ta = ta
        self.x_ta = ta
        self.sig = sig
        self.sym = sym
        self.beg_date = beg_date
        self.fin_date = fin_date
        self.AV = _AV
        self.NumSimulations = num_sims
        self.ta_yrs = ta/365
        # in case jump diffusion model is also in use
        self.poisson_rate = 0
        self.std_jump = 0
        self.mean_jump = 0
        
        # in case heston model is being used
        self.vov = 0
        self.kappa = 0
        self.theta = 0
        self.rho = 0
        
        # in case Rough Heston is used
        self.RH_vov = 0
        self.RH_kappa = 0
        self.RH_rho = 0
        self.RH_theta = 0
        self.RH_hurst = 0
        self.fseed = _seed
        self.rng = np.random.default_rng(_seed) # for similar RNG in all instances.
        self.draw_randoms()
        self.historical_data, self.historical_data_len = utils.get_historical_prices(sym, beg_date, fin_date, ta)
        
    # Random draws that will be used    
    def draw_randoms(self):
        if self.AV:
            pos_draws = self.rng.normal(size = (self.x_ta,int(self.NumSimulations/2)))
            self.Z1 = pos_draws
            self.draws = np.concatenate([pos_draws, -pos_draws], axis=1)
            print("generated AV")
        else:
            self.draws = self.rng.normal(size = (self.x_ta,int(self.NumSimulations)))
            print("generated All Random")
        self.sum_draws = np.sum(self.draws, axis=0) / np.sqrt(self.ta_yrs)
        
    
    # assign values for the jump diffusion model
    def setup_jd(self,possion_freq,jump_std,jump_mean):
        self.poisson_rate = possion_freq
        self.std_jump = jump_std
        self.mean_jump = jump_mean
        
    # assign values for the heston model
    def setup_HS(self,_kappa,_theta,_vov,_rho):
        self.kappa = _kappa
        self.theta = _theta
        self.vov = _vov
        self.rho = _rho
    
    def setup_RH(self,_kappa,_theta,_vov,_rho,_hurst):
        self.RH_kappa = _kappa
        self.RH_theta = _theta
        self.RH_vov = _vov
        self.RH_rho = _rho
        self.RH_hurst = _hurst
        
    def step_lognormal(self,index):
        if(index == 0):
            self.prev_vals = np.array(self.NumSimulations*[self.S0])     
        else:
            self.prev_vals = self.prev_vals * np.exp((self.IR-0.5*(self.sig**2))*models.dt+self.sig*np.sqrt(models.dt)*self.draws[index])
        
        return np.mean(self.prev_vals)
    
    
    def step_jump_diff(self,index):
        if(index == 0):
            self.jump_prev_vals = np.array(self.NumSimulations*[self.S0])
            self.k_jump = (np.exp(self.mean_jump+0.5*(self.std_jump**2))-1)
            self.poisson_jumps = self.rng.poisson(lam=models.dt*self.poisson_rate, size=(self.x_ta,self.NumSimulations))
            max_jumps = np.max(self.poisson_jumps)
            all_jumps = self.rng.normal(
                loc=self.mean_jump, 
                scale=self.std_jump,
                size=(self.x_ta, self.NumSimulations, max_jumps)
            )
            mask = np.arange(max_jumps) < self.poisson_jumps[:,:, None]
            self.all_jump_sums = np.sum(all_jumps * mask, axis=2)
            #vectorize approach for speed
            
        else:
            drift = (self.IR-0.5*(self.sig**2) - self.poisson_rate * self.k_jump)
            diffusion = self.sig*np.sqrt(models.dt)
            #possible_jumps = np.array([np.sum(np.random.normal(loc=self.mean_jump, scale=self.std_jump, size=x)) for x in self.poisson_jumps[index]])
            self.jump_prev_vals = self.jump_prev_vals * np.exp(drift *models.dt + diffusion * self.draws[index] + self.all_jump_sums[index])
        return np.mean(self.jump_prev_vals)
        
        
    def step_Heston(self,index):
        if(index == 0):
            self.Heston_prev_vals = np.array(self.NumSimulations*[self.S0])
            self.Heston_prev_vars = np.array(self.NumSimulations*[(self.sig)**2])
            if(self.AV):
                Z2 = self.rng.normal(size=(self.x_ta, int(self.NumSimulations/2)))
                self.CIR_random_draws = np.concatenate([self.rho*self.Z1 + np.sqrt(1-self.rho**2)*Z2,self.rho*(-self.Z1) + np.sqrt(1-self.rho**2)*(-Z2)], axis=1)
                U_all = self.rng.uniform(size=(self.x_ta, int(self.NumSimulations/2)))
                self.U_all = np.concatenate([U_all, 1-U_all], axis=1)
            else: 
                Z2 = self.rng.normal(size=(self.x_ta, int(self.NumSimulations)))
                self.CIR_random_draws = self.rho*self.draws + np.sqrt(1-self.rho**2)*Z2
                self.U_all = self.rng.uniform(size=(self.x_ta, int(self.NumSimulations)))
        else:
            Ms = self.theta+(self.Heston_prev_vars-self.theta)*np.exp(-self.kappa * self.dt)
            s_squares = (self.vov ** 2) * self.Heston_prev_vars * (1 - np.exp(-2 * self.kappa * self.dt)) / (2 * self.kappa)
            psis = s_squares/(Ms**2+1e-10)
            
            v_next = np.empty_like(self.Heston_prev_vars)
            # quadratic normal regime
            mask_qn = psis <= 1.5
            # Bernouli exponential regime
            mask_be = ~mask_qn
            # boolean indexing to find the inputs that follow each regime
            N2 = self.CIR_random_draws[index,:]
            U = self.U_all[index,:]
            # the indicies with following the quadratic normal regime
            if np.any(mask_qn):
                psi_qn = psis[mask_qn]
                M_qn = Ms[mask_qn]
                N2_qn = N2[mask_qn]
                b = np.sqrt((2/psi_qn)-1+np.sqrt(2/psi_qn*(2/psi_qn-1)))
                a = M_qn/(1+b**2)
                v_next[mask_qn] = a * (b+N2_qn) ** 2
                
            # the indicies with following the Bernouli Exponential regime
            if np.any(mask_be):
                psi_be = psis[mask_be]
                M_be = Ms[mask_be]
                U_be = U[mask_be]
                
                p = (psi_be - 1)/(psi_be + 1)
                Beta = (1-p)/M_be
                #to determine which condition it is 
                v_next[mask_be] = np.where(U_be <= p, 0, -np.log((1 - U_be) / (1 - p)) / Beta)
                
            self.Heston_prev_vars = v_next
            
            self.Heston_prev_vals = self.Heston_prev_vals*np.exp((self.IR-(1/2)*self.Heston_prev_vars)*self.dt+np.sqrt(self.dt*self.Heston_prev_vars)*self.draws[index,:])
        return np.mean(self.Heston_prev_vals)
            
    
    # def fractional_brownian_draws(self, _size):
    #     normal_array = np.random.normal(size=(self.NumSimulations, _size)) * (self.dt ** self.RH_hurst)
    #     lags = np.arange(1, _size + 1)
    #     self.gamma_const = gamma(self.RH_hurst + 0.5)
    #     kernel_coefs = (1 /  self.gamma_const) * (lags **  (self.RH_hurst - 0.5))
    #     kernel_coefs = np.insert(kernel_coefs, 0, 0.0)  # K(0) = 0
    #     kernel_coefs_reshaped = kernel_coefs[np.newaxis, :]  # shape (1, ta+1)
        
    #     fbm_increments = fftconvolve(normal_array, kernel_coefs_reshaped, mode='full')[:, :_size]
        
    #     return fbm_increments.T  # Transpose to (ta, NumSimulations)
   
    @staticmethod
    @njit(nogil=True, fastmath=True)
    def RH_Parallel_compute_vol(prev_var,RH_theta,RH_vov):
        variance_floor = max(1e-10, 1e-6 * RH_theta)  # Double protection
        return np.sqrt(np.maximum(prev_var, variance_floor)) if RH_vov > 0 else np.sqrt(prev_var)
        
    @staticmethod
    @njit(nogil=True, parallel=True)
    def RH_Parallel_compute_steps_var(bracket_terms, bs, index, v0):
        conv_sum = np.zeros(bracket_terms.shape[0]) 
        for i in prange(bracket_terms.shape[0]): 
            for j in range(index):
                conv_sum[i] += bracket_terms[i,j] * bs[index - j]  # No temp arrays
                    
        return v0 + conv_sum

    @staticmethod
    @njit(nogil=True)
    def RH_parallel_price_update(prev_vars,steps_vars,IR,dt,index,draws):
        new_prices = np.empty_like(prev_vars)
        sqrt_dt = np.sqrt(dt)
        for i in range(len(prev_vars)):
            adj_var = np.maximum(steps_vars[i], 1e-10)
            adj_vol = np.sqrt(adj_var)
            exponent = np.exp((IR - 0.5 * adj_var) * dt + adj_vol* sqrt_dt * draws[index, i])
            new_prices[i] = prev_vars[i] * exponent

        return new_prices
    
    # def fractional_brownian_draws(self, _size):
    #     with ThreadPoolExecutor(max_workers=8) as executor:
    #         paths = list(executor.map(
    #             lambda i: np.diff(FBM(n=_size + 1, hurst=self.RH_hurst, seed=self.fseed + i).fbm()),
    #             range(self.NumSimulations)
    #         ))
    #     return np.array(paths).T
    
    

    def fractional_brownian_draws(self, _size):
        # Pre-generate seeds from the instance's RNG
        seeds = self.rng.integers(0, 2**32, size=self.NumSimulations)

        with ThreadPoolExecutor(max_workers=8) as executor:
            paths = list(executor.map(
                lambda seed: np.diff(  # Take differences to get increments
                    FractionalBrownianMotion(
                        hurst=self.RH_hurst,
                        t=1,
                        rng=np.random.default_rng(seed)
                    ).sample(_size)  # Sample n+1 points
                ),
                seeds
            ))
        return np.array(paths).T

    def step_R_heston(self, index):
        if index == 0:
            self.gamma_const = gamma(self.RH_hurst + 0.5)
            self.fbm_vol = self.rho * self.draws + np.sqrt(1 - self.rho ** 2) * self.fractional_brownian_draws(self.x_ta)
            self.RH_prev_vals = np.full(self.NumSimulations, self.S0)
            self.v0 = self.sig ** 2
            self.RH_vars = np.full((self.NumSimulations, self.ta), self.v0)
            self.RH_lags = np.arange(1, self.ta + 1) * self.dt
            b = (self.RH_lags ** (self.RH_hurst - 0.5)) /  self.gamma_const
            self.bs = np.insert(b, 0, 0.0)
            self.bracket_terms = np.zeros((self.NumSimulations, self.x_ta))
            self.bracket_terms[:, 0] = (
                self.RH_kappa * (self.RH_theta - self.v0) * self.dt +
                self.RH_vov * np.sqrt(self.v0) * self.fbm_vol[0, :]
            )
        else:
            prev_var = self.RH_vars[:, index - 1]
            
            volatility = self.RH_Parallel_compute_vol(prev_var,self.RH_theta,self.RH_vov)
            
            current_brack  = (
                self.RH_kappa * (self.RH_theta - prev_var) * self.dt +
                self.RH_vov * volatility * self.fbm_vol[index, :]
            )
            self.bracket_terms[:, index]=current_brack
            
            self.RH_vars[:, index] =self.RH_Parallel_compute_steps_var(self.bracket_terms, self.bs, index, self.v0)

            self.RH_prev_vals = self.RH_parallel_price_update(self.RH_prev_vals,self.RH_vars[:,index],self.IR,self.dt,index,self.draws)

        return np.mean(self.RH_prev_vals)

                
       
    def step_historical(self,index):
        if(index<self.historical_data_len):
            return (self.historical_data).iloc[index]
        return None
    
    def show(self):
        print(f"start date: {self.beg_date}")
        print(f"end date: {self.fin_date}")
        print(f"Crypto: {self.sym}")
        print(f"volatility: {self.sig}")
        print(f"srike: {self.K}")
        print(f"S0: {self.S0}")
        print(f"IR: {self.IR}")
        print(f"ta:{self.ta}")
  
  # pathwise is only used for black scholes
   
    def PW_delta(self):
        delta_call = (np.mean(np.exp(-self.IR*self.ta_yrs)*(self.prev_vals>self.K)*(self.prev_vals/self.S0)))        
        delta_put = (np.mean(np.exp(-self.IR*self.ta_yrs)*(self.prev_vals<self.K)*(-self.prev_vals/self.S0)))
        
        return delta_call,delta_put
        
    def PW_rho(self):
        rho_call = (np.mean(np.exp(-self.IR*self.ta_yrs)*(self.prev_vals>self.K)*self.prev_vals*self.ta_yrs))/100
        rho_put = -(np.mean(np.exp(-self.IR*self.ta_yrs)*(self.prev_vals<self.K)*self.prev_vals*self.ta_yrs))/100
        
        return rho_call,rho_put
        
    def PW_vega(self):
        vega_call = 4*(np.mean(np.exp(-self.IR*self.ta_yrs)*(self.prev_vals>self.K)*self.prev_vals*(-self.sig*self.ta_yrs+np.sqrt(self.ta_yrs)*self.sum_draws)))/100
        vega_put = vega_call
        return vega_call,vega_put

    def PW_Theta(self):
        curr_call_option = np.maximum(0,self.prev_vals-self.K)
        curr_put_option =  np.maximum(0,self.K-self.prev_vals)
        theta_call = -np.mean((np.exp(-self.IR*self.ta_yrs))*((self.K<self.prev_vals)*self.prev_vals*((self.sig*self.sum_draws/(2*np.sqrt(self.ta_yrs)))+self.IR-(self.sig**2)/2)-self.IR*curr_call_option))/365
        theta_put = -np.mean((np.exp(-self.IR*self.ta_yrs))*((self.K>self.prev_vals)*self.prev_vals*((-self.sig*self.sum_draws/(2*np.sqrt(self.ta_yrs)))-self.IR+(self.sig**2)/2)-self.IR*curr_put_option))/365
        
        return theta_call,theta_put
    
    # def PW_Theta(self):
    #     curr_call_option = np.maximum(0,self.prev_vals-self.K)
    #     curr_put_option =  np.maximum(0,self.K-self.prev_vals)
        
    #     # Calculate raw thetas (original calculation)
    #     theta_call_raw = -np.mean((np.exp(-self.IR*self.ta_yrs))*((self.K<self.prev_vals)*self.prev_vals*((self.sig*self.sum_draws/(2*np.sqrt(self.ta_yrs)))+self.IR-(self.sig**2)/2)-self.IR*curr_call_option))/365
    #     theta_put_raw = -np.mean((np.exp(-self.IR*self.ta_yrs))*((self.K>self.prev_vals)*self.prev_vals*((-self.sig*self.sum_draws/(2*np.sqrt(self.ta_yrs)))-self.IR+(self.sig**2)/2)-self.IR*curr_put_option))/365
        
    #     # Universal crypto scaling factor
    #     crypto_scale = 3.0 + 2.0 * np.sqrt(self.sig / 0.8)  # Base 3x + vol adjustment
        
    #     # Scale to match other models
    #     theta_call = theta_call_raw * crypto_scale
    #     theta_put = theta_put_raw * crypto_scale
        
    #     return theta_call, theta_put
    
    def price_option(self,model_name):
        call, put = 0 , 0
        match model_name: 
            case "BLS":
                call = np.exp(-self.IR * self.ta_yrs)*np.mean(np.maximum(0,self.prev_vals-self.K))
                put = np.exp(-self.IR * self.ta_yrs)*np.mean(np.maximum(0,self.K-self.prev_vals))
            case "JUD":
                call = np.exp(-self.IR * self.ta_yrs)*np.mean(np.maximum(0,self.jump_prev_vals-self.K))
                put = np.exp(-self.IR * self.ta_yrs)*np.mean(np.maximum(0,self.K-self.jump_prev_vals))
            case "HES":
                call = np.exp(-self.IR * self.ta_yrs)*np.mean(np.maximum(0,self.Heston_prev_vals-self.K))
                put = np.exp(-self.IR * self.ta_yrs)*np.mean(np.maximum(0,self.K-self.Heston_prev_vals))
            case "RHS":
                call = np.exp(-self.IR * self.ta_yrs)*np.mean(np.maximum(0,self.RH_prev_vals-self.K))
                put = np.exp(-self.IR * self.ta_yrs)*np.mean(np.maximum(0,self.K-self.RH_prev_vals))
            
        return call,put
            



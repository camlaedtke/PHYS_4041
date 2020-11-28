import os
import numpy as np
from scipy.integrate import odeint

def generate_damped_pendulum_solution(N_SAMPLES=25000, 
                                      α=0.1, 
                                      β=8.91, 
                                      Δ=0.1,
                                      x0=[-1.193, -3.876], 
                                      one_step=False, 
                                      t_end=25, 
                                      noise=0.0):
    
    def damped_pendulum(y, t, α, β):
        θ, dθdt = y     
        return dθdt, -α*dθdt - β*np.sin(θ)
    
    if not one_step:
        t = np.linspace(0, t_end, 100000)
        sol = odeint(damped_pendulum, x0, t, args=(α, β))
        x = sol[:,0]
        dxdt = sol[:,1]
        return t, x, dxdt
    else:
        
        t = np.linspace(0, Δ, 10000)
    
        X = np.zeros((N_SAMPLES, 2))
        Y = np.zeros((N_SAMPLES, 2))

        for i in range(0,N_SAMPLES):
            if i%100==0:
                print("\r generating {} / {}".format(i, N_SAMPLES), end='')

            θ_0 = np.random.uniform(-np.pi, np.pi)
            dθdt_0 = np.random.uniform(-2*np.pi, 2*np.pi)
            x0 = [θ_0, dθdt_0]

            sol = odeint(damped_pendulum, x0, t, args=(α, β))
            ε_1 = np.random.uniform(-noise, noise)
            ε_2 = np.random.uniform(-noise, noise)

            θ_Δ = sol[-1,0]
            dθdt_Δ = sol[-1,1]

            X[i,0] = θ_0 + ε_1
            X[i,1] = dθdt_0 + ε_1

            Y[i,0] = θ_Δ + ε_2
            Y[i,1] = dθdt_Δ + ε_2

        return X, Y
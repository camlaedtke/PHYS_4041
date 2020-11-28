import numpy as np
from typing import List, Tuple



class SGD:
    
    def __init__(self, lr):
        """
        :param lr - learning rate
        """
        self._lr = lr

    def update(self, params_values, grads_values, nn_architecture):
        
        # iteration over network layers
        for layer_idx, layer in enumerate(nn_architecture, 1):
            params_values["W" + str(layer_idx)] -= self._lr * grads_values["dW" + str(layer_idx)] 
            params_values["b" + str(layer_idx)] -= self._lr * grads_values["db" + str(layer_idx)]
            
        return params_values
    
    
    
class Adam:
    
    def __init__(self, lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        """
        :param lr - learning rate
        :param beta1 -
        :param beta2 -
        :param eps - small value to avoid zero denominator
        """
        self._cache_v = {}
        self._cache_s = {}
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps

    def update(self, params_values, grads_values, nn_architecture):
        
        if len(self._cache_s) == 0 or len(self._cache_v) == 0:
            self._init_cache(grads_values, nn_architecture)
        
        for layer_idx, layer in enumerate(nn_architecture, 1):
            
            w = params_values["W" + str(layer_idx)]
            b = params_values["b" + str(layer_idx)]
            
            dw = grads_values["dW" + str(layer_idx)]        
            db = grads_values["db" + str(layer_idx)]
            
            dw_key, db_key = Adam._get_cache_keys(layer_idx)
            
            self._cache_v[dw_key] = self._beta1 * self._cache_v[dw_key] + (1 - self._beta1) * dw
            self._cache_v[db_key] = self._beta1 * self._cache_v[db_key] + (1 - self._beta1) * db

            self._cache_s[dw_key] = self._beta2 * self._cache_s[dw_key] + (1 - self._beta2) * np.square(dw)
            self._cache_s[db_key] = self._beta2 * self._cache_s[db_key] + (1 - self._beta2) * np.square(db)

            dw = self._cache_v[dw_key] / (np.sqrt(self._cache_s[dw_key]) + self._eps)
            db = self._cache_v[db_key] / (np.sqrt(self._cache_s[db_key]) + self._eps)
            
            params_values["W" + str(layer_idx)] = w - self._lr * dw   
            params_values["b" + str(layer_idx)] = b - self._lr * db
            
        return params_values
            
            
    def _init_cache(self, grads_values, nn_architecture):
        
        for layer_idx, layer in enumerate(nn_architecture, 1):
        
            dw = grads_values["dW" + str(layer_idx)]        
            db = grads_values["db" + str(layer_idx)]
            
            dw_key, db_key = Adam._get_cache_keys(layer_idx)
            
            self._cache_v[dw_key] = np.zeros_like(dw)
            self._cache_v[db_key] = np.zeros_like(db)
            self._cache_s[dw_key] = np.zeros_like(dw)
            self._cache_s[db_key] = np.zeros_like(db)


    @staticmethod
    def _get_cache_keys(idx: int) -> Tuple[str, str]:
        """
        :param idx - index of layer
        """
        return f"dw{idx}", f"db{idx}"
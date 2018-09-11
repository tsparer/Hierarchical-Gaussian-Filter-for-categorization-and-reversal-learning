# update equations #

from math import *
# List of Terms #

def s(mu_2):
        ans = ( 1/ (1+exp(-(mu_2))) )
        #print ans
        return ans

def sig_hat_1_k_calc(mu_hat_1_k_min_1):
        sig_hat_1 = (mu_hat_1_k_min_1)*(1- mu_hat_1_k_min_1)
        # double check subtraction, 2 or 1?  #
        return sig_hat_1


def sig_hat_2_k_calc (sig_2_k_min_1,  mu_3_k_min_1, ka, om): 
        sig_hat_2 = sig_2_k_min_1 + exp(ka*mu_3_k_min_1 + om) 
        return sig_hat_2

    

def sig_2_k_calc(sig_hat_2_k, sig_hat_1_k):
        sig_2 = 1/((1/sig_hat_2_k) + sig_hat_1_k)
        return sig_2



def update_mu_2 (mu_2_k_min_1, sig_2_k, mu_1_k):  #mu_1_k is actual observation"
        mu_2_k = mu_2_k_min_1 +  ( sig_2_k * (mu_1_k - s(mu_2_k_min_1)) )
        return mu_2_k


def update_mu_3 (mu_3_k_min_1, sig_3_k, sig_2_k_min_1, ka, om, sig_2_k, mu_2_k,
                 mu_2_k_min_1):
    
        pred_k_min_1 = mu_3_k_min_1
        
        rate = (sig_3_k *
                         (ka/2)* ( exp(ka*mu_3_k_min_1 + om)/
                                (  sig_2_k_min_1 + exp(ka*mu_3_k_min_1 + om))) )                

        pred_error = ( ((sig_2_k + (mu_2_k - mu_2_k_min_1)**2) /
                      (sig_2_k_min_1 + exp(ka*mu_3_k_min_1 + om)) ) - 1)

        mu_3_k = pred_k_min_1 + (rate * pred_error)

        return mu_3_k
    



#use the update equations to derive new estimates for mu3 and mu2?
#Either plug mu2 directly into sig to generate prediction
# or somehow plug ino original generative model? (Not really)
# equation 27 connects level three to level 2 by way of estimating sig_2

#check division, ints v. floats etc.



import hgf_online_updating_w_test_graph_2 as updates
import hgf_bin_update_equations as u_e



class update_single:

    def __init__ (self, var_name, mu_2_k_min_1, sig_2_k_min_1, mu_hat_1_k_min_1,
             mu_3_k_min_1, kappa, omega, sig_3_k):


            self.var_name = var_name

            self.mu_hat_1_k_min_1 = mu_hat_1_k_min_1
            
            self.mu_2_k_min_1  = mu_2_k_min_1
            self.sig_2_k_min_1 = sig_2_k_min_1
            
            self.mu_3_k_min_1 =  mu_3_k_min_1
            self.sig_3_k      =  sig_3_k

            self.kappa =  kappa
            self.omega =  omega

            self.counter = 0    #for tracking number of times update is called



    def update(self, obs):

        self.obs = obs


        self.mu_hat_1_k   = u_e.s(self.mu_2_k_min_1)
    

        self.sig_hat_1_k   = u_e.sig_hat_1_k_calc (self.mu_hat_1_k_min_1)
    

        self.sig_hat_2_k   = u_e.sig_hat_2_k_calc (self.sig_2_k_min_1,
                                                   self.mu_3_k_min_1,
                                                   self.kappa,
                                                   self.omega)
    

        self.sig_2_k       = u_e.sig_2_k_calc (self.sig_hat_2_k,
                                               self.sig_hat_1_k)
    

        self.mu_2_k        = u_e.update_mu_2 (self.mu_2_k_min_1,
                                              self.sig_2_k,
                                              self.obs)
    

        self.mu_3_k          = u_e.update_mu_3 (self.mu_3_k_min_1,
                                              self.sig_3_k,
                                              self.sig_2_k_min_1,
                                              self.kappa,
                                              self.omega,
                                              self.sig_2_k,
                                              self.mu_2_k,
                                              self.mu_2_k_min_1)

        self.counter       = self.counter + 1
        
        
        # resets initial values to new values (the "minus" is relative to
        # upcoming/future trials   ###


        self.mu_hat_1_k_min_1 = self.mu_hat_1_k

        self.mu_2_k_min_1     = self.mu_2_k
        self.sig_2_k_min_1    = self.sig_2_k

        self.mu_3_k_min_1     =  self.mu_3_k
        


    def get_predicted_values(self):
        return [self.mu_hat_1_k_min_1, self.mu_2_k_min_1, self.sig_2_k_min_1,
                self.mu_3_k_min_1  ]
    

    def get_sig_hat_1(self):
        sig_hat_1 = self.sig_hat_1_k
        return sig_hat_1


    def get_sig_2_k(self):
        sig_2= self.sig_2_k_min_1
        return sig_2

    def get_mu_3(self):
        mu_3 = self.mu_3_k_min_1
        return mu_3

    def get_mu_2(self):
        mu_2 = self.mu_2_k_min_1
        return mu_2
    

    def get_mu_predicted(self):
        mu_predicted = self.mu_hat_1_k_min_1
        return mu_predicted

    

        
    

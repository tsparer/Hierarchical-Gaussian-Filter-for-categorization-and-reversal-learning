import hgf_bin_update_equations as u_e

import numpy
import pylab

def update_all (mu_2_k_min_1, sig_2_k_min_1, mu_hat_1_k_min_1, mu_3_k_min_1,
            ka, om, sig_3_k, observation):

    #  Calculate updates  #
   # print ""
    #print "mu 2 k min 1", mu_2_k_min_1

    mu_hat_1_k = u_e.s(mu_2_k_min_1)
    #print "mu hat 1 k", mu_hat_1_k

    sig_hat_1_k = u_e.sig_hat_1_k_calc (mu_hat_1_k_min_1)
    #print "sig hat 1 k", sig_hat_1_k

    sig_hat_2_k = u_e.sig_hat_2_k_calc (sig_2_k_min_1, mu_3_k_min_1, ka, om)
    #print "sig hat 2 k", sig_hat_2_k

    sig_2_k     = u_e.sig_2_k_calc (sig_hat_2_k, sig_hat_1_k)
    #print "sig_2_k" , sig_2_k

    mu_2_k      = u_e.update_mu_2 (mu_2_k_min_1, sig_2_k, observation)
    # in update equations file observation reads mu_1_k)

    mu_3_k      = u_e.update_mu_3 (mu_3_k_min_1, sig_3_k, sig_2_k_min_1, ka, om,
                               sig_2_k, mu_2_k, mu_2_k_min_1)
    #print "mu 3 k       " , mu_3_k

    predicted_mu_1 = mu_hat_1_k         #  used mostly for reference  #

    return (mu_2_k, sig_2_k, mu_hat_1_k, mu_3_k,
            ka, om, sig_3_k, predicted_mu_1)


###DOUBLE CHECK!!! ABOVE RETURN VALUES!!!  ###
    #  Return adjusted values as tuple #

    # Following notes to be carried out in next function?  #
    #  Make Prediction  #
    #  e.g. s(mu_2_k)   #



    #  Record Prediction  #



    #  Record other values for data collection  #



    #  Increment values  #



  




def update_w_observations(obs):

    trial_number = 1
    
    mu_2_k_min_1 = 0
    
    sig_2_k_min_1 = 1
    
    mu_hat_1_k_min_1 = 0
    
    mu_3_k_min_1 = 0
    
    ka = 1.4
    
    om = -2.2
    
    sig_3_k = .5

    obs_in = obs        # <-- input data as array  #

    predictions_record = [0.0 for i in range (len(obs))] 

    param = []


    for i in range (len (obs)):
      
        #print ""
        #print trial_number
        #print ""
        trial_number = trial_number+1
        
        observation = obs_in[i]

        #  unpack tuple returned by update all to update values  #
        #  parens needed?  #

        

            # make sure to update prediction at mu_1 level as well!  #
            
        all_pred = update_all (mu_2_k_min_1, sig_2_k_min_1, mu_hat_1_k_min_1,
                               mu_3_k_min_1, ka, om, sig_3_k, observation)

        # print all_pred  # tuple of new values


        a,b,c,d,e,f,g,h = all_pred # placeholders for tuple unpacking

        mu_2_k_min_1 = a
        #print "mu_2_k_min_1"
        #print mu_2_k_min_1
        #print ""
        #print ""
        
        sig_2_k_min_1 = b
        
        mu_hat_1_k_min_1 = c
        
        mu_3_k_min_1 = d

        ka = e
        
        om = f
        
        sig_3_k = g
        
        predicted_mu_1 = h

        #if trial_number == 10: #lame kludge kill switch
         #   return   

        # mu_2=1


        #mu_2_k_min_1, sig_2_k_min_1, mu_hat_1_k_min_1, mu_3_k_min_1,
        #ka, om, sig_3_k, predicted_mu_1 = all_pred

        

        #print "predicted mu 1"
        #print predicted_mu_1
        #print ""
        




        predictions_record[i] = predicted_mu_1
            #may need to adjust pred_mu to make only binary predictions #


            #params, other #

       # print "predictions record"
       # print predictions_record
        #print " "

    return predictions_record



def main():

    number_of_data_points = 0

    data_file = open('test_data', 'r')
    obs = []
    results =[]
    perc_error_list = []


    for line in data_file:
        line = float(line)
        obs.append(line)
        number_of_data_points = number_of_data_points + 1

    #print number_of_data_points
    #print obs
    
    predicted = update_w_observations(obs)
    
    

    for i in range (len(predicted)):
        results.append ([predicted[i],obs[i]])
        perc_error = abs((obs[i] - predicted[i])/100)
        perc_error_list.append (round(perc_error,3))
        #print perc_error


    
    x_axis = numpy.arange(0,number_of_data_points,1)
    y_axis_predicted = numpy.array(predicted)
    y_axis_actual = numpy.array(obs)
    
    pylab.plot(x_axis, y_axis_actual, 'ro', x_axis, y_axis_predicted, 'g')
    pylab.show()

    

    return 0

 
            

        

            

        

                               

    

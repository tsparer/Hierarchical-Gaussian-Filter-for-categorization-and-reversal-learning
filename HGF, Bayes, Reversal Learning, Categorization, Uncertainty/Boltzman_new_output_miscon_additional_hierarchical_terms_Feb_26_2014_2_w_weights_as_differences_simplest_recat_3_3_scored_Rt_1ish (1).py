import category_and_learner_objects_3 as cl
import update_object_2_2 as u_o
import numpy
import pylab
import random
import math

def init_multi_HGF(
                        num_feat, kappa = 1.4, omega = -2.2,
                        sig_3_k = .5, mu_2_k_min_1 = 0, sig_2_k_min_1 = 1,
                        mu_hat_1_k_min_1 = .1, mu_3_k_min_1 = 0
                     ):

    HGF_list = [0.0 for i in range(num_feat)]


    for i in range(num_feat):
                     
        HGF_list[i] = u_o.update_single(i, mu_2_k_min_1,
                                    sig_2_k_min_1, mu_hat_1_k_min_1, mu_3_k_min_1,
                                        kappa, omega,sig_3_k )
    

    return HGF_list




def update_multi_HGF(inst, HGF_list):

    
    for i in range (len(inst)):
        feat = inst[i]
        HGF_list[i].update(feat)



def get_mu_hat_HGF_list(HGF_list):
    
    list_len = len(HGF_list)
    
    mu_hat_list = [0.0 for i in range (list_len) ]

    for i in range(list_len):
        mu_hat = HGF_list[i].get_mu_predicted()
        mu_hat_list[i] = mu_hat



    return mu_hat_list



def get_mu_2_HGF_list(HGF_list):
    
    list_len = len(HGF_list)
    
    mu_2_list = [0.0 for i in range (list_len) ]

    for i in range(list_len):
        mu_2 = HGF_list[i].get_mu_2()
        mu_2_list[i] = mu_2



    return mu_2_list


def get_mu_3_HGF_list(HGF_list):
    
    list_len = len(HGF_list)
    
    mu_3_list = [0.0 for i in range (list_len) ]

    for i in range(list_len):
        mu_3 = HGF_list[i].get_mu_3()
        mu_3_list[i] = mu_3



    return mu_3_list



def get_sig_2_HGF_list(HGF_list):
    
    list_len = len(HGF_list)
    
    sig_2_list = [0.0 for i in range (list_len) ]

    for i in range(list_len):
        sig_2 = HGF_list[i].get_sig_2_k()
        sig_2_list[i] = sig_2



    return sig_2_list




def n_bayes_mult(inst, F_given_C, action_bias = 1, c_prior = .5):

    # action_bias is a special parameter tacked on at the end

 #   print inst


    adj = [0 for i in range(len(inst))]

    for i in range(len(inst)):
        if inst[i] == 1:
            adj[i] = 1

        elif inst[i] == 0:
            adj[i] = .001

  #  print adj

    pr_c_given_f = c_prior     #to be multiplied shortly

    for i in range(len(inst)):
        current_f = adj[i] * F_given_C[i]

        pr_c_given_f = pr_c_given_f * current_f


    pr_c_given_f = pr_c_given_f * action_bias


    return pr_c_given_f




def nec(p, q):
  
    if (p == 1 and q == 1):
        evaluate = 1

    else:
        evaluate = 0

    return evaluate

def rev(inst):
    reve = [0 for i in range(len(inst))]

    for i in range(len(inst)):
        if inst[i] == 1:
            reve[i] = 0

        elif inst[i] == 0:
            reve[i] = 1

    return reve

    




def nec_loop(p, q):

    ans = [0 for i in range(len(p))]

    for i in range(len(p)):
       score = nec(p[i], q)

       ans[i] = score

    #print ans
    return ans


def cat_def(inst, def_1, def_2 = 0):

    if def_2 != 0:

        for i in range(len(inst)):
            if i == def_1:
                if inst[i] == 0:
                    return 0

            elif i == def_2:
                if inst[i] == 0:
                    return 0


        return 1
        
        


    else:
        if inst[def_1] == 0:
            return 0
        else:
            return 1


def score(inst, f_given_blank):
    score = [0 for i in range(len(inst))]

    for i in range(len(inst)):

        if inst[i] == 1:
            score[i] = float(f_given_blank[i])

        elif inst[i] == 0:
            score[i] = float(1 - f_given_blank[i])

    return score


def gen_inst(length):
    inst = [0 for i in range(length)]

    for i in range(length):
        inst[i] = random.randint(0,1)

    return inst
        



def test(num_trials):


    features = ['a','b','c','d', 'e']

    definition = ['n',0,0,0, 0]

    HGF_list_in = init_multi_HGF(len(features))

    HGF_list_not= init_multi_HGF(len(features))

    cat = cl.category(1, features, definition)




    miscon_weight_for_res = []  # formerly one_cat
    miscon_weight_for_tol = []   # formerly one_not

    targ_weight_for_res = []    # formerly targ_in
    targ_weight_for_tol = []    # formerly targ_not

    neut_f_weight_res =   []    #formerly rand_in =  []   # for comparison
    neut_f_weight_tol =   []    #formerly rand_not = []



    #additional misconception parameters

    miscon_mu2_for_res = []    #unbounded feature weight
    miscon_mu2_for_tol = []   

    miscon_sig2_for_res = [] # learning rate, precision weighted 
    miscon_sig2_for_tol = [] # learning rate, precision weighted

    miscon_mu3_for_res = []  #(inferred) volatility (variance of variance)
    miscon_mu3_for_tol = []  #(inferred) volatility (variance of variance)

    
    #additional target parameters

    targ_mu2_for_res = []    #unbounded feature weight
    targ_mu2_for_tol = []

    targ_sig2_for_res = []    # learning rate, precision weighted
    targ_sig2_for_tol = []    # learning rate, precision weighted

    targ_mu3_for_res = []    #(inferred) volatility (variance of variance)
    targ_mu3_for_tol = []    #(inferred) volatility (variance of variance)




    cat_actual = []
    action_chosen = []
    in_cat_list = []
    not_in_cat_l =[]

    

    

    last_act_list = []

    last_act = 1

    action_in_mu = []
    action_not_mu =[]

    likelihoods_as_subjective_weights = []

    RTS = []

    for i in range(num_trials):

        #print '\n\n', i

        inst = gen_inst(5)

        if i <= 60:
            feedback = cat_def(inst, 1)

        else:
            feedback = cat_def(inst, 3)




#        action_impact = random.random()

 #       if (action_impact >= .1) and (last_act == 0) and (feedback == 1):

  #          feedback = 0


        likelihoods_in = get_mu_hat_HGF_list(HGF_list_in)
        #print "likelihoods" , likelihoods_in

        scores_in = score(inst, likelihoods_in)

        miscon_weight_for_res.append(likelihoods_in[1])

        likelihoods_not =  get_mu_hat_HGF_list(HGF_list_not)


        scores_not = score(inst, likelihoods_not)

        miscon_weight_for_tol.append(likelihoods_not[1])


        choose_not_cat = n_bayes_mult(inst, scores_not)
        #print "marg in cat", choose_not_cat
        choose_in_cat = n_bayes_mult(inst, scores_in)
        #print "marg not in cat",choose_in_cat

        neut_f_weight_res.append(likelihoods_in[2])   
        neut_f_weight_tol.append(likelihoods_not[2])

        targ_weight_for_res.append(likelihoods_in[3])
        targ_weight_for_tol.append(likelihoods_not[3])

        Boltz_denom = choose_not_cat + choose_in_cat

        Boltz_in  = choose_in_cat/Boltz_denom
        Boltz_not = choose_not_cat/Boltz_denom

        random_action_compare = random.random()

        if random_action_compare <= Boltz_in:
            guess = 1   
        else:
            guess = 0

        if guess == 1:
            chosen_action = 1
                
        elif guess == 0:
                chosen_action = 0




               
        cat_actual.append(feedback)
        action_chosen.append(chosen_action)
        in_cat_list.append(choose_in_cat)
        not_in_cat_l.append(choose_not_cat)
    

        num_feat = len(inst)

        if feedback == 1:
            update_multi_HGF(inst, HGF_list_in)

        elif feedback == 0:
            update_multi_HGF(inst, HGF_list_not)


            #additional parameter values
   
        list_mu2_res = get_mu_2_HGF_list(HGF_list_in)
        list_mu2_tol = get_mu_2_HGF_list(HGF_list_not)

        list_sig2_res = get_sig_2_HGF_list(HGF_list_in)
        list_sig2_tol = get_sig_2_HGF_list(HGF_list_not)
        
        list_mu3_res  = get_mu_3_HGF_list(HGF_list_in)
        list_mu3_tol  = get_mu_3_HGF_list(HGF_list_not)
        

            #additional misconception parameters

        miscon_mu2_for_res.append(list_mu2_res[1])    #unbounded feature weight
        miscon_mu2_for_tol.append(list_mu2_tol[1])

        miscon_sig2_for_res.append(list_mu3_res[1])    # learning rate, precision weighted 
        miscon_sig2_for_tol.append(list_mu3_tol[1])     # learning rate, precision weighted

        miscon_mu3_for_res.append(list_sig2_res[1])     #(inferred) volatility (variance of variance)
        miscon_mu3_for_tol.append(list_sig2_tol[1])     #(inferred) volatility (variance of variance)

        
        #additional target parameters

        targ_mu2_for_res.append(list_mu2_res[3])    #unbounded feature weight
        targ_mu2_for_tol.append(list_mu2_tol[3])

        targ_sig2_for_res.append(list_sig2_res[3])      # learning rate, precision weighted
        targ_sig2_for_tol.append(list_sig2_tol[3])      # learning rate, precision weighted

        targ_mu3_for_res.append(list_mu3_res[3])        #(inferred) volatility (variance of variance)
        targ_mu3_for_tol.append(list_mu3_tol[3])       #(inferred) volatility (variance of variance)



        subjective_weights = [0 for k in range(len(likelihoods_in))]

        for k in range(len(likelihoods_in)):
            w_in =  (likelihoods_in[k] -.5)   *2
            w_not = (likelihoods_not[k] - .5) *2
            
            subjective_weights[k] = ( abs(w_in) + abs(w_not) )   /2


        
  ################################################################

        if (i == 1 or i == 15 or i == 30 or i == 45 or i == 59 or i == 61 or i == 75 or
            i == 90 or i == 105 or i == 119):

            likelihoods_as_subjective_weights.append(subjective_weights)

            
            

    #print likelihoods_as_subjective_weights   

    outfile = open("feature_weights.txt", "w")

    for i in range(len(likelihoods_as_subjective_weights)):
        weight_string = str(likelihoods_as_subjective_weights[i])
        outfile.write( weight_string + ",  \n\n")

    outfile.close()

    # Records new parameter values
            
    check = []
    for i in range(len(cat_actual)):
        if cat_actual[i] == action_chosen[i]:
            c = 1
        else:
            c = 0
        check.append(c)

    #print check

    pos_tot = 0

    for i in range(len(check)):
        if check[i] == 1:
            pos_tot = pos_tot + 1

    perc_pos = float(pos_tot)/len(cat_actual)
   # print pos_tot, "out of", len(cat_actual)

    #print perc_pos




    ###    HGF_list_in  = resistant

    ###     HGF_list_not =  tolerant




    


    x_axis = numpy.arange(0,num_trials,1)
    y_axis = numpy.arange (0, num_trials)

    y_miscon_mu2_for_res  =   numpy.array(miscon_mu2_for_res)
    y_miscon_mu2_for_tol  =   numpy.array(miscon_mu2_for_tol)

    y_miscon_sig2_for_res =   numpy.array(miscon_sig2_for_res)
    y_miscon_sig2_for_tol =   numpy.array(targ_sig2_for_tol)

    y_miscon_mu3_for_res =    numpy.array(miscon_mu3_for_res)
    y_miscon_mu3_for_tol =    numpy.array(miscon_mu3_for_tol)


    #print x_axis, y_axis, y_targ_mu2_for_res,
    #print y_targ_mu2_for_tol,

    #print y_targ_sig2_for_res,
    #print  y_targ_sig2_for_tol ,

    #print y_targ_mu3_for_res ,
    #print y_targ_mu3_for_tol, 


    pylab.plot(
                x_axis,
                y_miscon_mu2_for_res, 'r' ,
                y_miscon_mu2_for_tol,  'r--',

                y_miscon_sig2_for_res, 'g', 
                y_miscon_sig2_for_tol , 'g--',

                y_miscon_mu3_for_res , 'b',
                y_miscon_mu3_for_tol, 'b--',
                )
    pylab.show()





    x_axis = numpy.arange(0,num_trials,1)
    
    y_axis_actual_1        =  numpy.array(check)
    

    y_axis_one_cat         =  numpy.array(miscon_weight_for_res)
    y_axis_not_in_cat      =  numpy.array(not_in_cat_l)
    y_axis_one_not         =  numpy.array(miscon_weight_for_tol)

    y_rand_in  = numpy.array(neut_f_weight_res)

    
#    y_act_bias = numpy.array(last_act_list)

    y_targ_in  = numpy.array(targ_weight_for_res)
    y_targ_not = numpy.array(targ_weight_for_tol)

    #y_axis_actual_2 = numpy.array(actual_2)
    #y_axis_predicted_2 = numpy.array(test_2)

    # "y_axis_prob_action_in, 'g--',"
    #  " y_axis_prob_action_not, 'r--'"
    # RTS, 'k'

    
    #pylab.plot(x_axis, y_axis_actual_1, 'ro',  
    #          y_axis_one_cat, 'b--',
    #          y_axis_one_not, 'c--', 
    #          y_targ_in, 'm--', y_targ_not, 'y--')
    #pylab.show()


    test_out = []

    test_out.append(check)
    test_out.append(miscon_weight_for_res)
    test_out.append(miscon_weight_for_tol)
    test_out.append(targ_weight_for_res)
    test_out.append(targ_weight_for_tol)
    test_out.append(neut_f_weight_res)
    test_out.append(neut_f_weight_tol)

    return test_out
   
    

def average_2_list(list_1, list_2):


# at least 2 x 2 lists, of equal sizes

    output_list = [0 for i in range(len(list_1))]

    for i in range(len(list_1)):
        n_1 = list_1[i]
        n_2 = list_2[i]
        avg = (float(n_1 + n_2))/2

        output_list[i] = avg
                   
            
    return output_list

def average_2_matrix(mat_1, mat_2):

    output_mat = [0 for i in range (len(mat_1))]

    for i in range(len(mat_1)):
        avg_list = average_2_list(mat_1[i], mat_2[i])

        output_mat[i] = avg_list

    return output_mat
    


def multi_test_average(num_runs, num_trials = 120, num_test_outputs = 7):

    avg_output = [[0.0 for j in range (num_trials)]for i in range(num_test_outputs)]

    for i in range(num_runs):
        run_output = test(num_trials)

        avg_output = average_2_matrix(avg_output, run_output)

    avg_correct                 = avg_output[0]
    avg_miscon_weight_for_res   = avg_output[1]
    avg_miscon_weight_for_tol   = avg_output[2]
    avg_targ_weight_for_res     = avg_output[3]
    avg_targ_weight_for_tol     = avg_output[4]
    avg_neut_f_weight_res       = avg_output[5]
    avg_neut_f_weight_tol       = avg_output[6]


    x_axis = numpy.arange(0,num_trials,1)
    
    y_axis_actual_1        =  numpy.array(avg_correct)
    

    y_axis_one_cat         =  numpy.array(avg_miscon_weight_for_res)
    #y_axis_not_in_cat      =  numpy.array(not_in_cat_l)
    y_axis_one_not         =  numpy.array(avg_miscon_weight_for_tol)

    y_rand_in  = numpy.array(avg_neut_f_weight_res)

    
#    y_act_bias = numpy.array(last_act_list)

    y_targ_in  = numpy.array(avg_targ_weight_for_res)
    y_targ_not = numpy.array(avg_targ_weight_for_tol)

    pylab.plot(x_axis, y_axis_actual_1, 'ro',  
              y_axis_one_cat, 'b--',
              y_axis_one_not, 'c--', 
              y_targ_in, 'm--', y_targ_not, 'y--')
    pylab.show()
    

    #return avg_output
        

def average_all():
    x = 0

    h = [[1,1][2,2]]
    g = [[1,1],[7,8]] 
def graph_run(num_learners, num_runs):
    x = 0

    #initialize average?

    for i in range(num_learners):
        current_res = test(num_runs)
        average = average_all(average, current_res)



    #graph average



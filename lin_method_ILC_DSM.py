
import numpy as np
import sys
from scipy import signal, linalg
import math
from balreal import balreal



def generate_dac_output(C, ML):
    if C.shape[0] > ML.shape[0]:
        print(C.shape[0])
        print(ML.shape[0])
        raise ValueError('Not enough channels in model.')

    Y = np.zeros(C.shape)
    
    match 2:
        case 1: # use loops
            for k in range(0,C.shape[0]):
                for j in range(0,C.shape[1]):
                    c = C[k,j]
                    ml = ML[k,c]
                    Y[k,j] = ml
        case 2: # use numpy indexing
            for k in range(0,C.shape[0]):
                Y[k,:] = ML[k,C[k,:]]
        
    return Y


def nsq(Nb, X, Dq, b, a,  Qstep, Vmin, YQns):
    """
    Xcs - Input signal
    b, a    - Filter transfer functions
    lvl_dict    - quantizer levels
    """
    
    # Noise-shaping filter (using a simple double integrator)
    # Hns_tf = signal.TransferFunction([1, -2, 1], [1, 0, 0], dt=1)  # double integrator
    # Mns_tf = signal.TransferFunction([2, -1], [1, 0, 0], dt=1)  # Mns = 1 - Hns
    Mns_tf = signal.TransferFunction(b-a, a, dt=1)  # Mns = 1 - Hns
    Mns = Mns_tf.to_ss()
    Ad, Bd, Cd, Dd = balreal(Mns.A, Mns.B, Mns.C, Mns.D)

    C = np.zeros((1, X.size)).astype(int)  # save codes

    satcnt = 0  # saturation counter (generate warning if saturating)
    
    FB_ON = True  # turn on/off feedback, for testing
    # Initialise state, output and error
    xns = np.zeros((Ad.shape[0], 1))  # noise-shaping filter state
    yns = np.zeros((1, 1))  # noise-shaping filter output
    e = np.zeros((1, 1))  # quantiser error

    for i in range(0, X.size):
        x = X[i]
        d = Dq[i]

        if FB_ON: w = x - yns[0, 0]  # use feedback
        else: w = x

        u = w + d

        # requantization
        # Re-quantizer (mid-tread)
        q = math.floor(u/Qstep + 0.5)  # quantize
        c = q - math.floor(Vmin/Qstep)  # code

        # Saturation (can't index out of bounds)
        if c > 2**Nb - 1:
            c = 2**Nb - 1
            # satcnt = satcnt + 1
            # if satcnt >= 10:
            #     print(f'warning: pos. sat. -- cnt: {satcnt}')
            
        if c < 0:
            c = 0
            # satcnt = satcnt + 1
            # if satcnt >= 10:
            #     print(f'warning: neg. sat. -- cnt: {satcnt}')
        
        C[0, i] = c  # save code
        # Output models
        yi = YQns[c]  # ideal levels
        e[0] = yi - w

        # Noise-shaping filter
        xns = Ad@xns + Bd@e  # update state
        yns = Cd@xns  # update filter output
    return C
    

def get_ILC_control(Nb, Xcs, Dq,  Q, L, G, itr, b, a, Qstep, Vmin,  Qtype, YQ, ILns):

    Xcs = Xcs.reshape(-1,1)
    Dq = Dq.reshape(-1,1)
    Xcs = Xcs + Dq


    u = np.zeros_like(Xcs)
    # U = u

    # Store Codes
    C = []

    u_dsm_inp = Xcs + u
    # U_DSM_INP = u_dsm_inp
    # Intial Output
    y = G @ u
    Y = y

    # Initial Error 
    e = Xcs - y     # Reference/test signal - output signal
    # E = e

    for i in range(itr):

        # Delta-sigma modulation / return code

        u_dsm_c = nsq(Nb, u_dsm_inp, Dq,  b, a,  Qstep, Vmin, YQ.squeeze())
        u_dsm_out = generate_dac_output(u_dsm_c, ILns.reshape(1,-1))


        # ILC
        # Quantize control
        q_u_new = direct_quantization(u_dsm_out, Qstep, YQ, Qtype)               
        # Convert quantized signal to code
        q_u_new_code = generate_code(q_u_new, Qstep, Vmin, Qtype).squeeze()
        # Assing measured levels according to the code
        q_u_new_dac = generate_dac_output(q_u_new_code.reshape(1,-1), ILns.reshape(1,-1))
        q_u = np.array(q_u_new_dac).reshape(-1,1)

        # Output 
        y = G @ q_u
        y = y.reshape(-1,1)

        # Error 
        e = Xcs - y 

        # Calculate new feed forwared         
        u_new = Q @ (u + L @ e)       
        u = u_new 

        # Update control using ILC algorithm, Q filter matrix and Learning matrix
        u_dsm_inp = Xcs + u

        # Store values 
        # Y = np.hstack((Y, y))
        # U = np.hstack((U, u_new))
        # U_DSM_INP = np.hstack((U_DSM_INP, u_dsm_inp ))
        # E = np.hstack((E, e))         

    C = q_u_new_code
    return C
    # return Y, U, E

def learningMatrices(len_X, im):

    """  Q-filter and Learning matrix generated using results from:
    D. A. Bristow, M. Tharayil and A. G. Alleyne, "A survey of iterative learning control," 
    in IEEE Control Systems, vol. 26, no. 3, pp. 96-114, June 2006
    """ 


    """ INPUTS:
    len_X   - Length of reference signal. Q,L,G matrix dimension should match (len_X x len_X)
    im      - filter's impulse response
    """

    """ OUTPUTS:
    Q       - Q-filtering matrix
    L       - Learning matrix
    G       - Plant output matrix
    """

    # len_X = len(X)      # Length of test signal
    h = im[0]     # Impulse response 

    # Tuning matrices
    We = np.identity(len_X)
    Wf = np.identity(len_X)*1e-4
    Wdf = np.identity(len_X)*1e-1

    RowVec = np.zeros((1, len_X))
    ColumnVec =  h[0:len_X]
    ColumnVec = np.reshape(ColumnVec, (len(ColumnVec),1))

    # Output Matrix
    G = linalg.toeplitz(ColumnVec, RowVec)

    # Q-filter and Learning Matrices
    Subinverse11 =  G.transpose() @ We @ G 
    SubinverseQ = Subinverse11 + Wf + Wdf
    SubinverseL =  Subinverse11 + Wdf
    SubinverseQ = np.linalg.inv(SubinverseQ)
    SubinverseL = np.linalg.inv(SubinverseL)

    # Q-filter matrix   
    Q = SubinverseQ @ (G.transpose()@ We @ G + Wdf) 

    # Learning matrix 
    L = SubinverseL @ (G.transpose()@We)

    # Check if the stability and convergence condition are satisfied
    if False:
        ILCloop =  Q - np.matmul(L,G)
        eig_ILC, vec_ILC = np.linalg.eig(ILCloop)       
        eig_ILC_mon, vec_ILC_mon = np.linalg.eig(ILCloop @ ILCloop.transpose())

        if max(eig_ILC) <=1:
            print('Stablity Condition Satisfied')
            if max(eig_ILC_mon) <=1:
                print('ILC Monotonic Convergent Condition also Satisfied')
        else:
            sys.exit('Stability condition not satisfied. Change tuning matrices')

    return Q, L, G


def direct_quantization(Xcs, Qstep, Q_levels, Qtype):
    # Direct quatinzer
    """ INPUTS:
    Xcs         - Reference signal
    Qstep       - Qantizer step size
    Q_levels    - Quantizer levels
    Qtype       - Quantizer type; midread or midrise
    """

    """ OUTPUTS:
    q_Xcs       - Quantized signal with Qstep, step size
    """
    # Range of the quantizer
    Vmax = np.max(Q_levels)
    Vmin = np.min(Q_levels)
    # Select quantizer type
    match Qtype:
        case 1:
           q_Xcs = np.floor(Xcs/Qstep +1/2)*Qstep
        case 2:
            q_Xcs = np.floor(Xcs/Qstep )*Qstep +1/2
    # Quatizer saturation within its range
    # for i in range(0,len(q_Xcs)):
    #     if q_Xcs[i] <= Vmin:
    #         q_Xcs[i] = Vmin
    #     elif q_Xcs[i] >= Vmax:
    #         q_Xcs[i] = Vmax
    np.place(q_Xcs, q_Xcs> Vmax, Vmax)
    np.place(q_Xcs, q_Xcs < Vmin, Vmin)

    return q_Xcs
    


def generate_code(q_Xcs, Qstep, Vmin, Qtype):
    # Converter the quantized values to unsigned integers

    """ INPUTS:
    q_Xcs       - Quantized signal
    Qstep       - Qantizer step size
    Vmin        - Quantizer lower range
    Qtype       - Quantizer type; midread or midrise
    """

    """ OUTPUTS:
    q_code      - code corresponsing to the quantized values and quantizer levels
    """
    match Qtype:
        case 1:
            q_code = q_Xcs/Qstep - np.floor(Vmin/Qstep)
        case 2:
            q_code = q_Xcs/Qstep - np.floor(Vmin/Qstep) - 1/2
    return q_code.astype(int)

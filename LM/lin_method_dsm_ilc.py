#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" DSM-ILC IMPLEMENTATION

@author: Bikash Adhikari
@date: 22.02.2024
@license: BSD 3-Clause
"""

import numpy as np
from scipy import signal, linalg
import math
from utils.balreal import balreal
import tqdm

class DSM_ILC:

    def __init__(self, Nb, Qstep, Vmin, Vmax,  Qtype, Qmodel):
        """
        Constructor for the Delta-sigma modulation with iterative learning control.
        :param Nb: Number of bits 
        :param Qstep: Quantizer step size / Least significant bit (LSB) 
        :param X: Reference/Test signal 
        :param QL: Quantization levels 
        :param b, a: Numerator and denominator;  Reconstruction filter
        :param imp_res: reconstruction filter impulse response
        """
        self.Nb = Nb
        self.Qstep = abs(Qstep)
        self.Vmin = Vmin 
        self.Vmax = Vmax
        self.Qtype = Qtype
        self.Qmodel = Qmodel


    # LEARNING MATRICES
    """  Q-filter and Learning matrix generated using results from:
    D. A. Bristow, M. Tharayil and A. G. Alleyne, "A survey of iterative learning control," 
    in IEEE Control Systems, vol. 26, no. 3, pp. 96-114, June 2006
    """ 
    def learningMatrices(self, len_X,  We, Wf, Wdf, im):
        """ inputs:
        len_x   - length of reference signal. q,l,g matrix dimension should match (len_x x len_x)
        im      - filter's impulse response
        we, wf, wdf - tuning matrices
        """

        """ outputs:
        q       - q-filtering matrix
        l       - learning matrix
        g       - plant output matrix
        """

        # Impulse response
        h = im[0]      

        # Tuning matrices
        # We = np.identity(len_X)
        # Wf = np.identity(len_X)*1e-4
        # Wdf = np.identity(len_X)*1e-1

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
    
 
    def get_codes(self, Xcs, Dq, itr, YQns, MLns,  Q, L, G):

        """ INPUTS:
        Xcs     - Reference/Test signal 
        Dq      - Dither signal
        itr     - Number of iterations
        YQns    - Ideal quantisation levels
        MLns    - Measured/ calibrated quantisation levels
        Q, L    - Q-filtering and Learning matrices
        G       - Output matrix; Reconstruction filter
        b1, a1  - Transfer functions; Reconstruction filter; numerator and denominator, respectively
        """

        """ OUTPUTS:
        Codes
        """

        # Storage container: Codes
        ILC_C = []

        # Initial input signal
        init_u = np.zeros_like(Xcs)

        # Reference signal update
        u_dsm_inp = Xcs + init_u

        # Output 
        y = G @ init_u.reshape(-1,1)

        # Error 
        ilc_err = Xcs - y.squeeze()

        # ILC loop
        for i in tqdm.tqdm(range(itr)):
            #NSD:
            ##########################
            match 2:
                case 1:
                    bns = np.array([1, -2, 1])
                    ans =  np.array([1, 0, 0])
                    Mns_tf = signal.TransferFunction( ans-bns, ans, dt=1)  # Mns = 1 - Hns
                    Mns = Mns_tf.to_ss()
                    Ad, Bd, Cd, Dd = balreal(Mns.A, Mns.B, Mns.C, Mns.D)

                case 2:
                    AM = np.array([[0.0, 0.0], [1.0, 0.0]])
                    BM = np.array([[2.0], [0.0]])
                    CM = np.array([[1.0, -0.5]])
                    DM = np.array([[0.0]])
                    Ad, Bd, Cd, Dd = balreal(AM, BM, CM, DM)

            # Initialise state, output and error
            xns = np.zeros((Ad.shape[0], 1))  # noise-shaping filter state
            yns = np.zeros((1, 1))  # noise-shaping filter output
            e = np.zeros((1, 1))  # quantiser error
            NSQ_C = np.zeros((1, Xcs.size)).astype(int)

            # Noise shaping loop
            for j in range(0,u_dsm_inp.size):
                w = u_dsm_inp[j] - yns[0,0]
                u_dsm = w + Dq[j]

                # Requantize
                q_u_dsm = math.floor(u_dsm/self.Qstep + 0.5)
                q_u_dsm_code = q_u_dsm + 2**(self.Nb-1)

                # Set saturation limits
                if q_u_dsm_code < 0: q_u_dsm_code = 0
                if q_u_dsm_code > 2**self.Nb-1: q_u_dsm_code = 2**self.Nb -1

                # Store values
                NSQ_C[0,j] = q_u_dsm_code

                # Generate DAC output
                q_dsm_i = YQns[q_u_dsm_code]
                q_dsm_m = MLns[q_u_dsm_code]

                # Generate error
                match self.Qmodel:
                    case 1:
                        e[0] = q_dsm_i - w
                    case 2:
                        e[0] = q_dsm_m - w
                
                # noise shaping filter 
                xns = Ad@xns + Bd@e
                yns = Cd@xns
            #################################

            # DAC output
            match self.Qmodel:
                case 1:
                    u_dsm_out = self.generate_dac_output(NSQ_C, YQns.reshape(1,-1))
                case 2:
                    u_dsm_out = self.generate_dac_output(NSQ_C, MLns.reshape(1,-1))

            # Filter output
            y = G @ u_dsm_out.reshape(-1,1)
            
            # ILC error 
            ilc_err = Xcs.reshape(-1,1) - y.reshape(-1,1)

            # Calculate new feed forward control signal 
            u_new = Q@(init_u.reshape(-1,1) + L@ilc_err.reshape(-1,1))
            init_u = u_new.squeeze()

            # update dsm input; add generated feed forward signal to the reference/test signal
            u_dsm_inp = init_u + Xcs

            # Codes 
            ILC_C = NSQ_C

        return ILC_C


    # Generate DAC output
    def generate_dac_output(self,C, ML):
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


    # def nsq(self, X, Dq, YQns, MLns,b,a):
    #     """
    #     X
    #         input signal
    #     Dq
    #         re-quantiser dither
    #     YQns, 1d array
    #         ideal, uniform output levels (ideal model)
    #     MLns, 1d array
    #         measured, non-unform levels (calibration model)
    #     Qstep, Vmin, Nb
    #         quantiser params. (for re-quantisation and code generation)
    #     QMODEL
    #         choice of quantiser model
    #             1: ideal
    #             2: measured/calibrated
    #     """
    #     # Noise-shaping filter (using a simple double integrator)
    #     # b = np.array([1, -2, 1])
    #     # a =  np.array([1, 0, 0])
    #     # Hns_tf = signal.TransferFunction(b, a, dt=1)  # double integrator
    #     Mns_tf = signal.TransferFunction( a-b, a, dt=1)  # Mns = 1 - Hns
    #     Mns = Mns_tf.to_ss()
    #     # Make a balanced realisation.
    #     # Less sensitivity to filter coefficients in the IIR implementation.
    #     # (Useful if having to used fixed-point implementation and/or if the filter order is to be high.)
    #     Ad, Bd, Cd, Dd = balreal(Mns.A, Mns.B, Mns.C, Mns.D)
    #     # Initialise state, output and error
    #     xns = np.zeros((Ad.shape[0], 1))  # noise-shaping filter state
    #     yns = np.zeros((1, 1))  # noise-shaping filter output
    #     e = np.zeros((1, 1))  # quantiser error

    #     C = np.zeros((1, X.size)).astype(int)  # save codes

    #     satcnt = 0  # saturation counter (generate warning if saturating)
        
    #     FB_ON = True  # turn on/off feedback, for testing
        
    #     for i in range(0, X.size):
    #         x = X[i]  # noise-shaper input
    #         d = Dq[i]  # re-quantisation dither
            
    #         if FB_ON: w = x - yns[0, 0]  # use feedback
    #         else: w = x
            
    #         u = w + d  # re-quantizer input
            
    #         # Re-quantizer (mid-tread)
    #         q = math.floor(u/self.Qstep + 0.5)  # quantize
    #         c = q - math.floor(self.Vmin/self.Qstep)  # code
    #         C[0, i] = c  # save code

    #         # Saturation (can't index out of bounds)
    #         if c > 2**self.Nb - 1:
    #             c = 2**self.Nb - 1
    #             satcnt = satcnt + 1
    #             if satcnt >= 10:
    #                 print(f'warning: pos. sat. -- cnt: {satcnt}')
                
    #         if c < 0:
    #             c = 0
    #             satcnt = satcnt + 1
    #             if satcnt >= 10:
    #                 print(f'warning: neg. sat. -- cnt: {satcnt}')
            
    #         # Output models
    #         yi = YQns[c]  # ideal levels
    #         ym = MLns[c]  # measured levels
            
    #         # Generate error
    #         match self.Qmodel:  # model used in feedback
    #             case 1:  # ideal
    #                 e[0] = yi - w
    #             case 2:  # measured/calibrated
    #                 e[0] = ym - w
            
    #         # Noise-shaping filter
    #         xns = Ad@xns + Bd@e  # update state
    #         yns = Cd@xns  # update filter output

    #     return C

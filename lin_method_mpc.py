
import numpy as np
from scipy import linalg , signal
import sys
import random
import gurobipy as gp
from gurobipy import GRB
import tqdm


class MPC:
    def __init__(self, Nb, Qstep, N_PRED, Xcs, QL, A, B, C, D):
        """
        Constructor for the Model Predictive Controller.
        :param Nb: Number of bits 
        :param Qstep: Quantizer step size / Least significant bit (LSB) 
        :param N_PRED: Prediction horizon | int 
        :param Xcs: Reference/Test signal 
        :param QL: Quantization levels 
        :param A, B, C, D: Matrices; state space representation of the reconstruction filter
        """
        self.Nb = Nb
        self.Qstep = abs(Qstep)
        self.N_PRED = N_PRED
        self.Xcs = Xcs
        self.QL = QL.reshape(1,-1)
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        # self.x0 = x0
        
    def state_prediction(self, st, con):
        """
        Predict the state for the given initial condition and control
        """
        x_iplus1 = self.A @ st + self.B * con
        return x_iplus1


    def get_codes(self):

        # Scale the input to the quantizer levels to run it as an MILP
        Xs = self.Xcs.squeeze()
        X = Xs/self.Qstep + 2**(self.Nb-1)

        #  Scale ideal levels  
        QLS = (self.QL /self.Qstep ) + 2**(self.Nb-1)
        QLS = QLS[0].squeeze()

        # Storage container for code
        C = []

        # Loop length
        len_MPC = X.size - self.N_PRED

        # State dimension
        x_dim =  int(self.A.shape[0]) 

        # Initial state
        init_state = np.zeros(x_dim).reshape(-1,1)

        # MPC loop
        for j in tqdm.tqdm(range(len_MPC)):

            m = gp.Model("MPC- INL")
            u = m.addMVar(self.N_PRED, vtype=GRB.INTEGER, name= "u", lb = 0, ub =  2**self.Nb-1) # control variable
            x = m.addMVar((x_dim*(self.N_PRED+1),1), vtype= GRB.CONTINUOUS, lb = -GRB.INFINITY, ub = GRB.INFINITY, name = "x")  # State varible 


            # Add objective function
            Obj = 0

            # Set initial constraint
            m.addConstr(x[0:x_dim,:] == init_state)
            for i in range(self.N_PRED):
                k = x_dim * i
                st = x[k:k+x_dim]
                con = u[i] - X[j+i]

                # Objective update
                e_t = self.C @ x[k:k+x_dim] + self.D * con
                Obj = Obj + e_t * e_t

                # Constraints update
                f_value = self.A @ st + self.B * con
                st_next = x[k+x_dim:k+2*x_dim]
                m.addConstr(st_next == f_value)

            # Gurobi model update
            m.update

            # Set Gurobi objective
            m.setObjective(Obj, GRB.MINIMIZE)

            # 0 - Supress log output, 1- Print log outputs
            m.Params.LogToConsole = 0

            # Gurobi setting for precision  
            m.Params.IntFeasTol = 1e-9
            m.Params.IntegralityFocus = 1

            # Optimization 
            m.optimize()

            # Extract variable values 
            allvars = m.getVars()
            values = m.getAttr("X",allvars)
            values = np.array(values)

            # Extract only the value of the variable "u", value of the variable "x" are not needed
            C_MPC = values[0:self.N_PRED]

            # Ideally they should be integral, but gurobi generally return them in floating point values according to the precision tolorence set: m.Params.IntFeasTol
            # Round off to nearest integers
            C_MPC = C_MPC.astype(int)

            # Store only the first value /code
            C.append(C_MPC[0])

            # Get DAC level according to the coe
            U_opt = QLS[C_MPC[0]] 

            # State prediction 
            con = U_opt - X[j]
            x0_new = self.state_prediction(init_state, con)

            # State update for subsequent prediction horizon 
            init_state = x0_new

        return np.array(C).reshape(1,-1)



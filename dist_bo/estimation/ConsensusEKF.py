import numpy as np
class ConsensusEKF:
    """
        The distributed Extended Kalman Filter with consensus on qhat enabled.
        
        The update formula is modified from equation (20) in
        
            Olfati-Saber, R. (2009). Kalman-Consensus Filter: Optimality, stability, and performance. Proceedings of the 48h IEEE Conference on Decision and Control (CDC) Held Jointly with 2009 28th Chinese Control Conference, 7036–7042. https://doi.org/10.1109/CDC.2009.5399678
        
        The main idea is to perform consensus on z(mean estimation) in the correction step, but do not consensus on covariant matrices to reduce the computation/communication overhead.
        
        We only communicate z after prediction step, so the consensus term is based on z predicted instead of z corrected, which is a subtle difference from Olfati-Saber's paper.
        
        The estimator assumes a constant-velocity source movement model, with the velocity to be the fundamental uncertainty.
    """
    def __init__(self,q_0,R_mag=1,Q_mag=1,C_gain = 0.0,x_min=-np.inf,x_max=np.inf,y_min = -np.inf,y_max=np.inf):
        '''
        q_0 should be a vector, the initial guess of source position.
        
        R_mag,Q_mag,C_gain should be scalars.
        '''
        
        self.clipping = {'mins':[x_min,y_min],'maxes':[x_max,y_max]}
        
        self.q0 = q_0 # Mean
        self.q0_dot = np.zeros(self.q0.shape) # Velocity Mean
        
        self.z = np.hstack([self.q0,self.q0_dot]) # The source predicted state, public variable.
        self._zbar = np.hstack([self.q0,self.q0_dot]) # The source corrected state, private variable.
        
        
        
        self.qdim = len(self.q0)
        
        self.P = np.eye(len(self.z)) # Covariance of [q,qdot]. Initialized to be the identity matrix
        
        self.R_mag = R_mag 
        self.Q_mag = Q_mag
        self.C_gain = C_gain # The consensus gain. See the update function.
    
    def reset(self):
        self.z = np.hstack([self.q0,self.q0_dot])
        self._zbar = np.hstack([self.q0,self.q0_dot]) 
        self.P = np.eye(len(self.z)) # Covariance of [q,qdot]. Initialized to be the identity matrix
      
    def dfdz(self,z):
        n = len(z)//2
        O =np.zeros((n,n))
        I=np.eye(n)
        return np.vstack([np.hstack([I,I]),np.hstack([O,I])])

    def f(self,z):
        """
            The constant velocity model.
        """
        A = self.dfdz(z)
        
        return A.dot(z)

    def get_z(self):
        return self.z
        
    def get_q(self):
        return self.z[:len(self.q0)]

    def update(self,h,dhdz,y,p,z_neighbor,z_neighbor_bar=None,consensus_weights=None):
        """
        h is a function handle h(z,p), the measurement function that maps z,p to y.
        dhdz(z,p) is its derivative function handle.
        
        y is the actual measurement value, subject to measurement noise.
        
        p is the positions of the robots.
        
        z_neighbor is the list of z estimations collected from the neighbors(including self).
        """
        y = np.array(y).flatten()
        p = np.array(p).reshape(-1,2)
        z_neighbor = np.array(z_neighbor).reshape(-1,4)

        A = self.dfdz(self._zbar)
        C = dhdz(self.z,p)
        R = self.R_mag*np.eye(len(y))
        Q = self.Q_mag*np.eye(len(self.z))
        
        # The Kalman Gain
        K = A.dot(self.P).dot(C.T).dot(    np.linalg.inv(C.dot(self.P).dot(C.T)+R)      )

        # Mean and covariance update
        N = len(z_neighbor)

        if consensus_weights is None:
        
#         Consensus variation (5), consensus on z_hat.
#         self._zbar= self.z+K.dot(y-h(self.z,p)) +\
#                     self.C_gain*np.ones((1,N)).dot(z_neighbor-self.z).flatten() 
#         self.z = self.f(self._zbar)


#         Consensus variation (6), consensus on z_bar
#         self._zbar= self._zbar+K.dot(y-h(self.z,p)) +\
#                     self.C_gain*np.ones((1,N)).dot(z_neighbor_bar-self._zbar).flatten()
#         self.z = self.f(self._zbar)
        
        
        
#         The Stable version of consensus scheme
            
            new_z = np.array(self.z)
            
            # new_z = self.f(new_z) + \
            #          K.dot(y-h(new_z,p)) +\
            #     (self.C_gain*np.ones((1,N)).dot(z_neighbor-new_z)).flatten() # The consensus term.
            new_z = self.f(new_z) + \
                 K.dot(y-h(new_z,p)) 
            
        else:
            # print('Two pass parallel')
            # print(consensus_weights,z_neighbor)

            new_z = consensus_weights.dot(z_neighbor)/np.sum(consensus_weights)  # The consensus term.
            new_z =  self.f(new_z)+K.dot(y-h(new_z,p))    
            
        if np.any(np.isnan(new_z)):
            pass
        else:             

            new_z[:2] = np.clip(new_z[:2],self.clipping['mins'],self.clipping['maxes']) # The clipping prevents the EKF estimate to go crazy.

            self.z = new_z          
            self.P = A.dot(self.P).dot(A.T)+ Q- K.dot(C.dot(self.P).dot(C.T)+R).dot(K.T)
        
    def update_and_estimate_loc(self,h,dhdz,y,p,z_neighbor,z_neighbor_bar=None,consensus_weights=None):
        if not np.any(y == np.inf):
            self.update(h,dhdz,y,p,z_neighbor,z_neighbor_bar,consensus_weights)

        return self.get_q()

    
        
        
import numpy as np
from sklearn.linear_model import LinearRegression

def single_meas_func(C1,C0,k,b,dist):
    eps = 1e-6 

    # b is typically negative. We need to ensure raising a positive value to power b.
    diff = dist-C1
    diff[diff<=0]=eps

    return k*diff**b+C0

def joint_meas_func(C1s,C0s,ks,bs,x,ps):

    # Casting for the compatibility of jax.numpy

    C1=np.array(C1s)
    C0=np.array(C0s)
    k=np.array(ks)
    b=np.array(bs)
    p=np.array(ps)

    # Keep in mind that x is a vector of [q,q'], thus only the first half of components are observable.    
    dists=np.linalg.norm(x[:len(x)//2]-p,axis=-1)

    return single_meas_func(C1,C0,k,b,dists) 

def dhdr(r,C1s,C0s,ks,bs):
    eps = 1e-6  
    # b is typically negative. We need to ensure raising a positive value to power b.
    diff = r-C1s
    diff[diff<=0]=eps

    return ks*bs*(diff)**(bs-1)

def d2hdr2(r,C1s,C0s,ks,bs):
    eps = 1e-6 
    # b is typically negative. We need to ensure raising a positive value to power b.
    diff = r-C1s
    diff[diff==0]=eps

    return dhdr(r,C1s,C0s,ks,bs)*(bs-1)/diff

def analytic_dhdz(x,ps,C1s,C0s,ks,bs):

    q = x.flatten()
    q = q[:len(q)//2]
    dhdq = analytic_dhdq(q,ps,C1s,C0s,ks,bs)
    # return jnp.hstack([dhdq,np.zeros(dhdq.shape)])
    return np.hstack([dhdq,np.zeros(dhdq.shape)])

def analytic_dhdq(q,ps,C1s,C0s,ks,bs):
    # rs = jnp.linalg.norm(ps-q,axis=1)
    rs = np.linalg.norm(ps-q,axis=-1)
   
    r_hat = ((ps-q).T/rs).T
    d = dhdr(rs,C1s,C0s,ks,bs)
    dhdq=-(d * r_hat.T).T
    return dhdq

def analytic_FIM(q,ps,C1s,C0s,ks,bs):
    # rs = np.linalg.norm(ps-q,axis=1)
    rs = np.linalg.norm(ps-q,axis=-1)
    r_hat = ((ps-q).T/rs).T


    d = dhdr(rs,C1s,C0s,ks,bs)
    dd = d2hdr2(rs,C1s,C0s,ks,bs)       

    As = (-d*r_hat.T).T
   

    return As.T.dot(As) # Current FIM

def F_single(dh,qhat,ps):
    A = dh(qhat,ps)
    return A.T.dot(A)

def joint_F_single(qhat,ps,C1,C0,k,b): # Verified to be correct.
    # The vectorized version of F_single.
    # The output shape is (N_sensor, q_dim, q_dim).
    # Where output[i]=F_single(dh,qhat,ps[i])
    A = analytic_dhdq(qhat,ps,C1s=C1,C0s=C0,ks=k,bs=b)
    return A[:,np.newaxis,:]*A[:,:,np.newaxis]


def analytic_dLdp(q,ps,C1s,C0s,ks,bs,FIM=None):
    """
        The gradient is taken with respect to all the ps passed in. 

        The FIM is by default calculated internally, but if it is passed in, will
        use the passed in FIM for the calculation of Q below.
    """
  
    rs = np.linalg.norm(ps-q,axis=-1)
    r_hat = ((ps-q).T/rs).T
    t_hat=np.zeros(r_hat.shape)
    t_hat[:,0]=-r_hat[:,1]
    t_hat[:,1]=r_hat[:,0]

    d = dhdr(rs,C1s,C0s,ks,bs)
    dd = d2hdr2(rs,C1s,C0s,ks,bs)


    if FIM is None:
        wrhat=(d*r_hat.T).T
        Q = np.linalg.inv(wrhat.T.dot(wrhat)) # Default calculation of FIM^-1
    else:
        # print('Coordinating')
        if np.linalg.matrix_rank(FIM) < 2:
            FIM = FIM + 1e-9*np.eye(2)
        Q = np.linalg.inv(FIM) # Using the passed in FIM.

    c1 = -2*d*dd*np.linalg.norm(Q.dot(r_hat.T),axis=0)**2
    c2 = -2*(1/rs)*(d**2)*np.einsum('ij,ij->j',Q.dot(r_hat.T),Q.dot(t_hat.T))

    return (c1*r_hat.T+c2*t_hat.T).T

def local_dLdp(q,p,p_neighborhood,C1s,C0s,ks,bs):
    
    local_FIM = analytic_FIM(q,p_neighborhood,C1s,C0s,ks,bs)

    return analytic_dLdp(q,p,C1s,C0s,ks,bs,FIM = local_FIM)



def top_n_mean(readings,n):
    """
        top_n_mean is used to convert the reading vector of the 8 light-sensors installed 
        on Turtlebots into a single scalar value, representing the overall influence of 
        the light source to the Turtlebots.
    """
    if len(readings.shape)==1:
        rowwise_sort = np.sort(readings)
        return np.mean(rowwise_sort[-n:])

    rowwise_sort=np.sort(readings,axis=-1)
    return np.mean(rowwise_sort[:,-n:],axis=-1)

## The once and for all parameter calibration function.
def calibrate_meas_coef(robot_loc,target_loc,light_readings,fit_type='light_readings',loss_type='rmse'):

    def loss(C_1,dists,light_strengths,C_0=0,fit_type='light_readings',loss_type='rmse'):
        '''
            h(r)=k(r-C_1)**b+C_0
        '''
    
        x=np.log(dists-C_1).reshape(-1,1)
        y=np.log(light_strengths-C_0).reshape(-1,1)

        model=LinearRegression().fit(x,y)

        k=np.exp(model.intercept_[0])
        b=model.coef_[0][0]


        # print('fit_type:',fit_type)
        if fit_type=="light_readings":
            ## h(r)=k(r-C_1)**b+C_0
            yhat=k*(dists-C_1)**b+C_0
            
            if loss_type=='max':
                e=np.sqrt(np.max((yhat-light_strengths)**2))
            else:
                e=np.sqrt(np.mean((yhat-light_strengths)**2))
        elif fit_type=='dists':
            rh=rhat(light_strengths,C_1,C_0,k,b)
            if loss_type=='max':
                e=np.sqrt(np.max((rh-dists)**2))
            else:
                e=np.sqrt(np.mean((rh-dists)**2))
        
        return e,C_1,C_0,k,b

    dists=np.sqrt(np.sum((robot_loc-target_loc)**2,axis=-1))
    light_strengths=top_n_mean(light_readings,4)
    
    ls=[]
    ks=[]
    bs=[]
    C_1s= np.linspace(-10,np.min(dists)-0.01,100)
    C_0s=np.linspace(-10,np.min(light_strengths)-0.01,100)
    ls=[]
    mls=[]
    for C_1 in C_1s:
        for C_0 in C_0s:

            l,C_1,C_0,k,b=loss(C_1,dists,light_strengths,C_0=C_0,fit_type=fit_type,loss_type=loss_type)
            ls.append(l)
            ks.append(k)
            bs.append(b)
    
    ls=np.array(ls).reshape(len(C_1s),len(C_0s))

    best_indx=np.argmin(ls)
    best_l=np.min(ls)
    x,y=np.unravel_index(best_indx,ls.shape)
    
    return C_1s[x],C_0s[y],ks[best_indx],bs[best_indx]

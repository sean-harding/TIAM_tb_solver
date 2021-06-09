import numpy as np
import scipy as sp
import scipy.sparse
import matplotlib
import scipy.sparse.linalg
from math import pi,sin,cos,tan,sqrt,sinh
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as py
import time as t
import scipy.optimize
import scipy.fftpack as fourier
import time

'-------------------------------|Functions for spectral computation|---------------------------------'
def moments(H,GS):
    #Add do-orthogonalization checks
    '''Compute Chebyshev moments'''
    yield (GS.transpose()*GS).todense()[0,0]
    T1,T0 = H*GS,GS
    yield (GS.transpose()*T1).todense()[0,0]
    while True:
        T2 = 2*H*T1-T0
        T0,T1 = T1,T2
        yield (GS.transpose()*T2).todense()[0,0]

def reCheb(H,GS,N):
    o = float((GS.transpose()*H*GS).todense())
    m = float((GS.transpose()*GS).todense())
    chebVectors =[GS/m]
    new = H*GS - o*GS/m
    chebVectors.append(new/sp.sparse.linalg.norm(new))
    for k in range(2,N):
        T_next = 2*H*chebVectors[-1] - chebVectors[-2]      #Calculate Chebyshev states using un-orthogonalized
        #chebVectors.append(T_next.copy())
        #size = sp.sparse.linalg.norm(T_next)
        for t in range(2):
            overlaps = [float((vector.transpose()*T_next).todense()) for vector in chebVectors]
            for o,vec in zip(overlaps,chebVectors):
                T_next -= o*vec/float((vec.transpose()*vec).todense())
        T_next = T_next/sp.sparse.linalg.norm(T_next)
        chebVectors.append(T_next)
        #orth.append(T_next*size/sp.sparse.linalg.norm(T_next))  #After orthogonalization, make sure the magnitude is the same?
    return chebVectors

def cheb(freq):
    '''Computes Chebyshev polynomials. The first one comes with a factor
    of 0.5 to simplify the code for computing the spectral function''' 
    t0,t1 = sp.ones(len(freq)),freq
    yield t0*0.5
    yield t1
    while True:
        t2 = 2*freq*t1-t0
        yield t2
        t0,t1 = t1,t2

def build(O,orth):
    v0 = orth[0]
    ham = [float((v.transpose()*O*v0).todense()) for v in orth[0:3]] + [0]*(len(orth)-3)
    for i,vec1 in enumerate(orth[1:]):  #i is the index 1 less than the position of vec1 in the list
        mElement = [float((v.transpose()*O*vec1).todense()) for v in orth[i:i+3]]
        mElement = [0]*i + mElement +[0]*(len(orth)-len(mElement)-i)
        ham = sp.sparse.vstack([ham,sp.sparse.csr_matrix(mElement)])
    return ham

def linearPrediction(moments,first,how_many):
    '''Predicts how_many more chebyshev moments by a linear iterative scheme, from a test_set'''
    first = int(first)
    L = len(moments[first:])    #Train on the set first:end of size L
    #p = min(int(L/2),25)        #Use a window of size p: i.e. for each moment first:L, predict that moment using the previous p moments
    p=35
    R = sp.zeros([p,p],dtype='float64')
    X=sp.zeros(p,dtype='float64')
    print('Window size:{}\nTraining set size:{}\nFirst moment:{}'.format(p,L,first))
    #Set up the relevant matrices which are required
    for i in range(p):
        for j in range(p):
            n=0
            while n<L:
                R[i,j] = R[i,j]+ moments[first+n-j-1]*sp.conj(moments[first+n-i-1])
                n+=1
    for i in range(p):
        n=0
        while n<L:
            X[i] = X[i] + moments[first+n]*sp.conj(moments[first+n-i-1])
            n+=1
    R = -1*sp.matrix(R) + sp.eye(len(R))*(10**-9)
    X = -1*sp.matrix(X).transpose()
    A = -1.0*sp.linalg.pinv(R)*X
    M = sp.eye(len(A)-1)
    M = sp.hstack([M,sp.zeros([len(M),1])])
    M = sp.vstack([-1*A.transpose(),M])
    M = sp.matrix(M)
    moving_set = sp.copy(moments[-p:len(moments)])
    moving_set = sp.flip(moving_set,axis=0)
    moving_set = sp.matrix(moving_set).transpose()
    weight = sp.linalg.norm(M*moving_set)
    eVals,V_l,V_r = sp.linalg.eig(M,left=True,right=True)
    V_l = sp.matrix(V_l).getT()
    V_l = sp.matrix(V_l)
    M = V_l*M*V_l.getI()
    diagonals = sp.diag(M).copy()
    divergent = sp.where(np.abs(diagonals)>=1)[0]
    print('Number of divergent eigenvalues is {}'.format(len(divergent)))
    for d in divergent:
        #M[d,d]=0.0+0*1j                #Various protocols to deal with divergent eigenvectors
        M[d,d]=0.99*M[d,d]/abs(M[d,d])
        #M[d,d]=M[d,d]/abs(M[d,d])
    M = V_l.getI()*M*V_l
    new_weight = sp.linalg.norm(M*moving_set)
    print('Discarded weight = %.3f percent'%float(abs((weight-new_weight)/weight)*100))
    new_moments = []
    k=0
    while k<how_many:                       #A robust modification would be to turn this into a generator to compute moments
        moving_set = M*moving_set           #on the fly, or to just iterate until the moments decay to zero
        new_moments.append(moving_set[0,0])
        k = k+1
    return sp.real(new_moments),abs((weight-new_weight)/weight)*100

'-------------------------------|LOAD IN RESULTS FROM MAIN.PY|---------------------------------'
print('Loading matrices...')
operators = sp.load("ops_tdens=1.0_U=3.00_c=0.00_J=0.10.npz")
H_plus = operators['H_plus']
H_minus = operators['H_minus']
a_up = operators['a_up']
c_up = operators['c_up']
GS = operators ['GS']
GS = sp.sparse.csr_matrix(GS)
E = operators['GSE']
E = float(E)

H_plus = sp.reshape(H_plus,(1,))[0]
H_minus = sp.reshape(H_minus,(1,))[0]
a_up = sp.reshape(a_up,(1,))[0]
c_up = sp.reshape(c_up,(1,))[0]
Delta = operators['delta']           #I have used Delta = q*Delta for the optimization!
'-------------------------------|Rescale|---------------------------------'
a = 10    #Rescaling factor    
b = 0*a    #Shift

nMoments = 250  #Ensure this is even as we only predict the even moments        
nKryl = 50                                 
nTot =2000
print('Rescaling matrices')
dPlus = sp.shape(H_plus)[0]
dMinus = sp.shape(H_minus)[0]
Hp_r = (H_plus - sp.sparse.eye(dPlus)*(E+b))/a
Hl_r = (H_minus - sp.sparse.eye(dMinus)*(E+b))/a  
'Frequency mesh in units of half-bandwidth,rescaled'
frequencies =  sp.linspace(-1.5,1.5,10000)
frequencies_rescaled = frequencies/a
'-------------------------------|Chebyshev - initial expansion|---------------------------------'
print('Calculating Chebyshev moments...')
#Initial Chebyshev moments with no fancy pre-processing
t1 = time.perf_counter()
gens = [moments(H,v) for H,v in zip((Hp_r,Hl_r),(c_up*GS,a_up*GS))]
mom_init = [[next(geny) for k in range(nMoments)] for geny in gens]        #Convert to genexp when we don't want to store these         
minit = [mom_init[0][n] + mom_init[1][n]*(-1)**n for n in range(nMoments)] #as this code is still being tested, we do want to plot
t2 = time.perf_counter()                                                   #the data obtained in intermediate steps
print("Elapsed: %.4f" % (t2-t1))

print('Performing Krylov subspace method')
#Construct a re-orthogonalized Krylov subspace in order to achieve a more stable Chebyshev iteration
kryl = (reCheb(M,v,nKryl) for M,v in zip((Hp_r,Hl_r),(c_up*GS,a_up*GS)))
H_kryl = (build(M,k) for M,k in zip((Hp_r,Hl_r),kryl))
#Re-do the expansion but in the Krylov basis
cheb_init = [sp.sparse.csr_matrix([sqrt(m[0])]+[0]*(nKryl-1)) for m in mom_init]    #Cheby_0 vectors in Krylov basis
regens = [moments(H,v.transpose()) for H,v in zip(H_kryl,cheb_init)]                #New generators
mom_recheb = [[next(geny) for k in range(nMoments)] for geny in regens]
mrecheb = [mom_recheb[0][n] + mom_recheb[1][n]*(-1)**n for n in range(nMoments)]
t3 = time.perf_counter()
print('Elapsed: %.4f' % (t3-t2))

'-------------------------------|PLOT INITIAL EXPANSION|---------------------------------'
py.plot(mrecheb,'x')
py.plot(minit,'o')
py.legend(['recalculated','initial'])
py.show()
#quit()
'-------------------------------|LINEARLY PREDICT TO HIGH FREQUENCY|---------------------------------'
print('Linear prediction')

theta = lambda n:pi*n/(nTot+1)
w = 2/(pi*sp.sqrt(1-frequencies_rescaled**2))
K_jackson = [(nTot-n-1)*cos(theta(n))+sin(theta(n))/tan(theta(n)/n) for n in range(1,len(mrecheb))]
K_jackson = [K_jackson[n]/(nTot+1) for n in range(len(K_jackson))]
K_jackson.insert(0,(nTot-1)/(nTot+1))

specFunctions = []
lpmoms = []
for moms in (mrecheb,minit):
    cheby = cheb(frequencies_rescaled)                     #Use generator here as we could do quick linear prediction, check if converged,
    m_init = [m*k for m,k in zip(moms,K_jackson)]          #and if not, restart the process
    moments_lp,tweight = linearPrediction(m_init,int(0.75*len(m_init)),nTot-len(m_init))
    A = sum(mu*w*next(cheby) for mu in m_init)/a          #Can save this to compare linear prediction with unpredicted spectra
    A+=sum(mu*w*next(cheby) for mu in moments_lp)/a 
    specFunctions.append(A)
    lpmoms.append(moments_lp)
figs,ax = py.subplots(2,1)
for i,moms in enumerate((mrecheb,minit)):
    ax[0].plot(sp.append(moms,lpmoms[i]),'o')
for spec in specFunctions:
    ax[1].plot(frequencies,spec*pi*Delta)
py.legend(['Recheb','Initial'])
py.show()
#sp.savez('spectrum_1.4_U=1.5_J=0.1.npz', frequencies=frequencies,A=A*pi*Delta)
print("Done")
quit()


colours = sp.full_like(sp.zeros([1,len(moments)]),['r'],dtype='str')
colours = sp.append(colours,sp.full_like(sp.zeros([1,len(moments_LP)]),['b'],dtype='str'))
figs,ax = py.subplots(2,1)
ax[0].scatter(sp.linspace(1,len(moments_LP),len(moments+moments_LP)),moments+moments_LP,c=colours)
#ax[0].plot(moments+moments_LP,'ro')
ax[1].plot(frequencies,A*pi*Delta)
ax[1].plot(frequencies,A_LP*pi*Delta)
ax[1].set_xlabel('Frequency/D')
ax[1].set_ylabel('AΔπ')
py.show()

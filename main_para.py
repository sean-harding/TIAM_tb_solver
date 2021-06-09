import scipy as sp
import scipy.linalg,scipy.sparse,scipy.sparse.linalg
import numpy as np
import helperFunctions as f
from operators_parallel import buildHamiltonian,getImpOperators
import time
from schmidtNew import schmidt,neighborTable
from math import pi,sqrt,tan,sinh,cosh,atan
from solver_parallel import solve

#---------|MODEL PARAMETERS|----------
nBath = 39          #Number of bath modes per band: must be odd so there is a zero mode
nSites = 2*nBath+2                                                                                                                                                           
N_up = int(nSites*0.5)                                              
N_dn = int(nSites*0.5)  
#Solver parameters                                                                    
cTol = 10**-50                                                                   

CFS = 0.0                       #Crystal field splitting
couplingStrength = 3*10**-2     #Hybridization strength is proportional to this
D =1                            #2D is the bandwidth

#---------|DISCRETIZE HYBRIDIZATION FUNCTION|----------
bathType = 'flat'
if bathType=='flat':
    hybFunction = lambda x:couplingStrength/(2*D)
elif bathType =='semiellipse':
    hybFunction = lambda x: 2*couplingStrength*sqrt(1-(x/D)**2)/(2*D)

t1 = time.perf_counter()
#Lin/log discretization of hybFunction. Sample points are logarithmically spaced in a small interval
#then linearly spaced up to the band edge
nLog = int(sp.ceil(0.4*nBath/2))
nLin = int(0.5*(nBath-1)-nLog)
Vb,Eb = f.discretize(hybFunction,nLin,nLog,D,lfactor=2.5,b=0.05)

#Eb = [round(E,15) for E in Eb]  #There can be rounding error which breaks particle-hole symmetry at half-filling
#Vb = [round(E,15) for E in Vb]
Delta = sum(k**2 for k in Vb)*pi

U=1.5*pi*Delta                  #Intra-orbital Hubbard interaction strength
J = 0.1*U                       #Exchange term
U2 = U-2*J                      #Inter-orbital Hubbard 

H_imp = sp.zeros([2,2])
H_imp[0,0] = CFS/2
H_imp[1,1] = -1*CFS/2

#---------|BUILD 1-P HAMILTONIAN IN STAR BASIS|----------
E_bath = sp.append(Eb,Eb)
H_bath = sp.diag(E_bath)
H_1p = sp.linalg.block_diag(H_imp,H_bath)
H_1p[0,2:2+len(Eb)] = Vb
H_1p[2:2+len(Eb),0] = Vb
H_1p[1,2+len(Eb):len(H_1p)] = Vb
H_1p[2+len(Eb):len(H_1p),1] = Vb
q=0.5                               #Yields right density. Away from weak interaction, may need to change q
Delta = q*Delta
density = 1
m0= Delta/tan(density*pi/4)
c = 0.5*(m0**2 - Delta**2)/m0           #Removed factor of 2

#Refining the basis for an optimal representation. Here we scan over z with fixed t'
t = lambda th: sqrt(Delta**2+c**2)*sinh(th)
if density >2:
    mu = lambda th: -sqrt(Delta**2+c**2)*cosh(th) + c
elif density <2:
    mu = lambda th: sqrt(Delta**2+c**2)*cosh(th) + c
else:
    mu = lambda th: 0
    t = lambda th: 0 
#chPot = (3*U-5*J)/2
#chPot = 0.1825                 #For d=0.35
#chPot = 0.2125                 #For d=0.4
#chPot = 0.247                  #Gives tdens = 1.8
#chPot = 0.605*(3*U-5*J)/2      #tdens=1.4
chPot = 0.28*(3*U-5*J)/2        #tdens=1.4
#---------|MANY-BODY SOLVER: SCAN Z WITH FIXED t,mu|----------
#Here, scan through z after having fixed t,mu using the above code.
#Can get approximate z from slave-boson code
for z in [0.7]:                 
    #Build reference Hamiltonian, H_f
    H_f = sp.linalg.block_diag(H_imp,H_bath)
    if density == 2:
        H_f[0,0],H_f[1,1] = 0,0                #At half-filling
        H_f[1,0],H_f[0,1] = 0,0
    else:
        H_f[0,2:2+len(Eb)] = [z*v for v in Vb]
        H_f[2:2+len(Eb),0] = [z*v for v in Vb]
        H_f[1,2+len(Eb):len(H_1p)] = [z*v for v in Vb]
        H_f[2+len(Eb):len(H_1p),1] = [z*v for v in Vb]
        H_f[0,0],H_f[1,1] = mu(0)*(z**2),mu(0)*(z**2)
        H_f[1,0],H_f[0,1] = t(0),t(0)
    
    #Construct optimal two chain basis
    eVals,eVecs = sp.linalg.eigh(H_f)
    order = sp.argsort(eVals)
    eVecs=eVecs[:,order]
    dens_matrix = sp.matmul(eVecs[:,0:N_up],eVecs[:,0:N_up].transpose())
    print("Z = {}, Density ={}".format(z,4*dens_matrix[0,0]))  
    U_s = schmidt(H_1p,dens_matrix,2)          
    dens_matrix = U_s.transpose()*dens_matrix*U_s
    H_s = U_s.transpose()*H_1p*U_s
    cfigs = f.initialCfig(H_s,sp.diag(dens_matrix),N_up)
    neighs = neighborTable(H_s)
    imp =[0,1]
    binomials = f.binomial(nSites+1)
    sites = sp.linspace(0,nSites-1,nSites,dtype='uint')
    ePrev=0

    #Many-body solver
    GS,GSE,cfigs,dens = solve(cfigs,binomials,H_s,neighs,N_up,N_dn,imp,ePrev,U,U2,J,chPot,output=True,predict=True,cTol=0.5*10**-6)
    print("E = {},dens = {}, basis size = {}\n".format(GSE,4*dens[0],len(cfigs)))
    sort = sp.flip(sp.argsort([abs(float(a)) for a in GS]))
    amps_sorted = GS[sort]
    cfigs = cfigs[sort]
    sites = sp.linspace(0,nSites-1,nSites,dtype='uint')
t2 = time.perf_counter()
#---------|MANY-BODY SOLVER: SCAN t,mu WITH FIXED Z=1|----------
'''
for theta in sp.linspace(0,pi/2,10):
    H_f = sp.copy(H_1p)
    H_f[0,0],H_f[1,1] = mu(theta),mu(theta)
    H_f[1,0],H_f[0,1] = t(theta),t(theta)
    eVals,eVecs = sp.linalg.eigh(H_f)
    order = sp.argsort(eVals)
    eVecs=eVecs[:,order]
    dens_matrix = sp.matmul(eVecs[:,0:N_up],eVecs[:,0:N_up].transpose())
    print("Mu = {}, t = {}, Density ={}".format(mu(theta),t(theta),4*dens_matrix[0,0]))
    U_s = schmidt(H_1p,dens_matrix,2)          
    dens_matrix = U_s.transpose()*dens_matrix*U_s
    H_s = U_s.transpose()*H_1p*U_s
    cfigs = f.initialCfig(H_s,sp.diag(dens_matrix),N_up)
    neighs = neighborTable(H_s)
    imp =[0,1]
    binomials = f.binomial(nSites+1)
    sites = sp.linspace(0,nSites-1,nSites,dtype='uint')
    ePrev=0
    GS,GSE,cfigs,dens = solve(cfigs,binomials,H_s,neighs,N_up,N_dn,imp,ePrev,U,U2,J,chPot,output=True,predict=True,cTol=10**-6)
    print("E = {},dens = {}, basis size = {}".format(GSE,4*dens[0],len(cfigs)))
quit()
'''
#---------|CLUSTER ANALYSIS OF RESULTS|----------
#Here is a preliminary cluster analysis of the results. The aim is to interface with a coupled-cluster strategy
#to improve results
'''
for i,c in enumerate(cfigs):
    occup = sites[f.getConfig(N_up,nSites,c['up'],binomials)==1]
    empup = sites[f.getConfig(N_dn,nSites,c['up'],binomials)==0]
    particles_up = sp.in1d(occup,virt)
    holes_up = sp.in1d(empup,core)
    print("Amplitude: {}".format(GS[i]))
    if len(occup[particles_up])>0:
        print('Up particles: {}'.format(occup[particles_up]))
    if len(empup[holes_up]>0):
        print('Up holes: {}'.format(empup[holes_up]))
    print("Amplitude: {}\n".format(GS[i]))

num_holes = []
num_particles = []
for state in cfigs['up']:
    c_up = f.getConfig(N_up,sp.shape(H_s)[0],state,binomials)
    #print(sp.diag(H_s[40:42]))
    #print(len(c_up[2:40]))
    #print(len(c_up[42:]))
    #quit()
    num_holes.append(4-sum(c_up[2:6]))
    num_particles.append(sum(c_up[8:]))
print(sp.unique(num_holes,return_counts=True))
print(sp.unique(num_particles,return_counts=True))
quit()
'''

#---------|COMPUTE OPERATORS FOR SPECTRAL FUNCTION CALCULATION|----------

print("Computing operators")
which = sp.linspace(0,len(cfigs)-1,len(cfigs),dtype='uint')
cfigs_minus,cfigs_plus,a_up,c_up = getImpOperators(cfigs,sites,0,N_up,N_dn,binomials,which)
wplus = sp.linspace(0,len(cfigs_plus)-1,len(cfigs_plus),dtype='uint')
wminus = sp.linspace(0,len(cfigs_minus)-1,len(cfigs_minus),dtype='uint')
H_plus= buildHamiltonian(cfigs_plus,H_s,[0,1],neighs,N_dn,U,U2,J,chPot,binomials,sites,N_up+1,wplus,return_energy=False)
H_minus = buildHamiltonian(cfigs_minus,H_s,[0,1],neighs,N_dn,U,U2,J,chPot,binomials,sites,N_up-1,wminus,return_energy=False)
sp.savez("ops_tdens=%.1f_U=%.2f_c=%.2f_J=%.2f" % (4*dens[0],U/(pi*Delta),cTol,J/U),cfigs_plus=cfigs_plus,cfigs_minus=cfigs_minus,a_up=a_up,c_up=c_up,GSE=GSE,delta=Delta,GS=GS,H_plus=H_plus,H_minus=H_minus)
t3 = time.perf_counter()

print("Done. Ground state calculaton: %.2f min. Computing operators: %.2f min" % ((t2-t1)/60,(t3-t2)/60))
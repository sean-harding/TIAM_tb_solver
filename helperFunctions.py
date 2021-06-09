#Contains helper functions to map a many-body bitstring to a hash index and back again
#Author: Sean M. Harding (University of Bristol,2018)

import scipy as sp
from math import sqrt
def binomial(N):
    B = {(0,0):1,(0,1):1,(1,1):1}
    for n in range(2,N+1):
        B[(0,n)],B[(n,n)] = 1,1
        for k in range(1,n):
            B[k,n] = B[(k,n-1)] + B[(k-1,n-1)]
    return(B)
'----------------------------------------------------------------------------' 
def getIndex(nSites,string,binomial):
    idx = 0
    n=0
    for i in range(nSites):
        if string[i]==1:
            n+=1
            idx += binomial.get((n,i),0)
    return(idx)
'----------------------------------------------------------------------------' 
def getConfig(nParticle,nSites,idx,binomial):
    p,n = idx,nParticle
    string = [0]*nSites
    for i in range(nSites):
        if p>=binomial.get((n,nSites-i-1),0):
            string[nSites-i-1] = 1
            #p -= binomial(n,nSites-i-1)
            p -= binomial.get((n,nSites-i-1),0)
            n-=1
    return sp.array(string)
'----------------------------------------------------------------------------' 
def discretizeFunc(func,numSamples,bandwidth,tpe='lin',logFactor=1.2,b=0.25):
    '''DEPRECIATED'''
    if tpe=='lin':
        dx = 2*bandwidth/numSamples
        bins = [-bandwidth+dx*n for n in range(0,numSamples+1)]
    elif tpe=='log':
        N=(numSamples+2)/2
        N = int(N)
        bins = [logFactor**-m for m in range(N)]
        bins2 = bins.copy()
        bins = [-x for x in bins2]
        bins2.reverse()
        bins = bins+bins2
    elif tpe=='linlog':
        n=int(numSamples/2)
        N=(numSamples-n+2)/2
        N = int(N)
        bins = [logFactor**-m for m in range(N)]
        bins2 = bins.copy()
        bins = [-x for x in bins2]
        bins2.reverse()
        dx = (bins[-1]-bins2[0])/n
        dx = abs(dx)
        bins3 = [bins[-1]+dx*k for k in range(1,n)]
        bins = bins+bins3+bins2
    elif tpe == 'linlog_inverted':
        #There are 46 modes per bath at N=92 bath sites
        n = int(numSamples*0.75/2)     #How many modes to put in the logarithmic part
        N = int(numSamples/2 - n)+1          #How many modes to put in the linear part
        print(n,N)
        b=0.25
        shift = -0.5*(logFactor+1)*(logFactor)**-n
        bins_log = [b*(logFactor)**-m +b*shift for m in range(n)]
        bins_log.reverse()
        dx_lin = (bandwidth-b)/N
        bins_lin = [b+dx_lin*k for k in range(1,N+1)]
        bins = (bins_log+bins_lin).copy()
        bins.reverse()
        bins.extend([-a for a in bins_log+bins_lin])
        bins.reverse()
    samples = [sp.integrate.quad(func,bins[n],bins[n+1])[0] for n in range(0,len(bins)-1)]
    energies = [sp.integrate.quad(lambda x:func(x)*x,bins[n],bins[n+1])[0] for n in range(0,len(bins)-1)]
    energies = map(lambda x,y:x/y,energies,samples)
    samples = map(sqrt,samples)
    print(len(list(energies)))
    quit()
    return list(energies),list(samples)
def initialCfig(H_schmidt,site_occupancies,N_up):
    nSites = len(H_schmidt)
    template = sp.zeros(len(H_schmidt))
    occupied = sp.where(site_occupancies>1-10**-8)
    partially_occupied = sp.where(site_occupancies>10**-8)[0]
    partially_occupied = sp.setdiff1d(partially_occupied,occupied)   #This gives the set exclusive.
    #print('Number of partially occupied orbitals: {}'.format(len(partially_occupied)))
    #partially_occupied = sp.setdiff1d(partially_occupied,imp)
    template[occupied[0]]=1
    MF_solution = []
    binomials = binomial(nSites+1)

    for site_1 in partially_occupied:
        for site_2 in sp.setdiff1d(partially_occupied,site_1):
            cfg_up = sp.copy(template)
            cfg_up[site_1] = 1
            cfg_up[site_2] = 1
            if len(MF_solution)==0:
                MF_solution = cfg_up
            else:
                MF_solution = sp.vstack([MF_solution,cfg_up])
        partially_occupied = sp.setdiff1d(partially_occupied,site_1)
    MF_hashes = []
    for cfg in MF_solution:
        MF_hashes.append(getIndex(nSites,cfg,binomials))
    cfgs_CI = []
    for hash1 in MF_hashes:
        for hash2 in MF_hashes:
            cfgs_CI.append([hash1,hash2,hash1*binomials[N_up,nSites]+hash2])
    cfgs_CI = sp.array(cfgs_CI)
    cfigs = sp.zeros(len(cfgs_CI),dtype=[('up',object),('dn',object),('idx',object)])
    cfigs['up'] = cfgs_CI[:,0]
    cfigs['dn'] = cfgs_CI[:,1]
    cfigs['idx'] = cfgs_CI[:,2]
    return cfigs

def discretize(func,Nl,Nlog,D,lfactor=1.2,b=0.25):
    #Gives 2*(Nl+Nlog+1) modes
    if Nl == 0:
        Nlog+=1
        shift = -0.5*(lfactor+1)*(lfactor)**-Nlog
        k = D/(1+shift)
        bins = [-1*(k*(lfactor)**-m +k*shift) for m in range(Nlog)]  
        bins_r = bins.copy()
        bins_r.reverse()
        bins.extend([-1*a for a in bins_r])
    elif Nlog == 0:
        dx_lin = 2*D/(2*Nl+1)
        bins = [D-dx_lin*k for k in range(0,2*Nl+2)]
        bins.reverse()        
        diff = [abs(bins[n]-bins[n-1]) for n in range(1,len(bins))]
    else:
        shift = -0.5*(lfactor+1)*(lfactor)**-Nlog
        k = b/(1+shift)
        bins_log = [k*(lfactor)**-m +k*shift for m in range(Nlog)]
        bins_log.reverse()
        dx_lin = (1-b)/(Nl+1)
        bins_lin = [b+dx_lin*n for n in range(1,Nl+2)]
        bins = (bins_log+bins_lin).copy()
        bins.reverse()
        bins.extend([-a for a in bins_log+bins_lin])
        bins.reverse()
    samples = [sp.integrate.quad(func,bins[n],bins[n+1])[0] for n in range(0,len(bins)-1)]
    energies = [sp.integrate.quad(lambda x:func(x)*x,bins[n],bins[n+1])[0] for n in range(0,len(bins)-1)]
    energies = map(lambda x,y:x/y,energies,samples)
    samples = map(sqrt,samples)
    return list(samples),list(energies)
import scipy as sp
import helperFunctions as f
import operators_parallel as ops
import multiprocessing as mp
from functools import partial,reduce
from time import perf_counter

def expandBasis(binomials,n_up,n_dn,nSites,neighbors,configs,which,active=False,exchange=False):
    '''Expands Hilbert space'''
    U,D,I = [],[],[]
    sites = sp.linspace(0,nSites-1,nSites,dtype='uint')
    if active is False:
        activeSpace = sites
    else:
        activeSpace = active
    for cfig in configs[which]:
        strings = [f.getConfig(n_up,nSites,cfig[spin],binomials) for spin in ('up','dn')]
        for string,which in zip(strings,('up','dn')):
            for o,u in ((o,u) for o in activeSpace[string[activeSpace]==1] for u in neighbors[o] if string[activeSpace][u]==0):
                cNew = sp.copy(string)
                cNew[o],cNew[u] = 0,1
                if which=='up':
                    U.append(f.getIndex(nSites,cNew,binomials))
                    D.append(cfig['dn'])
                else:
                    U.append(cfig['up'])
                    D.append(f.getIndex(nSites,cNew,binomials))
    I = [u*binomials[n_dn,nSites]+d for (u,d) in zip(U,D)]
    newCfigs = sp.zeros(len(I),dtype = [('up',object),('dn',object), ('idx',object)])
    newCfigs['up'] = U
    newCfigs['dn'] = D
    newCfigs['idx'] = I
    return newCfigs

def addExchangeTerms(cfigs,nUp,nDn,imp,nSites,binomials,which):
    '''Pairs of states are linked by the exchange term. expandBasis may miss one member of these pairs, and 
    it is the job of this function to supplement the cfigs list with any missing states'''
    #print("Manually adding exchange terms")
    #print("There are {} configurations".format(len(cfigs)))
    U,D,I = [],[],[]
    sec = []
    for cfg in cfigs[which]:
        upCfg = f.getConfig(nUp,nSites,cfg['up'],binomials)
        dnCfg = f.getConfig(nDn,nSites,cfg['dn'],binomials)
        sec.append([upCfg[imp],dnCfg[imp]])
    sec = [[(up1,dn1),(up2,dn2)] for [(up1,up2),(dn1,dn2)] in sec]
    doubles = (True if list(map(sum,k))==[2,0] or list(map(sum,k))==[0,2] else False for k in sec)   #True if doubly occupied
    spinFlips = (True if list(map(sum,k))==[1,1] and k[0]!=k[1] else False for k in sec) #True if both sites singly occupied by different spin
    doubles = (cfigs[i] for i,boole in zip(which,doubles) if boole==True)       #BUGGED. Wrong index i
    spinFlips = (cfigs[i] for i,boole in zip(which,spinFlips) if boole==True)
    for conf in doubles:
        c_up = f.getConfig(nUp,nSites,conf['up'],binomials)
        c_dn = f.getConfig(nDn,nSites,conf['dn'],binomials)
        if c_up[imp[0]]==1 and c_dn[imp[0]]==1:      #If double occupancy on site 1 
            c_up[imp[0]],c_dn[imp[0]] = 0,0
            c_up[imp[1]],c_dn[imp[1]] = 1,1
        else:
            c_up[imp[0]],c_dn[imp[0]] = 1,1
            c_up[imp[1]],c_dn[imp[1]] = 0,0
        U.append(f.getIndex(nSites,c_up,binomials))
        D.append(f.getIndex(nSites,c_dn,binomials))

    for conf in spinFlips:
        c_up = f.getConfig(nUp,nSites,conf['up'],binomials)
        c_dn = f.getConfig(nDn,nSites,conf['dn'],binomials)
        if c_up[imp[0]]==1:
            c_up[imp[0]],c_up[imp[1]]=0,1
            c_dn[imp[1]],c_dn[imp[0]]=0,1
        else:
            c_up[imp[0]],c_up[imp[1]]=1,0
            c_dn[imp[1]],c_dn[imp[0]]=1,0
        U.append(f.getIndex(nSites,c_up,binomials))
        D.append(f.getIndex(nSites,c_dn,binomials))
    I = [u*binomials[nDn,nSites]+d for (u,d) in zip(U,D)]
    newCfigs = sp.zeros(len(I),dtype = [('up',object),('dn',object), ('idx',object)])
    newCfigs['up'] = U
    newCfigs['dn'] = D
    newCfigs['idx'] = I
    #print("Generated {} new terms".format(len(newCfigs)))
    return newCfigs

def amps_pert(H_SIAM,GS,GSE,which):
    '''Using second-order perturbation theory, calculates amplitudes of new states perturbatively'''
    E = H_SIAM.diagonal()
    amps = sp.zeros(len(which))
    for i,m in enumerate(which):
        nonz = H_SIAM.getcol(m)[0:len(GS)].nonzero()[0]
        amps[i] = -1*sum(float(a*b) for (a,b) in zip(H_SIAM.getcol(m)[0:len(GS)][nonz].todense(),GS[nonz]))/(E[m]-GSE)
    return amps

def solve(cfigs,binomials,H_s,neighs,N_up,N_dn,imp,ePrev,U,U2,J,chemicalPotential,cTol=10**-8,predict=False,output=True):
    '''The main solver method'''
    converged = False
    nSites = len(H_s)
    k=0
    sites = sp.linspace(0,nSites-1,nSites,dtype='uint')
    while converged == False:
        nGen = len(cfigs)
        t0 = perf_counter()
        #Generate new configs
        which = [k for k in range(len(cfigs))]
        which = sp.array_split(which,4)
        with mp.Pool(4) as p:
            c_new = p.map(partial(expandBasis,binomials,N_up,N_dn,nSites,neighs,cfigs),which)
        c_new = reduce(sp.append,c_new)
        c_new = sp.unique(c_new)
        c_new = sp.setdiff1d(c_new,cfigs)
        new = len(c_new)
        cfigs = sp.append(cfigs,c_new)
        'New code: next 5 lines'
        #exchangeTerms = addExchangeTerms(cfigs,N_up,N_dn,[0,1],nSites,binomials,sabrina)
        #exchangeTerms = sp.unique(exchangeTerms)
        #exchangeTerms = exchangeTerms[sp.isin(exchangeTerms,cfigs,invert=True)]
        #cfigs = sp.append(cfigs,exchangeTerms)
        'End of new code'
        t1 = perf_counter()
        sites = sp.linspace(0,nSites-1,nSites,dtype='uint')
        
        #Calculate new matrix elements of the Hamiltonian involving the new basis states. 
        which = [k for k in range(len(cfigs))]
        which = sp.array_split(which,4)
        with mp.Pool(4) as p:
            res = p.map(partial(ops.buildHamiltonian,cfigs,H_s,imp,neighs,N_dn,U,U2,J,chemicalPotential,binomials,sites,N_up),which)
        H_SIAM = res[0][0]
        E = res[0][1]
        for (a,b) in res[1:]:
            H_SIAM = H_SIAM + a    
            E = sp.append(E,b)
        t2 = perf_counter()

        #Filter new amplitudes perturbatively
        which =sp.array_split([nGen+k for k in range(new)],4)
        if k>0 and predict==True:
            with mp.Pool(4) as p:
                amps_p = p.map(partial(amps_pert,H_SIAM,GS,GSE),which)
            amps_p = reduce(sp.append,amps_p)
            
            chop = [True]*nGen
            chop.extend([True if a**2>cTol*0.1 else False for a in amps_p])
            chop = sp.array(chop)
            nPert = len(cfigs)-nGen
            cfigs = cfigs[chop]
            H_SIAM = H_SIAM[:,chop]
            H_SIAM = H_SIAM[chop,:]
            trunc_pert = nPert-(len(cfigs)-nGen)
            #print("Number of states truncated: %.1i of %.1i (~%.1i%%)" % (trunc_pert,new,trunc_pert*100/new))
        t3 = perf_counter()

        #Now diagonalize Hamiltonian in full space to obtain new amplitudes
        GSE,GS = sp.sparse.linalg.eigsh(H_SIAM,k=1,which='SA')
        chop = sp.array([True if amp**2>cTol else False for amp in GS ])
        H_SIAM = H_SIAM[:,chop]
        H_SIAM = H_SIAM[chop,:]
        GS = GS[chop]
        truncatedWeight = 1-sp.linalg.norm(GS)
        GS = GS/sp.linalg.norm(GS)
        cfigs = cfigs[chop]
        ops_up = ops.numberOperators(cfigs,N_up,nSites,imp,binomials,'up',range(len(cfigs)))
        GS = sp.matrix(GS)
        D = []
        for op in ops_up:
            dens = float((GS.transpose()*op*GS))
            D.append(dens)
        t4 = perf_counter()
        if output==True:
            print("Density = {}".format(D))
            print("Basis size= {}".format(len(cfigs)))
            print("Energy, dE = {}, {}".format(float(GSE),abs(ePrev-float(GSE))))
            print("Truncated weight= {}".format(truncatedWeight))
            print("Timing:\nCalculating new basis:%.2f\nBuilding hamiltonian:%.2f\nPerturbative filtering:%.2f\nDiagonalization:%.2f" % (t4-t3,t3-t2,t2-t1,t1-t0))
        if abs(ePrev-float(GSE))<10**-8 or len(cfigs)==nGen:
            converged=True    
        ePrev = float(GSE)
        k+=1    
    return GS,GSE,cfigs,D
    
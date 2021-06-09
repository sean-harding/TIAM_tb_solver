import scipy as sp
import scipy.sparse
import helperFunctions as f

def getKinetic_p(cfigs,binomials,nUp,nDn,neighbors,tij,sites,D,which):
    '''Returns the kinetic part of the Hamiltonian between states in cfigs and cfigs_a[0:which]
    we put the 'which' statement at the end to allow paralellization with the "partial" function'''
    L = len(tij)
    I = []
    J = []
    Hij = []
    #E = []
    #sites = sp.linspace(0,len(tij)-1,len(tij),dtype='uint')
    energies = sp.diag(tij)
    #Lookup dictionary to speedup getting matrix elements
    #D = {cfg['idx']:pos for pos,cfg in enumerate(cfigs)}  #Loop through cfigs and calc. matrix elements to cfigs_a
    #for i,cfg in zip(which,cfigs[which]):                 #Slicing arrays does not make a copy. This is the best way to do parallelization
    #for i,cfg in enumerate(cfigs):
    for i,cfg in zip(which,cfigs[which]):
        strings = [f.getConfig(N,L,cfg[spin],binomials) for spin,N in zip(('up','dn'),(nUp,nDn))]   #Check correct particle number is being used
        occs = (sites[c==1] for c in strings)
        #E.append(sum(map(sum,(energies[sp.array(s)==1] for s in strings)))) #Work out 1-particle energy
        for bitString,occ,spin in zip(strings,occs,('up','dn')):    #Strings,occupancies,spin flavour
            for o,u in ((o,u) for o in occ for u in neighbors[o] if bitString[u]==0):
                cNew = sp.copy(bitString)
                cNew[o],cNew[u] = 0,1
                if spin=='up':
                    hsh = f.getIndex(L,cNew,binomials)*binomials[(nDn,L)] + cfg['dn']        #Get FULL index of new state 
                else:
                    hsh = cfg['up']*binomials[(nDn,L)] + f.getIndex(L,cNew,binomials)
                try:
                    mIndex = D[hsh]
                    phase = (-1.0)**int(sum(cNew[min(o,u)+1:max(o,u)]))  #Compute Fermi phase factor
                    Hij.append(phase*tij[o,u])                                                #Use python lists rather than numpy arrays as these are very very slow when appending
                    I.append(i)
                    J.append(mIndex)
                except:
                    pass  
    Tij = sp.sparse.coo_matrix((Hij,(I,J)),shape=(len(cfigs),len(cfigs)))
    Tij = Tij+ Tij.transpose()
    #H = sp.sparse.diags(E,0) + Tij
    return Tij

def exchange_p(cfigs,nUp,nDn,imp,nSites,binomials,D,which):
    #D = {b['idx']:a for a,b in enumerate(cfigs)}    #Only search for cfgs in cfgs2
    sec = []
    for cfg in cfigs[which]:
        upCfg = f.getConfig(nUp,nSites,cfg['up'],binomials)
        dnCfg = f.getConfig(nDn,nSites,cfg['dn'],binomials)
        sec.append([upCfg[imp],dnCfg[imp]])
    sec = [[(up1,dn1),(up2,dn2)] for [(up1,up2),(dn1,dn2)] in sec]
    doubles = (True if list(map(sum,k))==[2,0] or list(map(sum,k))==[0,2] else False for k in sec)   #True if doubly occupied
    #doubles_2 = (True if list(map(sum,k))==[0,2] else False for k in sec)   #True if site 2 doubly occupied
    spinFlips = (True if list(map(sum,k))==[1,1] and k[0]!=k[1] else False for k in sec) #True if both sites singly occupied by different spin
    #oneUpDn = (True if list(map(sum,k))==[1,1] else False for k in sec) #One up and one down on different sites
    doubles = (cfigs[i] for i,boole in zip(which,doubles) if boole==True)       #BUGGED. Wrong index i
    #doubles_2 = [cfg[i] for i,boole in enumerate(doubles_2) if boole==True]
    spinFlips = (cfigs[i] for i,boole in zip(which,spinFlips) if boole==True)
    #raise Exception("done")
    pairsSF,pairsPH = [],[]
    for conf in doubles:
        c_up = f.getConfig(nUp,nSites,conf['up'],binomials)
        c_dn = f.getConfig(nDn,nSites,conf['dn'],binomials)
        if c_up[imp[0]]==1 and c_dn[imp[0]]==1:      #If double occupancy on site 1 
            c_up[imp[0]],c_dn[imp[0]] = 0,0
            c_up[imp[1]],c_dn[imp[1]] = 1,1
        else:
            c_up[imp[0]],c_dn[imp[0]] = 1,1
            c_up[imp[1]],c_dn[imp[1]] = 0,0
        hsh = f.getIndex(nSites,c_up,binomials)*binomials[nDn,nSites]+f.getIndex(nSites,c_dn,binomials)
        try:
            mIndex = D[hsh]
            pairsPH.append([D[conf['idx']],mIndex])
        except:
            pass
    for conf in spinFlips:
        c_up = f.getConfig(nUp,nSites,conf['up'],binomials)
        c_dn = f.getConfig(nDn,nSites,conf['dn'],binomials)
        if c_up[imp[0]]==1:
            c_up[imp[0]],c_up[imp[1]]=0,1
            c_dn[imp[1]],c_dn[imp[0]]=0,1
        else:
            c_up[imp[0]],c_up[imp[1]]=1,0
            c_dn[imp[1]],c_dn[imp[0]]=1,0
        hsh = f.getIndex(nSites,c_up,binomials)*binomials[nDn,nSites]+f.getIndex(nSites,c_dn,binomials)
        try:
            mIndex = D[hsh]
            pairsSF.append([D[conf['idx']],mIndex])
        except:
            pass
    pairsSF = sp.array(pairsSF)
    pairsPH = sp.array(pairsPH)
    if len(pairsSF)!=0 and len(pairsPH) !=0:
        H_spinFlip = sp.sparse.coo_matrix((sp.ones(len(pairsSF)),(pairsSF[:,0],pairsSF[:,1])),shape=(len(cfigs),len(cfigs)))
        H_pairHopping = sp.sparse.coo_matrix((sp.ones(len(pairsPH)),(pairsPH[:,0],pairsPH[:,1])),shape=(len(cfigs),len(cfigs)))
        #H_ex = H_spinFlip - H_pairHopping
        H_ex = H_pairHopping-H_spinFlip
        return H_ex.tocsr()
    elif len(pairsPH)!=0:
        print("No spin flips")
        H_ex = sp.sparse.coo_matrix((sp.ones(len(pairsPH)),(pairsPH[:,0],pairsPH[:,1])),shape=(len(cfigs),len(cfigs)))
        return H_ex.tocsr()
    elif len(pairsSF)!=0:
        print("No pair hopping")
        H_ex = -1*sp.sparse.coo_matrix((sp.ones(len(pairsSF)),(pairsSF[:,0],pairsSF[:,1])),shape=(len(cfigs),len(cfigs)))
        return H_ex.tocsr()
    
    else:
        print('No exchange terms present!')
        pass
        #raise Exception('No exchange terms present!')
        #return H_ex.tocsr()

def numberOperators(cfigs,N,nSites,sites,binomials,spin,which):
    #Currently only works for cfigs being the full configuration list
    occs = []
    for cfig in cfigs[which]:
        occs.append([f.getConfig(N,nSites,cfig[spin],binomials)[s] for s in sites])
    #K = ([o[k] for o in occs] for k in sites)
    #return [sp.sparse.diags(x) for x in ([o[k] for o in occs] for k in sites)]
    return [sp.sparse.coo_matrix((x,(which,which)),shape=(len(cfigs),len(cfigs))) for x in ([o[k] for o in occs] for k in sites)]
def buildHamiltonian(cfigs,H_1p,imp,neighbors,N_dn,U,U2,J,mu,binomials,sites,N_up,which,return_energy=True):
    ##Need to move N_up to the end of the function call
    #cfigs,H_s,imp,neighs,N_up,N_dn,U,U2,J,chemicalPotential,binomials,len(cfigs),which
    D = {b['idx']:a for a,b in enumerate(cfigs)}
    H_kin = getKinetic_p(cfigs,binomials,N_up,N_dn,neighbors,H_1p,sites,D,which)
    #figs,binomials,nUp,nDn,neighbors,tij,sites,D,which
    num_up = numberOperators(cfigs,N_up,len(H_1p),imp,binomials,'up',which)     
    num_dn = numberOperators(cfigs,N_dn,len(H_1p),imp,binomials,'dn',which)   
    E = sp.zeros(len(which))
    for i,cfg in enumerate(cfigs[which]):
        strings = [f.getConfig(N,len(H_1p),cfg[spin],binomials) for spin,N in zip(('up','dn'),(N_up,N_dn))]
        E[i]=sum(map(sum,(sp.diag(H_1p)[c==1] for c in strings)))
        #E.append(sum(map(sum,(energies[sp.array(s)==1] for s in strings)))) #Work out 1-particle energy
    H_SIAM = sp.sparse.coo_matrix((E,(which,which)),shape=(len(cfigs),len(cfigs))) + H_kin
    H_SIAM = H_SIAM +  U*(num_up[0]*num_dn[0]+num_up[1]*num_dn[1])
    H_SIAM = H_SIAM + U2*(num_up[0]*num_dn[1]+num_up[1]*num_dn[0])
    H_SIAM = H_SIAM + (U2-J)*(num_up[0]*num_up[1]+num_dn[0]*num_dn[1])
    H_SIAM = H_SIAM - mu*(num_up[0]+num_up[1]+num_dn[0]+num_dn[1])
    
    if J!=0:
        H_ex = exchange_p(cfigs,N_up,N_dn,imp,len(H_1p),binomials,D,which)  #Could put in try/except loop
        if H_ex is not None:
            H_SIAM = H_SIAM + J*H_ex    #Change back to plus
        else:
            print("No exchange terms")
        #cfigs,nUp,nDn,imp,nSites,binomials,D,which
    if return_energy==True:
        return H_SIAM,E
    else:
        return H_SIAM
def getImpOperators(cfigs,sites,impurity,N_up,N_dn,binomials,which,spin='up'):
    def getNew(string,p,what):
        #n,nSites,index,binomials
        new = sp.copy(string)
        if what=='a':
            new[p] = 0
        else:
            new[p] = 1
        return new

    #Create all N+1 and N-1 configs
    HSdim = len(cfigs)
    nSites = len(sites)
    cfigsPlus,cfigsMinus,c_pairs,a_pairs,S = [],[],[],[],[]
    for i,cfig in zip(which,cfigs[which]):
        string = f.getConfig(N_up,nSites,cfig[spin],binomials)
        gens_a = ([getNew(string,s,'a'),s] for s in sites if string[s]==1)
        gens_c = ([getNew(string,s,'c'),s] for s in sites if string[s]==0)
        for g in gens_a:
            ind = f.getIndex(nSites,g[0],binomials)
            cfigsMinus.append((ind,cfig['dn'], ind*binomials[N_dn,nSites]+cfig['dn']))  #Is N_dn -> N_up??
            if g[1]==impurity:
                phase = (-1)**sp.sum(g[0][0:impurity])
                a_pairs.append((ind*binomials[N_dn,nSites]+cfig['dn'],i,phase))
        for g in gens_c:
            ind = f.getIndex(nSites,g[0],binomials)
            cfigsPlus.append((ind,cfig['dn'], ind*binomials[N_dn,nSites]+cfig['dn']))
            if g[1]==impurity: 
                phase = (-1)**sp.sum(g[0][0:impurity])
                c_pairs.append((ind*binomials[N_dn,nSites]+cfig['dn'],i,phase))
    #Remove duplicates
    cfigsMinus = sp.array(cfigsMinus,dtype=object)
    uniques = sp.unique(cfigsMinus[:,2],return_index=True)[1]
    cfigsMinus = cfigsMinus[uniques]
    Dminus = {hsh[2]:i for i,hsh in enumerate(cfigsMinus)}   

    cfigsPlus = sp.array(cfigsPlus,dtype=object)
    uniques = sp.unique(cfigsPlus[:,2],return_index=True)[1]
    cfigsPlus = cfigsPlus[uniques]
    Dplus = {hsh[2]:i for i,hsh in enumerate(cfigsPlus)}

    a_pairs = sp.array(a_pairs,dtype=object)    
    uniques = sp.unique(a_pairs[:,0],return_index=True)[1]
    a_pairs = a_pairs[uniques]

    c_pairs = sp.array(c_pairs,dtype=object)
    uniques = sp.unique(c_pairs[:,0],return_index=True)[1]
    c_pairs = c_pairs[uniques]
    #Now need to create the c/a operators
    a_imp = sp.sparse.coo_matrix((a_pairs[:,2],(a_pairs[:,1],[Dminus[k] for k in a_pairs[:,0]])), shape=(len(cfigs),len(cfigsMinus)),dtype='float64')
    c_imp = sp.sparse.coo_matrix((c_pairs[:,2],(c_pairs[:,1],[Dplus[k] for k in c_pairs[:,0]])), shape=(len(cfigs),len(cfigsPlus)),dtype='float64')
    cfigs_list = [sp.zeros(size,dtype = [('up',object),('dn',object),('idx',object)]) for op,size in zip((a_imp,c_imp),(len(cfigsMinus),len(cfigsPlus)))]
    for clist,pairs in zip(cfigs_list,[cfigsMinus,cfigsPlus]):
        clist['up'] = pairs[:,0]
        clist['dn'] = pairs[:,1]
        clist['idx'] = pairs[:,2]
    return cfigs_list[0],cfigs_list[1],(a_imp.tocsr()).transpose(),(c_imp.tocsr()).transpose()
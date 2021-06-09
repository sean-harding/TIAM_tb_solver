import scipy as sp
import scipy.linalg
import scipy.integrate

def band_diag(H_filled,block):
    H_filled = sp.matrix(H_filled)
    phi = [H_filled[n,block[-1]+1:] for n in block]
    M = [sp.outer(phi_n,phi_n) for phi_n in phi]
    M = [sp.matrix(m) for m in M]
    eigs = [sp.linalg.eigh(m) for m in M]   #Produces (eigval,eigvec) pairs
    D = [sp.matrix(evecs) for (evals,evecs) in eigs]

    directions = [d[:,-1] for d in D]
    'Nonzero eigenvectors are last'
    if len(directions)>1:
        direction_1 = directions[0]     #Currently just set up for two orbitals. Will have to extend
        direction_2 = directions[1]     #To a full Gram-Schmidt proceedure
        orth_1 = direction_2
        orth_2 = direction_1 - float(direction_1.transpose()*direction_2)*direction_2
        orth_2 = orth_2/sp.linalg.norm(orth_2)
        orth =[orth_1,orth_2]
    else:
        orth = [directions[0]]
    if len(orth)>1:
        stack = sp.append(orth[0],orth[1],axis=1)
        k=2
        while k<len(orth):
            stack = sp.append(stack,orth[k],axis=1)
            k+=1
        stack = stack.transpose()
        zero_vectors = sp.linalg.null_space(stack)
        #raise Exception("Done")
        for o in orth:
            zero_vectors = sp.append(o,zero_vectors, axis=1)
    else:
        zero_vectors = sp.linalg.null_space(orth[0].transpose())
        zero_vectors = sp.append(orth[0],zero_vectors, axis=1)
    U = sp.linalg.block_diag(sp.eye(block[-1]+1),zero_vectors)
    return sp.matrix(U)

def schmidt(H,DM,impuritySize):
    H_initial = sp.copy(H)
    H_initial = sp.matrix(H_initial)
    #First diagonalize impurity part of the density matrix and using this unitary, transform the impurity part of the site basis
    #u_imp = sp.linalg.eigh(DM[0:impuritySize,0:impuritySize])
    u_imp = sp.eye(impuritySize)
    U_imp = sp.linalg.block_diag(u_imp,sp.eye(len(H)-impuritySize))
    unitary = sp.matrix(U_imp)
    H2 = U_imp.transpose()*H_initial*U_imp
    #Do the same for the bath part and get the number of filled and empty sites
    u_rho = sp.linalg.eigh(DM[impuritySize:,impuritySize:])
    u2 = sp.matrix(u_rho[1])
    filled = len(u_rho[0][u_rho[0]>1-10**-8])
    empty = len(u_rho[0][u_rho[0]<10**-8])
    sort = sp.argsort(u_rho[0])
    sort = sp.flip(sort)
    u2 = u2[:,sort]
    U = sp.linalg.block_diag(sp.eye(impuritySize),u2)
    U = sp.matrix(U)
    H2 = U.transpose()*H2*U
    unitary = unitary*U
    #Extract submatrices for occupied/unoccupied
    H_filled = H2[0:impuritySize+filled,0:impuritySize+filled]
    H_empty = H2[impuritySize+filled:,impuritySize+filled:]
    first = impuritySize
    U_filled,U_empty = sp.eye(len(H_filled)),sp.eye(len(H_empty))
    U_filled,U_empty = sp.matrix(U_filled),sp.matrix(U_empty)
    block = [k for k in range(impuritySize)]
    while(len(H_filled)-first)>=impuritySize+1:
        u_filled = band_diag(H_filled,block)
        H_filled = u_filled.transpose()*H_filled*u_filled
        U_filled = U_filled*u_filled
        first+=impuritySize
        block = [b+impuritySize for b in block]
    first = impuritySize
    block = [k for k in range(impuritySize)]
    while(len(H_empty)-first)>=impuritySize+1:
        u_empty = band_diag(H_empty,block)
        H_empty = u_empty.transpose()*H_empty*u_empty
        U_empty = U_empty*u_empty
        first+=impuritySize
        block = [b+impuritySize for b in block]
    U_schmidt = sp.linalg.block_diag(U_filled,U_empty)
    U_schmidt = sp.matrix(U_schmidt)
    return unitary*sp.matrix(U_schmidt)

def neighborTable(H):
    '''Loop through a one-particle matrix and construct a neighbor table. Should limit this to the upper triangle
    for speed, but for <100 sites it isn't very slow'''
    neighborTable = {}
    for i in range(len(H)):
        n=[]
        for j in range(len(H)):
            if abs(H[i,j])>10**-10 and j!=i:
                n.append(j)
        neighborTable[i] = n
    return neighborTable
# Homework 4
# Ethan Watanabe
# Credit to Professor Siefman for all of the one-group diffusion solver functions and the inverse power block function

import numpy as np
import matplotlib.pyplot as plt
from matrix_solvers import gaussian_elimination, gauss_seidel, jacobi_iteration

def create_grid(R,I):
    
    '''
    Create cell edges and centers for a domain of
    size R and for I cells
    
    Args:
        R: size of domain
        I: number of cells
        
    Returns:
        Delta_r: width of each cell
        centers: cell centers of the grid
        edges: cell edges of the grid
    '''
    
    Delta_r = float(R)/I # divide size by # of cells
    # calculate the centers by getting each Delta_r
    # and adding 0.5 Delta_R
    centers = np.arange(I)*Delta_r + 0.5*Delta_r
    # Get edges by going beyond one cell
    edges = np.arange(I+1)*Delta_r
    
    return Delta_r, centers, edges
    
def diffusion_solver(R, I,
                     D, Sig_a, nuSig_f, Q,
                     BC, geometry):
    
    '''
    Solve the diffusion equation in a 1-D geometry
    use cell-averaged quantities
    
    Args:
        R: size of domain
        I: number of cells
        D: function, D(r), returns diffusion coefficent at r
        Sig_a: function, Sig_a(r), returns macroscopic abs xs at r
        nuSig_f: function, nuSig_f(r), return nu*macroscopic fission xs at r
        Q: function, Q(r),  returns source at r
        BC: boundary conditions at r=R in form [A,B,C]
        geometry:
            0 = slab
            1 = cylindrical
            2 = spherical
    
    Returns:
        centers: cell centers of grid
        phi: cell-averaged value of scalar flux
    '''
    
    Delta_R, centers, edges = create_grid(R,I)
    # Matrices for Ax=b solve, or A\phi=b
    A = np.zeros((I+1, I+1)) 
    b = np.zeros(I+1)
    
    # define vectors of the surface areas S at cell edges
    # and  volumes V at dr
    if geometry == 0:  # slab
        S = np.zeros_like(edges)+1 # 1-D slab surface area
        S[0] = 0. # at the center make it zero, forces reflective BCs
        # in slab it's dV = dr
        V = np.zeros_like(edges)+ + Delta_R
    
    elif geometry == 1: # cylinder
        # cylinder surface area is 4pi*r^2
        S = 2.*np.pi*edges
        # cylinder differential volume is 4pi*r^2
        # must substract inner cylinder from next outer cylinder
        V = np.pi * (edges[1:I+1]**2 - edges[0:I]**2)
    
    elif geometry == 2: # sphere
        # sphere surface area = 4 pi r^2
        S = 4*np.pi*edges**2
        # volume is 4/3 pi r^3, must subtract inner sphere from outer sphere
        V = 4/3 * np.pi * (edges[1:I+1]**3 - edges[0:I]**3)
    
    # Setup the BC at R
    A[I,I] = (BC[0]/2 + BC[1]/Delta_R)
    A[I,I-1] = (BC[0]/2 - BC[1]/Delta_R)
    b[I] = BC[2]

    r = centers[0]
    Dplus = 0
    # fill A matrix
    for i in range(I):
        r = centers[i]
        Dminus = Dplus
        Dplus = 2*(D(r)*D(r+Delta_R)) / (D(r)+D(r+Delta_R))
        A[i,i] = 1/(Delta_R * V[i])*Dplus*S[i+1] + Sig_a(r) - nuSig_f(r)
        
        if i > 0:
            A[i,i-1] = -1*Dminus/(Delta_R*V[i])*S[i]
            A[i,i] += 1./(Delta_R*V[i])*Dminus*S[i]
        
        A[i,i+1] = -Dplus/(Delta_R*V[i])*S[i+1]
        b[i] = Q(r)

    # solve the system: YOU MUST USE YOUR OWN SOLVER, use function below to verify
    # phi = np.linalg.solve(A, b)
    phi, aug_m = gaussian_elimination(A, b)
    # remove last element of phi because it's outside the domain
    phi = phi[0:I]
    
    return centers, phi


def LU_factor(A):

    # Find the factors of L and U and store all in A
    # L has implied 1s along its diagonal

    N = A.shape[0]
    
    for col in range(0,N):
        for row in range(col+1,N):
            mod_row = np.copy(A[row])
            factor = mod_row[col]/A[col, col]

            mod_row -= factor*A[col,:]
            
            # store the factor in the modified row
            mod_row[col] = factor
            # store only the part of the modified row that makes U
            mod_row = mod_row[col:N]

            A[row, col:N] = mod_row
            
    return 

def LU_solve(A, b):

    [Nrow, Ncol] = A.shape
    N = Nrow
    x = np.zeros(N)
    y = np.zeros(N) # dummy variable for L^-1 b

    # do a forward solve for y in Ly = b
    # Remember L is stored as a lower triangular matrix in A
    # The real L has 1s along diagonal
    for row in range(N):
        RHS = b[row]
        for col in range(row):
            RHS -= y[col]*A[row,col]
        y[row] = RHS

    # Do a backwards solve
    for row in range(N-1, -1, -1):
        RHS = y[row]
        for col in range(row+1, N):
            RHS -= x[col]*A[row,col]
        
        x[row] = RHS/A[row,row]

    return x


def inverse_power(A, B, epsilon=1e-6):
    
    '''
    Solve the generalized eigenvalue problem Ax = lB
    using the inverse power iteration algorithm
    
    Args:
        A: LHS matrix (cannot be singular)
        B: The RHS matrix
        epsilon: tolerance on eigenvalue
    Output:
        l: smallest eigenvalue of the problem
        x: associated eigenvector of smallest eigenvalue
    '''
    
    Nrows, Ncols = A.shape
    # Generate guess
    x = np.random.random((Nrows))
    x = x / np.linalg.norm(x)  # make norm(x) = 1
    
    l_old = 0
    converged = 0
    # compute LU factorization of A
    LU_factor(A)
    iteration = 0
    b_0s = []
    while not(converged):
        
        iteration += 1
        
        b = LU_solve(A, np.dot(B,x))
        b_0s.append(b[0])
        l = np.linalg.norm(b)
        x = b/l
        
        converged = (np.fabs(l-l_old) < epsilon)
        # print(f"Iteration: {iteration}, magnitude of l: {1/l:.4f}, epsilon:{np.fabs(l-l_old):.3e}")
        l_old = l
    
    sign = b_0s[iteration-1]/b_0s[iteration-2]

    return sign/l, x

def diffusion_setup(R, I, D, Sig_a, nuSig_f, BC, geometry):
    
    '''
    Solve the diffusion equation in a 1-D geometry
    use cell-averaged quantities
    
    Args:
        R: size of domain
        I: number of cells
        D: function, D(r), returns diffusion coefficent at r
        Sig_a: function, Sig_a(r), returns macroscopic abs xs at r
        nuSig_f: function, nuSig_f(r), return nu*macroscopic fission xs at r
        Q: function, Q(r),  returns source at r
        BC: boundary conditions at r=R in form [A,B,C]
        geometry:
            0 = slab
            1 = cylindrical
            2 = spherical
    
    Returns:
        centers: cell centers of grid
        phi: cell-averaged value of scalar flux
    '''
    
    Delta_R, centers, edges = create_grid(R,I)
    # Matrices for A and B containing loss and source terms, respectively
    A = np.zeros((I+1, I+1)) 
    B = np.zeros((I+1, I+1)) 
    
    # define vectors of the surface areas S at cell edges
    # and  volumes V at dr
    if geometry == 0:  # slab
        S = np.zeros_like(edges)+1 # 1-D slab surface area
        S[0] = 0. # at the center make it zero, forces reflective BCs
        # in slab it's dV = dr
        V = np.zeros_like(edges)+ + Delta_R
    
    elif geometry == 1: # cylinder
        # cylinder surface area is 4pi*r^2
        S = 2.*np.pi*edges
        # cylinder differential volume is 4pi*r^2
        # must substract inner cylinder from next outer cylinder
        V = np.pi * (edges[1:I+1]**2 - edges[0:I]**2)
    
    elif geometry == 2: # sphere
        # sphere surface area = 4 pi r^2
        S = 4*np.pi*edges**2
        # volume is 4/3 pi r^3, must subtract inner sphere from outer sphere
        V = 4/3 * np.pi * (edges[1:I+1]**3 - edges[0:I]**3)
    
    # Setup the BC at R
    A[I,I] = (BC[0]/2 + BC[1]/Delta_R)
    A[I,I-1] = (BC[0]/2 - BC[1]/Delta_R)

    # fill A matrix
    Dplus = 0
    for i in range(I):
        r = centers[i]
        Dminus = Dplus
        Dplus = 2*(D(r)*D(r+Delta_R)) / (D(r)+D(r+Delta_R))
        A[i,i] = 1/(Delta_R * V[i])*Dplus*S[i+1] + Sig_a(r)
        B[i,i] = nuSig_f(r)
        
        if i > 0:
            A[i,i-1] = -1*Dminus/(Delta_R*V[i])*S[i]
            A[i,i] += 1./(Delta_R*V[i])*Dminus*S[i]
        
        A[i,i+1] = -Dplus/(Delta_R*V[i])*S[i+1]
     
    return centers, A, B
        
def DiffusionEigenvalue(R, I,
                        D, Sig_a, nuSig_f,
                        BC, geometry, epsilon=1e-8):
    
    
    centers, A, B = diffusion_setup(R, I, D, Sig_a, nuSig_f, BC, geometry)
    l, phi = inverse_power(A,B,epsilon)
    # transform back from standard eigenvalue problem to generalized
    k = 1/l 
    phi = phi[0:I]
    
    return k, phi, centers

# The following works under the assumption that there is zero "upscatter"
#  we also assume that there are only two energy groups (fast and thermal)
#  NOT made for general two-group diffusion solvers
def two_group_setup(R, G, I, D, Sig_r, nuSig_f, Sig_s, BC, geometry):
    # let's build the M matrix and F matrix
    Delta_R, centers, edges = create_grid(R,I)
    # this for loop will populate the diagonal block matrices (row==col)
    M_matricies = {"M11": np.array([]),
                   "M22": np.array([])}
    for g in range(G):
        
        # Matrices for Ax=b solve, or A\phi=b
        A = np.zeros((I+1, I+1)) 
        
        # define vectors of the surface areas S at cell edges
        # and  volumes V at dr
        if geometry == 0:  # slab
            S = np.zeros_like(edges)+1 # 1-D slab surface area
            S[0] = 0. # at the center make it zero, forces reflective BCs
            # in slab it's dV = dr
            V = np.zeros_like(edges)+ + Delta_R
        
        elif geometry == 1: # cylinder
            # cylinder surface area is 4pi*r^2
            S = 2.*np.pi*edges
            # cylinder differential volume is 4pi*r^2
            # must substract inner cylinder from next outer cylinder
            V = np.pi * (edges[1:I+1]**2 - edges[0:I]**2)
        
        elif geometry == 2: # sphere
            # sphere surface area = 4 pi r^2
            S = 4*np.pi*edges**2
            # volume is 4/3 pi r^3, must subtract inner sphere from outer sphere
            V = 4/3 * np.pi * (edges[1:I+1]**3 - edges[0:I]**3)
        
        # Setup the BC at R
        A[I,I] = (BC[0]/2 + BC[1]/Delta_R)
        A[I,I-1] = (BC[0]/2 - BC[1]/Delta_R)

        r = centers[0]
        Dplus = 0
        # fill A matrix
        for i in range(I):
            r = centers[i]
            Dminus = Dplus
            Dplus = 2*(D(r, g)*D(r+Delta_R, g)) / (D(r, g)+D(r+Delta_R, g))
            A[i,i] = 2/(Delta_R * V[i])*Dplus*S[i+1] + Sig_r(r, g)
            
            if i > 0:
                A[i,i-1] = -1*Dminus/(Delta_R*V[i])*S[i]
                A[i,i] += 1./(Delta_R*V[i])*Dminus*S[i]
            
            A[i,i+1] = -Dplus/(Delta_R*V[i])*S[i+1]
        M_matricies.update({f"M{g+1}{g+1}": A})
    
    M11 = M_matricies.get("M11")
    M22 = M_matricies.get("M22")
    # next we will populate the intergroup coupling matrix 0-->1 (down scattering)
    for g in range(G-1):
        A = np.zeros((I+1, I+1))
        rows, cols = A.shape
        for i in range(rows-1):
            A[i,i] = -1*Sig_s(centers[i])
    M21 = A.copy()
    
    # next we will construct the F block matrix
    P11 = np.zeros([I+1])
    P12 = np.zeros([I+1])
    for r in range(I):
        P11[r] = nuSig_f(centers[r], 0)
        P12[r] = nuSig_f(centers[r], 1)
    P11 = np.diag(P11)
    P12 = np.diag(P12)
    return M11, M21, M22, P11, P12

def inverse_power_block(M11, M21, M22, P11, P12, epsilon=1e-6):
    Nrows, Ncols = M11.shape
    # Generate guess
    x1 = np.random.random((Nrows))
    x2 = np.random.random((Nrows))

    # Normalize Initial Guess, avoid uncontrolled growth during iteration
    l_old = np.linalg.norm(np.concatenate((x1, x2)))
    x1 = x1 / l_old # make norm(x) = 1
    x2 = x2 / l_old # make norm(x) = 1

    converged = 0
    # compute LU factorization of A, only done once to save time
    LU_factor(M11)
    LU_factor(M22)
    iteration = 0
    while not(converged):
        iteration += 1
        # compute the fission sources
        S11 = np.dot(P11, x1) # in group 1 from fission in group 1
        S12 = np.dot(P12, x2) # in group 2 from fission in group 2
        # Solve for unnormalized g1 flux
        b1 = LU_solve(M11, S11 + S12)

        # downscatter-driven flux calculation
        downscatter = np.dot(-M21, b1)
        # solve for unnormalized g2 flux
        b2 = LU_solve(M22, downscatter)

        # compute eigenvalue
        l = np.linalg.norm(np.concatenate((b1, b2)))
        # get eigenvectors, normalizing by 1 to prevent unbounded growth
        x1 = b1/l
        x2 = b2/l

        converged = (np.fabs(l-l_old) < epsilon)
        # print(f"Iteration: {iteration}, magnitude of l: {1/l:.6f}, epsilon:{np.fabs(l-l_old):.3e}")
        l_old = l

        k = 1./l
    return k, x1, x2

if __name__ == '__main__':
    import pandas as pd
    R = 200
    I = 10
    D = lambda r, g: 1.2*(g==0) + 0.4*(g==1 and r<=180) + 0.2*(g==1 and r>180)
    def Sig_r(r, g):
        if g == 0:
            if 20<r<=180:
                value = 0.02965
            elif r<=20:
                value = 0.02982
            elif r>180:
                value = 0.051
        elif g==1:
            if r<=20:
                value = 0.09848
            elif r>180:
                value = 0.04
            elif 20<r<=40 or 60<r<=80 or 100<r<=120 or 140<r<=160:
                value = 0.23633
            elif 40<r<=60 or 80<r<=100 or 120<r<=140 or 160<r<=180:
                value = 0.09308
        return value
    def nuSig_f(r, g):
        if g==0:
            if r<=20 or 40<r<=60 or 80<r<=100 or 120<r<=140 or 160<r<=180:
                value = 0.00457
            elif 20<r<=40 or 60<r<=80 or 100<r<=120 or 140<r<=160:
                value = 0.00685
            else:
                value = 0
        elif g==1:
            if r<=20 or 40<r<=60 or 80<r<=100 or 120<r<=140 or 160<r<=180:
                value = 0.1142
            elif 20<r<=40 or 60<r<=80 or 100<r<=120 or 140<r<=160:
                value = 0.3519
            else:
                value = 0
        return value
    def Sig_s(r):
        if r<=20 or 40<r<=60 or 80<r<=100 or 120<r<=140 or 160<r<=180:
            value = 0.02043
        elif 20<r<=40 or 60<r<=80 or 100<r<=120 or 140<r<=160:
            value = 0.01587
        elif r>180:
            value = 0.05
        return value
    BC = [0, 1, 0]

    M11, M21, M22, P11, P12 = two_group_setup(R, 2, I, D, Sig_r, nuSig_f, Sig_s, BC, geometry=1)
    k, phi1, phi2 = inverse_power_block(M11, M21, M22, P11, P12)
    # print(f"keff = {k}")
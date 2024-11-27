import numpy as np
import matplotlib.pyplot as plt

#Define necessaary constants
N = 51 #Gridsize
del_x = 1.0 / (N-1)
del_y = del_x
Re = 100.0 #Given
U_lid = 1.0
L = 1.0
Mu = (U_lid * L) / Re
rho = 1.164 #density of air at 30deg.C

# Function for thomas algorithm
def tdma_solve(a,b,c,d):
    n = len(a)
    bprime = b.copy()
    dprime = d.copy()
    for i in range(1,n+1):
        factor = a[i-1] / bprime[i-1]
        bprime[i] -= factor*c[i-1]
        dprime[i] -= factor*dprime[i-1]
    
    nx = len(d)
    x = np.zeros(nx)
    x[-1] = dprime[-1] / bprime[-1]
    for i in range(nx-2,-1,-1):
        x[i] = (dprime[i] - c[i]*x[i+1]) / bprime[i]
    return x

#Initialize the variables u,v,P,vorticity(W),Streamfunction(Psi)
W_old = np.zeros([N,N])
W_new = W_old.copy()
P_old = np.zeros([N,N])
P_new = P_old.copy()
Psi_old = np.zeros([N,N])
Psi_new = Psi_old.copy()
U = np.zeros([N,N])
V = np.zeros([N,N])
U[-1, :] = 1.0  #lid velocity

error = 1.0 #to get into the loop
while (error >= 1e-3):  #iterate till converged
    #Solve vorticity-transport equation:
    for i in range(1,N-1):
        for j in range(1,N-1):
            #Using first order upwind scheme:
            if(U[i][j] >= 0 and V[i][j] >= 0):
                W_new[i][j] = ((Mu/del_x)*(W_old[i][j+1] + W_new[i][j-1] + W_old[i+1][j] + W_new[i-1][j])
                                + U[i][j]*W_new[i][j-1] + V[i][j]*W_new[i-1][j])/(U[i][j] + V[i][j] + (4*Mu/del_x))
                continue
            elif(U[i][j] < 0 and V[i][j] >= 0):
                W_new[i][j] = ((Mu/del_x)*(W_old[i][j+1] + W_new[i][j-1] + W_old[i+1][j] + W_new[i-1][j]) 
                               - U[i][j]*W_new[i][j+1] + V[i][j]*W_new[i-1][j])/(V[i][j] - U[i][j] + (4*Mu/del_x))
            elif(U[i][j] >= 0 and V[i][j] < 0):
                W_new[i][j] = ((Mu/del_x)*(W_old[i][j+1] + W_new[i][j-1] + W_old[i+1][j] + W_new[i-1][j]) 
                               + U[i][j]*W_new[i][j-1] - V[i][j]*W_new[i+1][j])/(U[i][j] - V[i][j] + (4*Mu/del_x))
            elif(U[i][j] < 0 and V[i][j] < 0):
                W_new[i][j] = ((Mu/del_x)*(W_old[i][j+1] + W_new[i][j-1] + W_old[i+1][j] + W_new[i-1][j]) 
                               - U[i][j]*W_new[i][j+1] - V[i][j]*W_new[i+1][j])/((4*Mu/del_x) -U[i][j] - V[i][j])

    #Solve Poisson equation
    omega = 1.11 #Over relaxation parameter
    for i in range(1,N-1):
        a = np.ones(N-3) * 1.0 * omega
        b = np.ones(N-2) * -4.0
        c = np.ones(N-3) * 1.0 * omega
        d = np.zeros(N-2)
        for j in range(1,N-1):
            d[j-1] = -4.0*(1-omega)*Psi_old[i][j] - omega*(Psi_old[i+1][j] + Psi_new[i-1][j] + W_new[i][j]*(del_x**2))
        new_values = tdma_solve(a,b,c,d)
        Psi_new[i, 1:N-1] = new_values

    #Find U and V using Psi:
    for i in range(1,N-1):
        for j in range(1,N-1):
            U[i][j] = (Psi_new[i+1][j] - Psi_new[i-1][j])/(2*del_y)
            V[i][j] = (Psi_new[i][j-1] - Psi_new[i][j+1])/(2*del_x)

    #Implement Boundary Conditions for Vorticity
    W_new[:, 0] = -2.0 * (Psi_new[:,1] - Psi_new[:, 0]) / (del_x**2) #Left wall BC
    W_new[:, -1] = 2.0 * (Psi_new[:,-1] - Psi_new[:, -2]) / (del_x**2) #Right wall BC
    W_new[0, :] = -2.0 * (Psi_new[1, :] - Psi_new[0,:]) / (del_y**2) #Bottom wall BC
    W_new[-1, :] = (2.0/(del_y**2)) * (Psi_new[-1,:] - Psi_new[-2,:] - U_lid*del_y) # Top lid BC

    #Solve the pressure poisson equation
    for i in range(1,N-1):
        for j in range(1,N-1):
            P_new[i][j] = 0.25*(P_old[i][j+1] + P_old[i+1][j] + P_new[i-1][j] + P_new[i][j-1]) 
            - (rho/(2*del_x*del_x))*((Psi_new[i][j+1] - 2*Psi_new[i][j] + Psi_new[i][j-1])*(Psi_new[i+1][j] -2*Psi_new[i][j] 
            + Psi_new[i-1][j]) - (Psi_new[i+1][j+1] - Psi_new[i+1][j-1] - Psi_new[i-1][j+1] + Psi_new[i-1][j-1])**2)

    #Find value of convergence:
    error = 0
    for i in range(1,N-1):
        for j in range(1,N-1):
            error += abs(Psi_new[i][j] - Psi_old[i][j])
    
    if(error == 0.0): #To bypass the initial error = 0.0 value that we get from initilizing all the variables to zero.
        error = 1.0
    
    W_old = W_new.copy()
    Psi_old = Psi_new.copy()
    P_old = P_new.copy()

#Verify if the continuity equation is satisfied
continuity = 0
for i in range(1,N-1):
    for j in range(1,N-1):
        continuity += abs((U[i][j+1] - U[i][j-1])/(2*del_x) + (V[i+1][j] - V[i-1][j])/(2*del_y))
print('Continuity error: ',continuity)
#Post-Processing
U_var = []
V_var = []
Grid = []
half = (N-1)//2
for i in range(N):
    U_var.append(U[i][half])
    V_var.append(V[half][i])
    Grid.append(i*del_x)

plt.figure(figsize=(11,6))
plt.contourf(Psi_new)
plt.colorbar()
plt.title('Streamlines (Re = 100)')
plt.show()

plt.figure(figsize=(11,6))
plt.contourf(W_new)
plt.colorbar()
plt.title('Vorticity (Re = 100)')

plt.show()
plt.figure(figsize=(11,6))
plt.contourf(P_new)
plt.colorbar()
plt.title('Pressure (Re = 100)')
plt.show()

plt.figure(figsize=(11,6))
contour_plot_u = plt.contourf(Grid, Grid, U, cmap='jet', levels=100)
plt.colorbar(contour_plot_u, label='u Velocity')
contour_lines_u = plt.contour(Grid, Grid, U, colors='black', levels=15)
plt.clabel(contour_lines_u, inline=1, fontsize=8)
plt.title("Contour Plot for u (Re = 100)")
plt.xlabel("x---->")
plt.ylabel("y---->")

plt.figure(figsize=(11,6))
plt.plot(U_var,Grid)
plt.xlabel('U velocity ---->')
plt.ylabel('y---->')
plt.title(f'u velocity values at x = 0.5 (Re = 100, Grid: {N}x{N})')
# Data for U-velocity from ghia et. al
y_u = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000, 
                0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0000])
u_velocity = np.array([1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.20581, 
                       -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192, -0.03717, 0.00000])
plt.plot(u_velocity,y_u, marker='o')
plt.legend([f'My simulation (Grid: {N}x{N})','Ghia et. al (Grid: 129x129)'])
plt.show()

plt.figure(figsize=(11,6))
contour_plot_v = plt.contourf(Grid, Grid, V, cmap='jet', levels=100)
plt.colorbar(contour_plot_u, label='v Velocity')
contour_lines_v = plt.contour(Grid, Grid, V, colors='black', levels=15)
plt.clabel(contour_lines_v, inline=1, fontsize=8)
plt.title("Contour Plot for v (Re = 100)")
plt.xlabel("x---->")
plt.ylabel("y---->")

plt.figure(figsize=(11,6))
plt.plot(Grid,V_var)
plt.xlabel('x---->')
plt.ylabel('V velocity ---->')
plt.title(f'v velocity values at y = 0.5 (Re = 100, Grid: {N}x{N})')
# Data for V-velocity from Ghia et. al
x_v = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000, 
                0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000])
v_velocity = np.array([0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, -0.24533, 
                       0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 0.10091, 0.09233, 0.00000])
plt.plot(x_v,v_velocity, marker='o')
plt.legend([f'My simulation (Grid: {N}x{N})','Ghia et. al (Grid: 129x129)'])
plt.show()
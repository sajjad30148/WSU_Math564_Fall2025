def ToroidalPeriods(x):
    
    '''
    TR is a python package that contains the functions necessary for computing
    the toroidal mode resonant periods of the earth's mantle/crust.  The 
    relevant experimental data and earth model are built in to these functions.
    The user need only supply the 32 PREM model parameters to this function.
    Author: Tom Asaki
    Version: November 2025
    
    INPUTS:
        
       x    Earth model parameters that specify densities and 
            shear moduli.  x is a vector of length 32.
                   
    OUTPUTS:
    
       Tc   list of computed toroidal mode Periods (sec)
       Te   list of experimental toroidal mode Periods (sec)
    '''

    import numpy as np
    
    ##### set various internal parameters #####
    
    a = 6371  # earth radius (km)
    b = 3480  # core radius (km)
    NumSteps = 700
    MinNumSteps= 10
    Lradii = [ a , 6356 , 6346.6 , 6151 , 5971 , 5771 , 5701 , 5600 , 3630 , b ]
    
    # compute the radii for computations, including half steps                           
    fL=-np.diff(Lradii)/(a-b);
    r=np.copy(a);
    for k in range(len(Lradii)-1):
        nstps=max(MinNumSteps,int(np.ceil(fL[k]*NumSteps)))
        ds=(Lradii[k+1]-Lradii[k])/nstps
        r=np.append(r,np.linspace(Lradii[k]+ds,Lradii[k+1],nstps))
    
    # Preliminary Computations
    midr=np.append((r[1:]+r[:-1])/2,[-1])
    temp=np.vstack((r,midr))
    r=np.squeeze(np.reshape(temp,(1,-1),order='F'))
    r=r[:-1]
    rcm=r*100000
    z=r/a
    nr=len(z)
    
   
    # Compute rho values at each radius
    rho=np.zeros((nr,))
    for k in range(nr):
        rr=r[k]
        zz=z[k]
        if rr >= 6368:
            rho[k]=2.6
        elif rr >= 6356:
            rho[k]=2.6
        elif rr >= 6346.6:
            rho[k]=2.9
        elif rr >= 6151.0:
            rho[k]=x[0]*zz+x[1]
        elif rr >= 5791.0: 
            rho[k]=x[2]*zz+x[3]
        elif rr >= 5771.0: 
            rho[k]=x[4]*zz+x[5]
        elif rr >= 5701.0:
            rho[k]=x[6]*zz+x[7]
        elif rr >= 3480.0:
            rho[k]=x[8]*zz**3+x[9]*zz**2+x[10]*zz+x[11]
        else:
            rho[k]=np.nan
    
   
    # Compute mu values at each radius
    vs=np.zeros((nr,))
    for k in range(nr):
        rr=r[k]
        zz=z[k]
        if rr >= 6368:
            vs[k]=3.2
        elif rr >= 6356:
            vs[k]=3.2
        elif rr >= 6346.6:
            vs[k]=3.9
        elif rr >= 6151.0:
            vs[k]=x[12]*zz+x[13]
        elif rr >= 5791.0:
            vs[k]=x[14]*zz+x[15]
        elif rr >= 5771.0:
            vs[k]=x[16]*zz+x[17]
        elif rr >= 5701.0:
            vs[k]=x[18]*zz+x[19]
        elif rr >= 5600.0:
            vs[k]=x[20]*zz**3+x[21]*zz**2+x[22]*zz+x[23]
        elif rr >= 3630.0:
            vs[k]=x[24]*zz**3+x[25]*zz**2+x[26]*zz+x[27]
        elif rr >= 3480.0:
            vs[k]=x[28]*zz**3+x[29]*zz**2+x[30]*zz+x[31]
        else:
            vs[k]=np.nan
        
    vs=vs*100000         # conversion from km/s to cm/s
    mu=rho*(vs**2)         # cgs units
       
    # if bad values of rho or mu are provided, then stop
    if np.any(rho<=0) or np.any(vs<=0):
        return np.nan,np.nan
    
    # set frequency data
    data = np.array([[  2 , 0 , 2636.38 ] ,
                     [  3 , 0 , 1705.95 ] ,
                     [  4 , 0 , 1305.92 ] , 
                     [  5 , 0 , 1075.98 ] ,
                     [  6 , 0 ,  925.84 ] ,
                     [  7 , 0 ,  819.31 ] ,
                     [  8 , 0 ,  736.86 ] ,
                     [  9 , 0 ,  671.80 ] ,
                     [ 10 , 0 ,  618.97 ] ,
                     [ 12 , 0 ,  538.05 ] ,
                     [ 13 , 0 ,  506.07 ] ,
                     [ 14 , 0 ,  477.53 ] ,
                     [ 16 , 0 ,  430.01 ] ,
                     [ 17 , 0 ,  410.24 ] ,
                     [ 18 , 0 ,  391.82 ] ,
                     [ 20 , 0 ,  360.03 ] ,
                     [ 21 , 0 ,  346.50 ] ,
                     [ 22 , 0 ,  333.69 ] ,
                     [ 23 , 0 ,  321.70 ] ,
                     [ 24 , 0 ,  310.63 ] ,
                     [ 25 , 0 ,  300.37 ] ,
                     [ 26 , 0 ,  290.77 ] ,
                     [ 27 , 0 ,  281.75 ] ,
                     [ 28 , 0 ,  273.27 ] ,
                     [ 29 , 0 ,  265.30 ] ,
                     [ 30 , 0 ,  257.76 ] ,
                     [ 31 , 0 ,  250.66 ] ,
                     [ 32 , 0 ,  243.95 ] ,
                     [ 33 , 0 ,  237.59 ] ,
                     [ 34 , 0 ,  231.56 ] ,
                     [ 35 , 0 ,  225.83 ] ,
                     [ 36 , 0 ,  220.37 ] ,
                     [ 37 , 0 ,  215.17 ] ,
                     [ 38 , 0 ,  210.21 ] ,
                     [ 39 , 0 ,  205.47 ] ,
                     [ 40 , 0 ,  200.95 ] ,
                     [ 41 , 0 ,  196.60 ] ,
                     [ 42 , 0 ,  192.50 ] ,
                     [ 43 , 0 ,  188.51 ] ,
                     [ 44 , 0 ,  184.70 ] ,
                     [ 45 , 0 ,  181.04 ] ,
                     [ 46 , 0 ,  177.52 ] ,
                     [ 47 , 0 ,  174.10 ] ,
                     [ 48 , 0 ,  170.87 ] ,
                     [ 49 , 0 ,  167.73 ] ,
                     [ 50 , 0 ,  164.70 ] ,
                     [ 51 , 0 ,  161.78 ] ,
                     [ 52 , 0 ,  158.95 ] ,
                     [ 53 , 0 ,  156.23 ] ,
                     [ 54 , 0 ,  153.59 ] ,
                     [ 55 , 0 ,  151.04 ] ,
                     [  2 , 1 ,  756.57 ] ,
                     [  3 , 1 ,  695.18 ] ,
                     [  6 , 1 ,  519.09 ] ,
                     [  7 , 1 ,  475.17 ] ,
                     [  8 , 1 ,  438.49 ] ,
                     [  9 , 1 ,  407.74 ] ,
                     [ 10 , 1 ,  381.65 ] ,
                     [ 11 , 1 ,  359.13 ] ,
                     [ 12 , 1 ,  339.54 ] ,
                     [ 13 , 1 ,  322.84 ] ,
                     [ 15 , 1 ,  293.35 ] ,
                     [ 16 , 1 ,  280.56 ] ,
                     [ 17 , 1 ,  269.51 ] ,
                     [ 18 , 1 ,  259.00 ] ,
                     [ 19 , 1 ,  249.41 ] ,
                     [ 20 , 1 ,  240.88 ] ,
                     [ 21 , 1 ,  232.53 ] ,
                     [ 22 , 1 ,  225.22 ] ,
                     [ 23 , 1 ,  218.31 ] ,
                     [ 24 , 1 ,  211.91 ] ,
                     [ 25 , 1 ,  205.80 ] ,
                     [ 26 , 1 ,  200.24 ] ,
                     [ 27 , 1 ,  194.83 ] ,
                     [ 28 , 1 ,  189.94 ] ,
                     [ 29 , 1 ,  185.26 ] ,
                     [ 30 , 1 ,  180.80 ] ,
                     [ 31 , 1 ,  176.85 ] ,
                     [ 32 , 1 ,  172.98 ] ,
                     [ 33 , 1 ,  169.22 ] ,
                     [ 34 , 1 ,  165.72 ] ,
                     [ 35 , 1 ,  162.34 ] ,
                     [ 36 , 1 ,  159.09 ] ,
                     [ 37 , 1 ,  156.03 ] ,
                     [ 38 , 1 ,  153.13 ] ,
                     [ 39 , 1 ,  150.26 ] ,
                     [  4 , 2 ,  420.46 ] ,
                     [  7 , 2 ,  363.65 ] ,
                     [  8 , 2 ,  343.34 ] ,
                     [ 17 , 2 ,  219.95 ] ,
                     [ 18 , 2 ,  211.90 ] ,
                     [ 19 , 2 ,  204.63 ] ,
                     [ 21 , 2 ,  191.91 ] ,
                     [ 22 , 2 ,  186.19 ] ,
                     [ 25 , 2 ,  171.12 ] ,
                     [ 26 , 2 ,  166.50 ] ,
                     [ 28 , 2 ,  158.42 ] ,
                     [ 29 , 2 ,  154.64 ] ,
                     [  9 , 3 ,  259.26 ] ,
                     [ 11 , 3 ,  240.49 ] ,
                     [ 18 , 3 ,  184.09 ] ,
                     [ 19 , 3 ,  178.13 ] ,
                     [ 20 , 3 ,  172.74 ] ,
                     [ 21 , 3 ,  167.69 ] ,
                     [ 24 , 3 ,  154.67 ] ,
                     [ 11 , 4 ,  199.74 ] ,
                     [ 20 , 4 ,  155.64 ] ,
                     [ 21 , 4 ,  151.15 ] ] )
    
    N = data[:,0]
    Te = data[:,2]
    numf = len(N)
    
    # Main frequency search routine
    # Integrate Alterman equations with intial conditions y(a)=[1,0] to find
    # w values for which y(b)=[~,0].  Step coarsely through w to find sign
    # changes in the shear stress computed at b.  Then search finely to locate
    # w for which the shear stress at b is approximately zero.
    
    Tc = np.zeros((numf,))
    
    for k in range(numf):
    
        NP=N[k]**2+N[k]-2
        w0=(2*np.pi)/Te[k]
        
        wlo=w0*0.95
        y=RK4I(NP,wlo,rcm,rho,mu)
        ylo=y[1,0]
    
        whi=w0*1.05
        y=RK4I(NP,whi,rcm,rho,mu)
        yhi=y[1,0]
        
        [wlo,whi]=refine(wlo,whi,ylo,yhi,NP,rcm,rho,mu)
        Tc[k]=(4*np.pi)/(whi+wlo)
            
    return Tc,Te

#################################################################

def RK4I(NP,w,rcm,rho,mu):
    import numpy as np
    y=np.array([[1],[0]])
    for k in range(0,len(rcm)-2,2): 
        h=rcm[k+2]-rcm[k]
        k1=alterman(rcm[k],y,NP,w,rho[k],mu[k])
        k2=alterman(rcm[k+1],y+(h/2)*k1,NP,w,rho[k+1],mu[k+1])
        k3=alterman(rcm[k+1],y+(h/2)*k2,NP,w,rho[k+1],mu[k+1])
        k4=alterman(rcm[k+2],y+h*k3,NP,w,rho[k+2],mu[k+2])
        y=y+(h/6)*(k1+2*k2+2*k3+k4)
    return y

#################################################################

def alterman(r,y,NP,w,rho,mu):
    import numpy as np
    A=np.array([ [ 1/r , 1/mu ] , [ NP*mu/r**2-rho*w**2 , -3/r ] ])
    return A@y

################################################################

def refine(wlo,whi,ylo,yhi,NP,rcm,rho,mu):
    
    import numpy as np
    
    #wtol=((whi+wlo)**2/(8*np.pi))*0.01
    wtol=wlo*0.00001
    
    # compute bisection result as the initial third point
    wm=(whi+wlo)/2
    y=RK4I(NP,wm,rcm,rho,mu)
    ym=y[1,0]
    
    ww=[wlo,whi,wm]
    yy=[ylo,yhi,ym]
    
    while np.abs(ww[-1]-ww[-3])>wtol :
    
        # find the quadratic interpolation point wz
        h0=ww[-2]-ww[-3]
        d0=(yy[-2]-yy[-3])/h0
        h1=ww[-1]-ww[-2]
        d1=(yy[-1]-yy[-2])/h1
        a=(d1-d0)/(h0+h1)
        b=a*h1+d1
        c=yy[-1]
        wz=ww[-1]-(2*c)/(b+np.sign(b)*np.sqrt(b**2-4*a*c))
        ww.append(wz)
    
        # compute the function value at wz
        y=RK4I(NP,wz,rcm,rho,mu)
        newy=np.copy(y)
        yy.append(newy[1,0])
            
    return ww[-2:]

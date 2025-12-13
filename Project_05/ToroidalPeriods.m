function [Tc,Te]=ToroidalPeriods(x)

% Function Toroidal Periods computes resonant mantle-crust toroidal
% periods of the earth given density and elasticity data at fixed 
% radii.  This verion is a simplified and relatively inefficient 
% computation for use with MATH 564.
% Author: Tom Asaki
% Version: November 2025
%
% USAGE:
%
%   [Tc,Te]=ToroidalPeriods(x)
%
% INPUTS:
%
%   x       Earth model parameters that specify densities and 
%           shear moduli.  x is a vector of length 32.
%               
% OUTPUTS:
%
%   Tc   list of computed toroidal mode periods (sec)
%   Te   list of experimental toroidal mode periods (sec)
%

%% set various internal parameters

          a  = 6371       ; % earth radius (km)
          b  = 3480       ; % core radius (km)
   NumSteps  = 700        ; % number of radial integration steps
MinNumSteps  = 10         ; % minimum steps per earth layer 
Lradii       = [ a 6356 6346.6 6151 5971 5771 5701 5600 3630 b ];

% compute the radii for computations, including half steps                           
fL=-diff(Lradii)/(a-b);
r=a;
for k=1:length(Lradii)-1
    nstps=max(MinNumSteps,ceil(fL(k)*NumSteps));
    ds=(Lradii(k+1)-Lradii(k))/nstps;
    r=[r linspace(Lradii(k)+ds,Lradii(k+1),nstps)];     %#ok
end

%% Preliminary Computations

% Add midpoint values to the radial list, and construct relative radii.
% Also create radial values in units of cm instead of km.
midr=[(r(2:end)+r(1:end-1))/2,-1];
r=reshape([r;midr],[],1);
r(end)=[];
rcm=r*100000;
z=r./a;
nr=length(z);

% Compute rho values at each radius (PREM model)
rho=zeros(nr,1);
for k=1:nr
    rr=r(k);
    zz=z(k);
    if     rr >= 6368,   rho(k)=2.6;
    elseif rr >= 6356,   rho(k)=2.6;
    elseif rr >= 6346.6, rho(k)=2.9;
    elseif rr >= 6151.0, rho(k)=x(1)*zz+x(2);
    elseif rr >= 5791.0, rho(k)=x(3)*zz+x(4);
    elseif rr >= 5771.0, rho(k)=x(5)*zz+x(6);
    elseif rr >= 5701.0, rho(k)=x(7)*zz+x(8);
    elseif rr >= 3480.0, rho(k)=x(9)*zz^3+x(10)*zz^2+x(11)*zz+x(12);
    else, rho(k)=NaN;
    end
end

% Compute mu values at each radius (PREM model)
vs=zeros(nr,1);
for k=1:nr
    rr=r(k);
    zz=z(k);
    if     rr >= 6368,   vs(k)=3.2;
    elseif rr >= 6356,   vs(k)=3.2;
    elseif rr >= 6346.6, vs(k)=3.9;
    elseif rr >= 6151.0, vs(k)=x(13)*zz+x(14);
    elseif rr >= 5791.0, vs(k)=x(15)*zz+x(16);
    elseif rr >= 5771.0, vs(k)=x(17)*zz+x(18);
    elseif rr >= 5701.0, vs(k)=x(19)*zz+x(20);
    elseif rr >= 5600.0, vs(k)=x(21)*zz^3+x(22)*zz^2+x(23)*zz+x(24);
    elseif rr >= 3630.0, vs(k)=x(25)*zz^3+x(26)*zz^2+x(27)*zz+x(28);
    elseif rr >= 3480.0, vs(k)=x(29)*zz^3+x(30)*zz^2+x(31)*zz+x(32);
    else, vs(k)=NaN;
    end
end
vs=vs*100000;         % conversion from km/s to cm/s
mu=rho.*vs.^2;        % cgs units

% if bad values of rho or mu are provided, then stop
if any(rho<=0) || any(vs<=0)
    Tc=[];Te=[];
    return
end

% declare the experimental data
     data = [ 2   0   2636.38
              3   0   1705.95
              4   0   1305.92 
              5   0   1075.98
              6   0    925.84
              7   0    819.31
              8   0    736.86
              9   0    671.80
             10   0    618.97
             12   0    538.05
             13   0    506.07
             14   0    477.53
             16   0    430.01
             17   0    410.24
             18   0    391.82
             20   0    360.03
             21   0    346.50
             22   0    333.69
             23   0    321.70
             24   0    310.63
             25   0    300.37
             26   0    290.77
             27   0    281.75
             28   0    273.27
             29   0    265.30
             30   0    257.76
             31   0    250.66
             32   0    243.95
             33   0    237.59
             34   0    231.56
             35   0    225.83
             36   0    220.37
             37   0    215.17
             38   0    210.21
             39   0    205.47
             40   0    200.95
             41   0    196.60
             42   0    192.50
             43   0    188.51
             44   0    184.70
             45   0    181.04
             46   0    177.52
             47   0    174.10
             48   0    170.87
             49   0    167.73
             50   0    164.70
             51   0    161.78
             52   0    158.95
             53   0    156.23
             54   0    153.59
             55   0    151.04
              2   1    756.57
              3   1    695.18
              6   1    519.09
              7   1    475.17
              8   1    438.49
              9   1    407.74
             10   1    381.65
             11   1    359.13
             12   1    339.54
             13   1    322.84
             15   1    293.35
             16   1    280.56
             17   1    269.51
             18   1    259.00
             19   1    249.41
             20   1    240.88
             21   1    232.53
             22   1    225.22
             23   1    218.31
             24   1    211.91
             25   1    205.80
             26   1    200.24
             27   1    194.83
             28   1    189.94
             29   1    185.26
             30   1    180.80
             31   1    176.85
             32   1    172.98
             33   1    169.22
             34   1    165.72
             35   1    162.34
             36   1    159.09
             37   1    156.03
             38   1    153.13
             39   1    150.26
              4   2    420.46
              7   2    363.65
              8   2    343.34
             17   2    219.95
             18   2    211.90
             19   2    204.63
             21   2    191.91
             22   2    186.19
             25   2    171.12
             26   2    166.50
             28   2    158.42
             29   2    154.64
              9   3    259.26
             11   3    240.49
             18   3    184.09
             19   3    178.13
             20   3    172.74
             21   3    167.69
             24   3    154.67
             11   4    199.74
             20   4    155.64
             21   4    151.15] ;

 N = data(:,1);     % mode number
 m = data(:,2);     %#ok  overtone number (not directly needed!)
Te = data(:,3);     % experimental periods for each (N,m) pair

%% Main frequency search routine
% Integrate Alterman equations with intial conditions y(a)=[1,0] to find
% w values for which y(b)=[~,0].  Step coarsely through w to find sign
% changes in the shear stress computed at b.  Then search finely to locate
% w for which the shear stress at b is approximately zero.

Tc = zeros(size(Te));

for k=1:length(N)

    NP=N(k)^2+N(k)-2;
    w0=(2*pi)/Te(k);

    wlo=w0*(0.95);
    y=RK4I([1;0],NP,wlo,rcm,rho,mu);
    ylo=y(2);

    whi=w0*(1.05);
    y=RK4I([1;0],NP,whi,rcm,rho,mu);
    yhi=y(2);

    [wlo,whi]=refine(wlo,whi,ylo,yhi,NP,rcm,rho,mu);
    Tc(k)=(4*pi)/(whi+wlo);

end


return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y=RK4I(y,NP,w,rcm,rho,mu)
for k=1:2:length(rcm)-2
    h=rcm(k+2)-rcm(k);
    k1=alterman(rcm(k),y,NP,w,rho(k),mu(k));
    k2=alterman(rcm(k+1),y+(h/2)*k1,NP,w,rho(k+1),mu(k+1));
    k3=alterman(rcm(k+1),y+(h/2)*k2,NP,w,rho(k+1),mu(k+1));
    k4=alterman(rcm(k+2),y+h*k3,NP,w,rho(k+2),mu(k+2));
    y=y+(h/6)*(k1+2*k2+2*k3+k4);
end
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y=alterman(r,y,NP,w,rho,mu)
    y=[y(1)/r+y(2)/mu ; (NP*(mu/r^2)-w^2*rho)*y(1)-3*y(2)/r ];
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [wlo,whi]=refine(wlo,whi,ylo,yhi,NP,rcm,rho,mu)

% The refining steps use a quadratic model of the function to
% estimate the zero crossing (Muller Method). Typically this method
% converges in 3 to 5 iterations for this problem.

%wtol=((whi+wlo)^2/8/pi)*(0.01);
wtol=wlo*0.00001;

% compute bisection result as the initial third point
wm=(whi+wlo)/2;
y=RK4I([1;0],NP,wm,rcm,rho,mu);
ym=y(2);

ww=[wlo whi wm];
yy=[ylo yhi ym];

while abs(ww(end)-ww(end-2))>wtol 

    % find the quadratic interpolation point wz
    h0=ww(end-1)-ww(end-2);  
    d0=(yy(end-1)-yy(end-2))/h0;
    h1=ww(end)-ww(end-1);  
    d1=(yy(end)-yy(end-1))/h1;
    a=(d1-d0)/(h0+h1);
    b=a*h1+d1;
    c=yy(end);
    wz=ww(end)-(2*c)/(b+sign(b)*sqrt(b^2-4*a*c));
    ww=[ww wz];   %#ok

    % compute the function value at wz
    y=RK4I([1;0],NP,wz,rcm,rho,mu);
    yz=y(2);
    yy=[yy yz];   %#ok

end

whi=ww(end);
wlo=ww(end-1);

return

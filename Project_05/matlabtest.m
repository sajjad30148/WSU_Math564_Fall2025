

% Set an initial decision variable vector

x=[ 0.6 ,  2.6 ,  -3.6 ,  7.0 , -7.0 , 11.2 , -1.6 ,  5.0 , ...,
   -3.0 ,  5.6 ,  -6.4 ,  8.0 ,  5.6 , -1.0 , -4.4 ,  8.8 , ...
  -18.6 , 22.2 ,  -4.8 , 10.0 ,  0.8 , -2.0 ,-17.2 , 22.4 , ...
   -9.2 , 17.2 , -14.0 , 11.4 ,  1.0 , -2.2 ,  1.4 ,  6.4   ];

% Call the function to compute the resonant periods

[Tc,Te]=ToroidalPeriods(x);

% Compute an objective function (this is an example).

if isempty(Tc)
    f = inf;
else
    f = norm(Tc-Te)/norm(Te);
end
fprintf('\n\n  Objective Value = %7.5f\n\n',f)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function setup boundary conditions for the forward and the adjoint 
% diffusion problems.
%
% The forward diffusion model:
%
% -\nabla\cdot\gamma\nabla u + \sigma u = S  in \Omega
% \bnu\cdot\gamma \nabla u+\kappa u = f, on \partial\Omega
%
% with S=0, f given by the boundary source
%
% The adjoint diffusion model is:
%
% -\nabla\cdot\gamma\nabla w + \sigma w = S  in \Omega
% \bnu\cdot\gamma\nabla w +kappa w=f, on \partial\Omega
%
% with S=-(\Gamma\sigma u-H)\Gamma\sigma, f=0
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [qmatrix,gmatrix,hmatrix,rmatrix] = ...
         PATDiffBC(Type,SrcInfo,BdaryInfo,ks,p,e,u,time,kappa)

ne = size(e,2); % number of edges on the domain boundary

% This two lines indicate that the BC is NOT Dirichlet
hmatrix = zeros(1,2*ne);
rmatrix = zeros(1,2*ne);

% The following two lines set the BC to Robin type with the q-matrix given
% by the constant value kappa. The g-matrix needs to be setup later
qmatrix = kappa*ones(1,ne);
gmatrix = zeros(1,ne);

% Set the gmatrix
if strcmp(Type,'Adjoint') %
    for k = 1:ne
        gmatrix(k)=0.0;
    end
elseif strcmp(Type,'Forward') % the sources for forward problem are Gaussians
    xs=SrcInfo(1,ks);
    ys=SrcInfo(2,ks);
    srcseg=SrcInfo(3,ks);
    for k = 1:ne
        x1 = p(1,e(1,k)); % x at first point in segment
        y1 = p(2,e(1,k)); % y at first point in segment
        x2 = p(1,e(2,k)); % x at second point in segment
        y2 = p(2,e(2,k)); % y at second point in segment
        xm = (x1 + x2)/2; % x at segment midpoint
        ym = (y1 + y2)/2; % y at segment midpoint 
        gmatrix(k)=1.0;
        if BdaryInfo(2,k)==srcseg % if the edge lives on the same side with the source
            gmatrix(k) = gmatrix(k)+5*exp(-((xm-xs)^2+(ym-ys)^2)/0.02); % sources are Gaussians
        end
    end
else
    disp('Must specific problem type (Forward or Adjoint) to fix BC!');
end
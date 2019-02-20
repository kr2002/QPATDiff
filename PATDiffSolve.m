%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function solves the the forward and adjoint diffusion 
% equations with a P_1 finite element method
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function u=PATDiffSolve(Type,SrcInfo,BdaryInfo,ks,P,E,T,diff,sigma,kappa,S)

% interpolation to triangle middle point
diffm=pdeintrp(P,T,diff);
sigmam=pdeintrp(P,T,sigma);
Sm=pdeintrp(P,T,S);

% construct mass matrices
[K,M,F]=assema(P,T,diffm,sigmam,Sm);

% construct boundary conditions
pdebound =@(p,e,u,time)PATDiffBC(Type,SrcInfo,BdaryInfo,ks,p,e,[],[],kappa);
[Q,G,H,R] = assemb(pdebound,P,E);

% solve the PDE
u = assempde(K,M,F,Q,G,H,R);
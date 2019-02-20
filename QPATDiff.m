%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% QPATDiff: Minimization-based reconstruction algorithm for quantitative 
%           PAT with the diffusion model
%
%           It is well-known that we can only recover uniquely at most two 
%           of the three related coefficients. This code provides an option 
%           to choose the coefficients to reconstruct.
%
%           The reconstruction algorithm is based on the least-square 
%           formulation of the inverse problem.
%
% Author:    Kui Ren
% Address:   Math and ICES, UT Austin
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The diffusion model:
%
% -\nabla\cdot\gamma \nabla u + \sigma u = 0  in \Omega
% \bnu\cdot\nabla u+ \kappa u = f,  on \partial \Omega
%
% The measurement quantity:
% 
% H= \Gamma \sigma u  on \partial\Omega
%
% The data:
%
% (f_j, H_j), 1\le j\le N_s
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The algorithm: minimize the functional:
%
% \Phi= 1/2*\sum_{j=1}^{N_s} \int_\Omega 
%       (\Gamma \sigma u_j-H_j)^2 dx + regularization
%
% Gradient of the objective functions is computed with the adjoint state 
% method. 
%
% The adjoint problems are:
%
%  -\nabla\cdot\gamma \nabla w_j + \sigma w_j 
%                         =-r_j\Gamma\sigma in \Omega
%  \bnu\cdot\gamma \nabla w_j +\kappa w_j = 0
%
%  where r_j:=(\Gamma \sigma u_j-H_j)  
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Note:
%
% To avoid having both \Gamma and \gamma, we replace
% \gamma with \diff for the diffusion coefficient in 
% the code.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Usage: 
%
% Step 1: Generate geometry using PDETOOL, and save 
%         the data.
% Step 2: Modifying the setup in boundary sources and 
%         related parameters.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;  clear all; 

tic;
tb=toc;

% Load geometry and boundary information
disp(' ');
disp(' ');
disp('Setting simulation geometry and parameters .......');
disp(' ');

load 'geo-2b2';

MaxIT=200;

Ns=36;

% Decide which parameter to be reconstructed
MinVar='Sigma'; % 'Gamma','Sigma','Diff','GS','GD','SD' or 'All'

% create a Cartesian grid for inversion
dx=0.025; x=0:dx:2;
dy=0.025; y=0:dy:2;
Nx=length(x);
Ny=length(y);
% [X,Y]=meshgrid(x,y);

% Generate regular finite element mesh on rectangular geometry
[P,E,T]=poimesh(geo,Nx-1,Ny-1); 

M=Nx*Ny; % total number of nodes in the mesh

% Setup information on sources and boundary edges
SrcInfo=SetSources(Ns);
BdaryInfo=SetBdaryInfo(P,E);

% Set true parameters for light propagation
%Gammat=zeros(M,1); Difft=zeros(M,1); sigmat=zeros(M,1); % true parameters

rec1=[0.5 0.8; 0.5 0.8]; 
rec2=[1.4 1.7; 1.4 1.7];
rec3=[1.0 1.8; 0.3 0.6];
circ1=[1.0 1.5 0.2];
circ2=[1.5 1.0 0.3];

Gammat=0.6*ones(M,1);
%Gammat=0.5+0.2*ind_circ(P,circ3)+0.3*ind_circ(P,circ4)+0.3*ind_circ(P,circ5);
%Difft=0.03-0.01*ind_circ(P,circ3)-0.02*ind_circ(P,circ4); 
Difft=0.2*ones(M,1);
%sigmat=0.2+0.1*cos(pi*P(1,:)'-pi).*cos(pi*P(2,:)'-pi);
%sigmat=0.2+0.2*ind_circ(P,circ1)+0.1*ind_circ(P,circ2)+0.2*ind_circ(P,circ5)+0.2*ind_circ(P,circ6);
sigmat=0.2+0.1*ind_rec(P,rec1)+0.1*ind_rec(P,rec2)+0.2*ind_rec(P,rec3);

% interpolate to Cartesian grid
Gammatg=tri2grid(P,T,Gammat,x,y); % true value of sigma on Cartesian grid
sigmatg=tri2grid(P,T,sigmat,x,y); % true value of sigma on Cartesian grid
Difftg=tri2grid(P,T,Difft,x,y); % true value of sigma on Cartesian grid

if strcmp(MinVar,'Gamma')||strcmp(MinVar,'GS')||strcmp(MinVar,'GD')||strcmp(MinVar,'All')
    figure;
    pcolor(x,y,Gammatg); axis tight; colorbar('SouthOutside');
    axis square; axis off; shading interp;
    title('true \Gamma');
    drawnow;
end
if strcmp(MinVar,'Diff')||strcmp(MinVar,'GD')||strcmp(MinVar,'SD')||strcmp(MinVar,'All')
    figure; 
    pcolor(x,y,Difftg); axis tight; colorbar('SouthOutside');
    axis square;  axis off; shading interp;
    title('true \gamma');
    drawnow;
end
if strcmp(MinVar,'Sigma')||strcmp(MinVar,'GS')||strcmp(MinVar,'SD')||strcmp(MinVar,'All')
    figure;
    pcolor(x,y,sigmatg); axis tight; colorbar('SouthOutside');
    axis square;  axis off; shading interp;
    title('true \sigma');
    drawnow;
end

kappa=0.20; % the kappa in the boundary condition

disp('Finished setting simulation geometry and parameters .......');

% Generating synthetic data
disp(' ');
disp(' ');
disp('Generating synthetic data .......');
disp(' ');

noiselevel=0.0; % set noise level
srczero=zeros(M,1);
Hm=zeros(M,Ns);
for ks=1:Ns
    
    % Solve the diffusion equation
    ut=PATDiffSolve('Forward',SrcInfo,BdaryInfo,ks,P,E,T,Difft,sigmat,kappa,srczero);

    Ht=Gammat.*sigmat.*ut;
    
    %utg=tri2grid(P,T,Ht,x,y);
    %figure;
    %pcolor(x,y,utg); axis tight; colorbar('SouthOutside');
    %axis square; axis off; shading interp;
    %drawnow;
    %pause
    
    % Add noise to data
	Hm(:,ks)=Ht.*(1+noiselevel*2*(rand(M,1)-0.5));
    
    disp(['Synthetic data generated for source #: ' num2str(ks)]);
    disp('  ');

    clear ut Ht;
end
disp('Finished generating synthetic data .......');

% Setup initial guess
disp(' ');
disp(' ');
disp('Setting initial guess .......');
disp(' ');

Gamma0=0.5*ones(M,1);
if ~strcmp(MinVar,'Gamma') && ~strcmp(MinVar,'GS')&& ~strcmp(MinVar,'GD') && ~strcmp(MinVar,'All')
    Gamma0=Gammat;
end
Gamma0g=tri2grid(P,T,Gamma0,x,y);

Diff0=0.02*ones(M,1);
if ~strcmp(MinVar,'Diff') && ~strcmp(MinVar,'GD')&& ~strcmp(MinVar,'SD') && ~strcmp(MinVar,'All')
    Diff0=Difft;
end    
Diff0g=tri2grid(P,T,Diff0,x,y);

sigma0=0.2*ones(M,1);
if ~strcmp(MinVar,'Sigma') && ~strcmp(MinVar,'GS')&& ~strcmp(MinVar,'SD') && ~strcmp(MinVar,'All')
    sigma0=sigmat;
end    
sigma0g=tri2grid(P,T,sigma0,x,y);

if strcmp(MinVar,'Gamma')||strcmp(MinVar,'GS')||strcmp(MinVar,'GD')||strcmp(MinVar,'All')
    figure;
    pcolor(x,y,Gamma0g); axis tight; colorbar('SouthOutside');
    axis square; axis off; shading interp;
    %caxis([0.05 0.25]);
    title('initial guess of \Gamma');
    drawnow;
end
if strcmp(MinVar,'Diff')||strcmp(MinVar,'GD')||strcmp(MinVar,'SD')||strcmp(MinVar,'All')
    figure;
    pcolor(x,y,Diff0g); axis tight; colorbar('SouthOutside');
    axis square; axis off; shading interp;
    %caxis([0.05 0.25]);
    title('initial guess of \gamma');
    drawnow;
end
if strcmp(MinVar,'Sigma')||strcmp(MinVar,'GS')||strcmp(MinVar,'SD')||strcmp(MinVar,'All')
    figure;
    pcolor(x,y,sigma0g); axis tight; colorbar('SouthOutside');
    axis square; axis off; shading interp;
    %caxis([0.05 0.25]);
    title('initial guess of \sigma');
    drawnow;
end

X0=[Gamma0' Diff0' sigma0']';

disp('Finished setting initial guess .......');

% This short part is only for debugging
%[f0 g0]=PATDiffObj(X0,MinVar,x,y,dx,dy,Nx,Ny,P,E,T,Ns,Hm,SrcInfo,BdaryInfo,kappa);
%g0g=tri2grid(P,T,g0(1:M),x,y);
%g0g=tri2grid(P,T,g0(M+1:2*M),x,y);
%g0g=tri2grid(P,T,g0(2*M+1:3*M),x,y);
%figure;
%pcolor(x,y,g0g); axis tight; colorbar('SouthOutside');
%axis square; shading interp;
%title('Gradient');
%drawnow;

OptimMethod='UNCON';

% Setup the minimization algorithm
disp(' ');
disp(' ');
disp('Minimizing objective function .......');
disp(' ');

f=@(X) PATDiffObj(X,MinVar,x,y,dx,dy,Nx,Ny,P,E,T,Ns,Hm,SrcInfo,BdaryInfo,kappa);

if strcmp(OptimMethod,'UNCON')
    options=optimoptions(@fminunc,'Algorithm','quasi-newton', ...
    'Display','iter-detailed','GradObj','on','TolFun',1e-12,...
    'MaxIter',MaxIT);
    [X,fval,exitflag,output,grad]=fminunc(f,X0,options);
else
    % Set inequality constraint
    Aieq=zeros(1,3*M);
    Bieq=0;
    % Set equality constraint
    Aeq=zeros(1,3*M);
    Beq=0;
    % Set upper and lower bounds
    LB=[0.4*ones(1,M) 0.005*ones(1,M) 0.05*ones(1,M)]';
    UB=[0.9*ones(1,M) 0.040*ones(1,M) 0.50*ones(1,M)]';

    options=optimoptions(@fmincon,'Algorithm','trust-region', ...
    'Display','iter-detailed','GradObj','on','TolFun',1e-12,...
    'MaxIter',MaxIT);
    %options=optimset('Display','iter-detailed','GradObj','on','TolFun',1e-12,'MaxIter',MaxIT);
    %options = optimset('algorithm','sqp','maxfunevals',5000,'maxiter',100);
    %options = optimset(options,'tolx',1e-9,'tolcon',1e-9,'tolfun',1e-6);
    %options = optimset(options,'GradObj','on','GradConstr','off');
    
    [X,fval,exitflag,output,lambda]=fmincon(f,X0,Aieq,Bieq,Aeq,Beq,LB,UB,[],options);
    %[X,fval,exitflag,output]=fmincon(f,X0,zeros(M,M),zeros(M,1),[],[],LB,UB);
end

disp(' ');
disp(' ');
disp('Finished minimizing objective function .......');

disp(' ');
disp(' ');
disp('Plotting final results .......');
disp(' ');

Gammar=X(1:M);
Diffr=X(M+1:2*M);
sigmar=X(2*M+1:3*M);
% Plot reconstruction results
if strcmp(MinVar,'Gamma')||strcmp(MinVar,'GS')||strcmp(MinVar,'GD')||strcmp(MinVar,'All')
    Gammarg=tri2grid(P,T,Gammar,x,y);
    figure;
    pcolor(x,y,Gammarg); axis tight; colorbar('SouthOutside');
    %caxis([0.40 1.10]);
    axis square; axis off; shading interp;
    title('recovered \Gamma');
    drawnow;
end
if strcmp(MinVar,'Diff')||strcmp(MinVar,'GD')||strcmp(MinVar,'SD')||strcmp(MinVar,'All')
    Diffrg=tri2grid(P,T,Diffr,x,y);
    figure;
    pcolor(x,y,Diffrg); axis tight; colorbar('SouthOutside');
    %caxis([0.01 0.03]);
    axis square; axis off; shading interp;
    title('recovered \gamma');
    drawnow;
end
if strcmp(MinVar,'Sigma')||strcmp(MinVar,'GS')||strcmp(MinVar,'SD')||strcmp(MinVar,'All')
    sigmarg=tri2grid(P,T,sigmar,x,y);
    figure;
    pcolor(x,y,sigmarg); axis tight; colorbar('SouthOutside');
    %caxis([0.10 0.50]);
    axis square; axis off; shading interp;
    title('recovered \sigma');
    drawnow;
end

disp('Finished plotting final results .......');

save Exp01-Info geo P E T SrcInfo BdaryInfo kappa Ns MinVar MaxIT ...
                  OptimMethod noiselevel dx dy -ASCII
save Exp01-Results Gammat Gamma0 Gammar Difft Diff0 Diffr ...
                  sigmat sigma0 sigmar -ASCII

te=toc;
disp(' ');
disp(' ');
disp(['The code run for: ' num2str(te-tb) ' seconds']);
disp(' ');
disp(' ');

% This last line is used to close MATLAB after the computation. It is 
% only used when runing the code in background.

%exit; % to exit MATLAB 
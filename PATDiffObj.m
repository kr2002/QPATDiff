%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function evaluate the objective function and its gradients with 
% respect to the optimization variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f g]=PATDiffObj(X,MinVar,x,y,dx,dy,Nx,Ny,P,E,T,...
    Ns,Hm,SrcInfo,BdaryInfo,kappa)

M=Nx*Ny; % total number of nodes in the mesh
ne = size(SrcInfo,2); % number of edges/nodes on the domain boundary

Gammac=X(1:M); % current value of Gamma
Diffc=X(M+1:2*M);% current value of Diff
sigmac=X(2*M+1:3*M); % current value of sigma

f=0.0;
g=zeros(3*M,1);
for ks=1:Ns
    
    Hc=zeros(M,1); % predicted data on measurement locations
    rz=zeros(M,1); % residual on measurement locations
    srczero=zeros(M,1); % zero volume source for forward diffusion 
 
    uc=PATDiffSolve('Forward',SrcInfo,BdaryInfo,ks,P,E,T,Diffc,sigmac,kappa,srczero);
    Hc=Gammac.*sigmac.*uc;
    
    %Hcg=tri2grid(P,T,Hc,x,y);
    %figure;
    %pcolor(x,y,Hcg); axis tight; colorbar('SouthOutside');
    %axis square; axis off; shading interp;
    %drawnow;
    
    HmL=Hm(:,ks);
    rz=(Hc-HmL)./HmL; % for normalized objective function
    %rz=(Hc-HmL); % for unnormalized objective function
    
    % the contribution to the objective function from source ks
    f=f+0.5*sum(rz.^2)*dx*dy;
    
    % the contribution to the gradient from source ks
    if nargout > 1         
        if strcmp(MinVar,'Gamma')|strcmp(MinVar,'GS')|strcmp(MinVar,'GD')|strcmp(MinVar,'All')
            % the gradient w.r.t Gamma
            g(1:M)=g(1:M)+rz.*sigmac.*uc./HmL*dx*dy; % for normalized objective function
            %g(1:M)=g(1:M)+rz.*sigmac.*uc*dx*dy; % for unnormalized objective function
        end       
        if ~strcmp(MinVar,'Gamma')
            % solve the adjoint diffusion equation
            srcadj=-rz.*Gammac.*sigmac./HmL; % for normalized objective function
            %srcadj=-rz.*Gammac.*sigmac; % for unnormalized objective function
            
            wc=PATDiffSolve('Adjoint',SrcInfo,BdaryInfo,ks,P,E,T,Diffc,sigmac,kappa,srcadj);
            
            %wcg=tri2grid(P,T,wc,x,y);
            %figure;
            %pcolor(x,y,wcg); axis tight; colorbar('SouthOutside');
            %axis square; axis off; shading interp;
            %drawnow;
            %pause;
    
            % the gradient w.r.t Diff            
            if strcmp(MinVar,'Diff')|strcmp(MinVar,'GD')|strcmp(MinVar,'SD')|strcmp(MinVar,'All')
                [ucx, ucy]=pdegrad(P,T,uc); [wcx, wcy]=pdegrad(P,T,wc);
                gugw=ucx.*wcx+ucy.*wcy;
                gugwg=pdeprtni(P,T,gugw);
                g(M+1:2*M)=g(M+1:2*M)+gugwg*dx*dy; % for both normalized and unnormalized obj
            end
            % the gradient w.r.t sigma
            if strcmp(MinVar,'Sigma')|strcmp(MinVar,'GS')|strcmp(MinVar,'SD')|strcmp(MinVar,'All')
                g(2*M+1:3*M)=g(2*M+1:3*M)+(rz.*Gammac./HmL+wc).*uc*dx*dy; % for normalized objective function
                %g(2*M+1:3*M)=g(2*M+1:3*M)+(rz.*Gammac+wc).*uc*dx*dy; % for normalized objective function
            end
        end        
    end
    
end

% Add regularization terms to both the objective function and its gradients
betaG=0e-16; betaD=100*betaG; betaS=10*betaG; % regularization parameters

if strcmp(MinVar,'Gamma')|strcmp(MinVar,'GS')|strcmp(MinVar,'GD')|strcmp(MinVar,'All')
    [Gx,Gy] = pdegrad(P,T,Gammac);
    Gx1=pdeprtni(P,T,Gx); Gy1=pdeprtni(P,T,Gy);
    f=f+0.5*betaG*sum(Gx1.^2+Gy1.^2)*dx*dy;
    if nargout >1
        [Gxx, Gxy]=pdegrad(P,T,Gx1); [Gyx, Gyy]=pdegrad(P,T,Gy1);
        Gx2=pdeprtni(P,T,Gxx); Gy2=pdeprtni(P,T,Gyy);
        DeltaGamma=Gx2+Gy2;
        g(1:M)=g(1:M)-betaG*DeltaGamma*dx*dy;
        for j=1:ne
            nd=BdaryInfo(1,j);
            g(nd)=g(nd)-betaG*BdaryInfo(3,j)*Gx1(nd)+BdaryInfo(4,j)*Gy1(nd)*BdaryInfo(5,j);
        end
    end
end
if strcmp(MinVar,'Diff')|strcmp(MinVar,'GD')|strcmp(MinVar,'SD')|strcmp(MinVar,'All')
    [Dx,Dy] = pdegrad(P,T,Diffc);
    Dx1=pdeprtni(P,T,Dx); Dy1=pdeprtni(P,T,Dy);
    f=f+0.5*betaD*sum(Dx1.^2+Dy1.^2)*dx*dy;
    if nargout >1
        [Dxx, Dxy]=pdegrad(P,T,Dx1); [Dyx, Dyy]=pdegrad(P,T,Dy1);
        Dx2=pdeprtni(P,T,Dxx); Sy2=pdeprtni(P,T,Dyy);
        DeltaDiff=Dx2+Dy2;
        g(M+1:2*M)=g(M+1:2*M)-betaD*DeltaDiff*dx*dy;
        for j=1:ne
            nd=BdaryInfo(1,j);
            g(M+nd)=g(M+nd)-betaD*BdaryInfo(3,j)*Dx1(nd)+BdaryInfo(4,j)*Dy1(nd)*BdaryInfo(5,j);
        end
    end
end
if strcmp(MinVar,'Sigma')|strcmp(MinVar,'GS')|strcmp(MinVar,'SD')|strcmp(MinVar,'All')
    [Sx,Sy] = pdegrad(P,T,sigmac);
    Sx1=pdeprtni(P,T,Sx); Sy1=pdeprtni(P,T,Sy);
    f=f+0.5*betaS*sum(Sx1.^2+Sy1.^2)*dx*dy;
    if nargout >1
        [Sxx, Sxy]=pdegrad(P,T,Sx1); [Syx, Syy]=pdegrad(P,T,Sy1);
        Sx2=pdeprtni(P,T,Sxx); Sy2=pdeprtni(P,T,Syy);
        DeltaSigma=Sx2+Sy2;
        g(2*M+1:3*M)=g(2*M+1:3*M)-betaS*DeltaSigma*dx*dy;
        for j=1:ne
            nd=BdaryInfo(1,j);
            g(2*M+nd)=g(2*M+nd)-betaS*BdaryInfo(3,j)*Sx1(nd)+BdaryInfo(4,j)*Sy1(nd)*BdaryInfo(5,j);
        end
    end
end
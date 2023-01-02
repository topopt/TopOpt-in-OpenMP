%%%% 3D TOPOLOGY OPTIMIZATION CODE, MGCG ANALYSIS %%%%
% nelx - number of elements in x
% nely - number of elements in y
% nelz - number of elements in z
% volfrac - number of elements in x
% penal - number of elements in x
% rmin - 
% ft   - 
% nl -
% cgtol -
% cgmax -
function top3dmgcg_matrixfree(nelx,nely,nelz,volfrac,penal,rmin,ft,nl,cgtol,cgmax)
%
% example run command:
%
% top3dmgcg_matrixfree(64,32,32,0.12,3,2.4,1,4,1e-5,100)
% top3dmgcg_matrixfree(24,12,12,0.12,3,2.4,1,3,1e-5,100) 
% top3dmgcg_matrixfree(2,4,6,0.12,3,1.6,1,2,1e-5,100)
% 
% MATERIAL PROPERTIES
close all
gridContext.E0 = 1;
gridContext.Emin = 1e-6;
gridContext.nu = 0.3;
gridContext.penal = penal;
gridContext.nelx = nelx;
gridContext.nely = nely;
gridContext.nelz = nelz;

gridContext.elementSizeX = 0.5;
gridContext.elementSizeY = 0.5;
gridContext.elementSizeZ = 0.5;

%% PREPARE FINITE ELEMENT ANALYSIS
% Prepare fine grid
nelem = nelx*nely*nelz;

% number of nodes
nx = nelx+1; 
ny = nely+1; 
nz = nelz+1;

% size of state matrix and vectors
ndof = 3*nx*ny*nz;

% Prologation operators
Pu=cell(nl-1,1);
Pd=cell(nl-1,1);
for l = 1:nl-1
    [Pu{l,1}] = stateProjection(nelz/2^(l-1),nely/2^(l-1),nelx/2^(l-1));
    Pd{l,1} = Pu{l,1}';
end

% Define loads and supports (cantilever)
nodenrs(1:ny,1:nz,1:nx) = reshape(1:ny*nz*nx,ny,nz,nx);
%F = sparse(3*nodenrs(1:nely+1,1,nelx+1),1,-sin((0:nely)/nely*pi),ndof(1),1); % Sine load, bottom right
%F = sparse(3*nodenrs(1:nely+1,1,nelx+1),1,[-0.5; -ones(nely-1,1); -0.5],ndof(1),1); % constant load
F = sparse(3*nodenrs(1:nely+1,1,nelx+1),1,-ones(nely+1,1),ndof(1),1); % constant loa

U = zeros(ndof,1);

%% PREPARE FILTER
iH = ones(nelx*nely*nelz*(2*(ceil(rmin)-1)+1)^3,1);
jH = ones(size(iH));
sH = zeros(size(iH));
k = 0;
for i1 = 1:nelx
    for k1 = 1:nelz
        for j1 = 1:nely
            e1 = (i1-1)*nely*nelz + (k1-1)*nely + j1;
            for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),nelx)
                for k2 = max(k1-(ceil(rmin)-1),1):min(k1+(ceil(rmin)-1),nelz)
                    for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),nely)
                        e2 = (i2-1)*nely*nelz + (k2-1)*nely + j2;
                        k = k + 1;
                        iH(k) = e1;
                        jH(k) = e2;
                        sH(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2+(k1-k2)^2));
                    end
                end
            end
        end
    end
end
H = sparse(iH,jH,sH);
Hs = sum(H,2);
%% INITIALIZE ITERATION
x = volfrac*ones(nelem(1),1);
xPhys = x;
loop = 0;
change = 1;


%% START ITERATION
while change > 1e-2 && loop < 100
    loop = loop+1;
    %% FE-ANALYSIS
    gridContext.xPhys = xPhys;
    [cgiters,cgres,U] = mgcg_matrixfree(F,U,Pu,Pd,nl,5,cgtol,cgmax, gridContext);

    %% OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    [c,dc] = getComplianceAndSensetivity(U, gridContext);
    dv = ones(nelem(1),1);
    %% FILTERING/MODIFICATION OF SENSITIVITIES
    
    if ft == 1
        dc(:) = H*(x(:).*dc(:))./Hs./max(1e-3,x(:));
    elseif ft == 2
        dc(:) = H*(dc(:)./Hs);
        dv(:) = H*(dv(:)./Hs);
    end
    %% OPTIMALITY CRITERIA UPDATE OF DESIGN VARIABLES AND PHYSICAL DENSITIES
    g = mean(xPhys(:))-volfrac;
    l1 = 0; l2 = 1e9; move = 0.2;
    while (l2-l1)/(l1+l2) > 1e-6
        lmid = 0.5*(l2+l1);
        xnew = max(0,max(x-move,min(1,min(x+move,x.*sqrt(-dc./dv/lmid)))));
        gt=g+sum((dv(:).*(xnew(:)-x(:))));
	if gt>0, l1 = lmid; else l2 = lmid; end
    end
    change = max(abs(xnew(:)-x(:)));
    x = xnew;
    
    %% FILTERING OF DESIGN VARIABLES½1
    if ft == 1,         xPhys = xnew;
    elseif ft == 2,     xPhys(:) = (H*xnew(:))./Hs;
    end   
    %% PRINT RESULTS
    fprintf(' It.:%4i Obj.:%6.3e Vol.:%6.3e ch.:%4.2e relres: %4.2e iters: %4i \n',...
        loop,c,mean(xPhys(:)),change,cgres,cgiters);
    if mod(loop,10)==0
        %% PLOT
        isovals = shiftdim(reshape(xPhys,nely,nelz,nelx),2);
        isovals = smooth3(isovals,'box',1);
        patch(isosurface(isovals,0.5),'FaceColor',[0 0 1],'EdgeColor','none');
        patch(isocaps(isovals,0.5),'FaceColor',[1 0 0],'EdgeColor','none');
        view(3); axis equal tight off; camlight; drawnow
    end
end
%% PLOT
isovals = shiftdim(reshape(xPhys,nely,nelz,nelx),2);
isovals = smooth3(isovals,'box',1);
patch(isosurface(isovals,0.5),'FaceColor',[0 0 1],'EdgeColor','none');
patch(isocaps(isovals,0.5),'FaceColor',[1 0 0],'EdgeColor','none');
view(3); axis equal tight off; camlight;
end

%% FUNCTION mgcg - MULTIGRID PRECONDITIONED CONJUGATE GRADIENTS
function [i,relres,u] = mgcg_matrixfree(b,u,Pu,Pd,nl,nswp,tol,maxiter,gridContext)

r = b - matvecprod(u, gridContext);
res0 = norm(b); 

% Jacobi smoother
omega = 0.6;
invD = cell(nl-1,1);
invD{1,1} = 1./ generateMatrixDiagonal(gridContext);
for l = 2:nl
    invD{l,1} = 1./ generateMatrixDiagonalSubspace(gridContext,l);
end

Kc = generateMatrixSubspace(gridContext, nl);
Lfac = chol(Kc,'lower'); 
Ufac = Lfac';

for i=1:1e6 
    z = VCycle(r,Pu,Pd,1,nl,invD,omega,nswp,gridContext,Lfac,Ufac);
    rho = r'*z;
    
    if i==1
        p=z;
    else
        beta=rho/rho_p;
        p=beta*p+z;
    end
    q=matvecprod(p, gridContext);
    dpr=p'*q;
    alpha=rho/dpr;
    u=u+alpha*p;
    r=r-alpha*q;
    rho_p=rho;
    relres=norm(r)/res0;
    if relres<tol || i>=maxiter
        break
    end
    %fprintf('it.: %d, rho: %e \n',i,relres);
end
end

%% FUNCTION VCycle - COARSE GRID CORRECTION
function z = VCycle(r,Pu,Pd,l,nl,invD,omega,nswp,gridContext,Lfac,Ufac)
z = 0*r;

if (l==1)
    z = smthdmpjac(z,r,invD{l,1},omega,nswp,gridContext);
    d = r - matvecprod(z, gridContext);
else
    z = smthdmpjacSubspace(z,r,invD{l,1},omega,nswp,gridContext,l);
    d = r - matvecprodSubspace(z, gridContext,l); 
end

dh2 = Pd{l,1}*d;
if (nl == l+1)
    % vh2 = Ufac \ (Lfac \ dh2);
    vh2 = coarse_cg(dh2,0*dh2,nswp,1e-6,100,invD,gridContext,l+1);
    %vh2 = smthdmpjacSubspace(0*dh2,dh2,invD{l+1,1},omega,10*nswp,gridContext,l+1);
else
    vh2 = VCycle(dh2,Pu,Pd,l+1,nl,invD,omega,nswp,gridContext,Lfac,Ufac);
end
v = Pu{l,1}*vh2;
z = z + v;

if (l==1)
	z = smthdmpjac(z,r,invD{l,1},omega,nswp,gridContext);
else
    z = smthdmpjacSubspace(z,r,invD{l,1},omega,nswp,gridContext,l);
end
end

%% FUNCTIODN smthdmpjac - DAMPED JACOBI SMOOTHER
function [u] = smthdmpjac(u,b,invD,omega,nswp,gridContext)
for i = 1:nswp
    u = u - omega*invD.* matvecprod(u, gridContext) + omega*invD.*b;
end
end

%% FUNCTIODN smthdmpjac - DAMPED JACOBI SMOOTHER
function [u] = smthdmpjacSubspace(u,b,invD,omega,nswp,gridContext,l)
for i = 1:nswp
    u = u - omega*invD.* matvecprodSubspace(u, gridContext,l) + omega*invD.*b;
end
end

%% FUNCTION mgcg - MULTIGRID PRECONDITIONED CONJUGATE GRADIENTS
function [u] = coarse_cg(b,u,nswp,tol,maxiter,invD,gridContext,l)

r = b - matvecprodSubspace(u, gridContext,l);
res0 = norm(b); 

omega = 0.6;

for i=1:1e6 
    z = smthdmpjacSubspace(0*r,r,invD{l,1},omega,nswp,gridContext,l);
    %z = VCycle(r,Pu,Pd,1,nl,invD,omega,nswp,gridContext,Lfac,Ufac);
    rho = r'*z;
    
    if i==1
        p=z;
    else
        beta=rho/rho_p;
        p=beta*p+z;
    end
    q=matvecprodSubspace(p, gridContext,l);
    dpr=p'*q;
    alpha=rho/dpr;
    u=u+alpha*p;
    r=r-alpha*q;
    rho_p=rho;
    relres=norm(r)/res0;
    if relres<tol || i>=maxiter
        break
    end
    %fprintf('it.: %d, rho: %e \n',i,relres);
end
end

%% FUNCTION naive matrix-vector product
function [v] = matvecprod(u, gridContext)

% unpack for readability, as this is mock code anyway
E0 = gridContext.E0;
Emin = gridContext.Emin;
penal = gridContext.penal;

ny = gridContext.nely +1;
nz = gridContext.nelz +1;

nelx = gridContext.nelx;
nely = gridContext.nely;
nelz = gridContext.nelz;

fixeddofs = getFixedDof(nelx,nely,nelz);

% correct version should also depend on dimensions of domain
KE = Ke3DSize(gridContext.nu,...
    gridContext.elementSizeX,...
    gridContext.elementSizeY,...
    gridContext.elementSizeZ);

v = zeros(size(u));
for i = 1:nelx
    for k = 1:nelz
        for j = 1:nely
            elementIndex = (i-1)*nely*nelz + (k-1)*nely + j;
            KELocal = (Emin+gridContext.xPhys(elementIndex)'.^penal*(E0-Emin)) * KE;
            
            nx_1 = i;
            nx_2 = i+1;
            nz_1 = k;
            nz_2 = k+1;
            ny_1 = j;
            ny_2 = j+1;
            
            nIndex1 = (nx_1-1)*ny*nz + (nz_1-1)*ny + ny_2;
            nIndex2 = (nx_2-1)*ny*nz + (nz_1-1)*ny + ny_2;
            nIndex3 = (nx_2-1)*ny*nz + (nz_1-1)*ny + ny_1;
            nIndex4 = (nx_1-1)*ny*nz + (nz_1-1)*ny + ny_1;
            nIndex5 = (nx_1-1)*ny*nz + (nz_2-1)*ny + ny_2;
            nIndex6 = (nx_2-1)*ny*nz + (nz_2-1)*ny + ny_2;
            nIndex7 = (nx_2-1)*ny*nz + (nz_2-1)*ny + ny_1;
            nIndex8 = (nx_1-1)*ny*nz + (nz_2-1)*ny + ny_1;
            
            edof = [...
                3*nIndex1-2:3*nIndex1 3*nIndex2-2:3*nIndex2 ...
                3*nIndex3-2:3*nIndex3 3*nIndex4-2:3*nIndex4 ...
                3*nIndex5-2:3*nIndex5 3*nIndex6-2:3*nIndex6 ...
                3*nIndex7-2:3*nIndex7 3*nIndex8-2:3*nIndex8 ...
            ];
            
            v(edof) = v(edof) + KELocal * u(edof);
        end
    end
end

% set boundary conditions
v(fixeddofs) = u(fixeddofs);

end

%% FUNCTION naive generation of Matrix Diagonal
function [d] = generateMatrixDiagonal(gridContext)

% unpack for readability, as this is mock code anyway
E0 = gridContext.E0;
Emin = gridContext.Emin;
penal = gridContext.penal;

ny = gridContext.nely +1;
nz = gridContext.nelz +1;

nelx = gridContext.nelx;
nely = gridContext.nely;
nelz = gridContext.nelz;

fixeddofs = getFixedDof(nelx,nely,nelz);

% correct version should also depend on dimensions of domain
KE = Ke3DSize(gridContext.nu,...
    gridContext.elementSizeX,...
    gridContext.elementSizeY,...
    gridContext.elementSizeZ);

d = zeros(3*(nelx+1)*(nely+1)*(nelz+1),1);
for i = 1:nelx
    for k = 1:nelz
        for j = 1:nely
            elementIndex = (i-1)*nely*nelz + (k-1)*nely + j;
            KELocal = (Emin+gridContext.xPhys(elementIndex)'.^penal*(E0-Emin)) * KE;
            
            edof = getEdof(i,j,k,ny,nz);
            
            d(edof) = d(edof) + diag(KELocal);
        end
    end
end

% set boundary conditions
d(fixeddofs) = 1.0;

end

%% FUNCTION naive matrix-vector product
function [v] = matvecprodSubspace(u, gridContext, l)

% unpack for readability, as this is mock code anyway
nelxc = gridContext.nelx/ 2^(l-1);
nelyc = gridContext.nely/ 2^(l-1);
nelzc = gridContext.nelz/ 2^(l-1);
nzc = nelzc+1;
nyc = nelyc+1;

ncell = 2^(l-1);

fixeddofs = getFixedDof(nelxc,nelyc,nelzc);

v = zeros(size(u));

KEpre = getKEPreIntegration(l, gridContext);

% loop over coarse mesh
for i = 1:nelxc
    for k = 1:nelzc
        for j = 1:nelyc
            KE = assembleKEFromPre(i,j,k,ncell,KEpre,gridContext);
            
            edof = getEdof(i,j,k,nyc,nzc);
            
            v(edof) = v(edof) + KE * u(edof);
        end
    end
end

% set boundary conditions
v(fixeddofs) = u(fixeddofs);

end

%% FUNCTION naive generation of Matrix Diagonal
function [d] = generateMatrixDiagonalSubspace(gridContext, l)

% unpack for readability, as this is mock code anyway
nelxc = gridContext.nelx/ 2^(l-1);
nelyc = gridContext.nely/ 2^(l-1);
nelzc = gridContext.nelz/ 2^(l-1);
nzc = nelzc+1;
nyc = nelyc+1;

ncell = 2^(l-1);

KEpre = getKEPreIntegration(l, gridContext);

fixeddofs = getFixedDof(nelxc,nelyc,nelzc);

d = zeros(3*(nelxc+1)*(nelyc+1)*(nelzc+1),1);

for i = 1:nelxc
    for k = 1:nelzc
        for j = 1:nelyc
            
            KE = assembleKEFromPre(i,j,k,ncell,KEpre,gridContext);
            
            edof = getEdof(i,j,k,nyc,nzc);
            
            d(edof) = d(edof) + diag(KE);
        end
    end
end

% set boundary conditions
d(fixeddofs) = 1.0;

end

%% FUNCTION naive generation of Matrix Diagonal
function [K] = generateMatrixSubspace(gridContext, l)
nelxc = gridContext.nelx/ 2^(l-1);
nelyc = gridContext.nely/ 2^(l-1);
nelzc = gridContext.nelz/ 2^(l-1);
nzc = nelzc+1;
nyc = nelyc+1;

ndof = 3*nzc*nyc*(nelxc+1);

ncell = 2^(l-1);

KEpre = getKEPreIntegration(l, gridContext);

fixeddofs = getFixedDof(nelxc,nelyc,nelzc);

N = ones(ndof,1); 
N(fixeddofs) = 0; 
Null = spdiags(N,0,ndof,ndof);

iK = zeros(24*24*nelxc*nelyc*nelzc,1);
jK = zeros(24*24*nelxc*nelyc*nelzc,1);
sK = zeros(24*24*nelxc*nelyc*nelzc,1);

cc=1;

for i = 1:nelxc
    for k = 1:nelzc
        for j = 1:nelyc
            
            KE = assembleKEFromPre(i,j,k,ncell,KEpre,gridContext);
            
            edof = getEdof(i,j,k,nyc,nzc);
            
            edofJ = repmat(edof,1,24);
            edofI = repmat(edof,24,1);
            
            iK(cc:cc+24*24-1) = edofI(:);
            jK(cc:cc+24*24-1) = edofJ(:);
            sK(cc:cc+24*24-1) = KE(:);
            cc = cc + 24*24;
        end
    end
end

K = sparse(iK,jK,sK,ndof,ndof);

% set boundary conditions
K = Null'*K*Null - (Null-speye(ndof,ndof));
end

%% FUNCTION naive computation of compliance
function [c,dc] = getComplianceAndSensetivity(u, gridContext)

% unpack for readability, as this is mock code anyway
E0 = gridContext.E0;
Emin = gridContext.Emin;
penal = gridContext.penal;

ny = gridContext.nely +1;
nz = gridContext.nelz +1;

nelx = gridContext.nelx;
nely = gridContext.nely;
nelz = gridContext.nelz;

% correct version should also depend on dimensions of domain
KE = Ke3DSize(gridContext.nu,...
    gridContext.elementSizeX,...
    gridContext.elementSizeY,...
    gridContext.elementSizeZ);

c = 0;
dc = zeros(size(gridContext.xPhys));

for i = 1:nelx
    for k = 1:nelz
        for j = 1:nely
            
            elementIndex = (i-1)*nely*nelz + (k-1)*nely + j;
            
            edof = getEdof(i,j,k,ny,nz);
            
            ce = u(edof)' * KE * u(edof);
        
            c = c + ce * (Emin+gridContext.xPhys(elementIndex)'.^penal*(E0-Emin));
            dc(elementIndex) = ce * -penal*(E0-Emin)*gridContext.xPhys(elementIndex).^(penal-1);
        end
    end
end
end

%% FUNCTION getEdof - get edof at element i,j,k
function [edof] = getEdof(i,j,k,ny,nz)
nx_1 = i;
nx_2 = i+1;
nz_1 = k;
nz_2 = k+1;
ny_1 = j;
ny_2 = j+1;

nIndex1 = (nx_1-1)*ny*nz + (nz_1-1)*ny + ny_2;
nIndex2 = (nx_2-1)*ny*nz + (nz_1-1)*ny + ny_2;
nIndex3 = (nx_2-1)*ny*nz + (nz_1-1)*ny + ny_1;
nIndex4 = (nx_1-1)*ny*nz + (nz_1-1)*ny + ny_1;
nIndex5 = (nx_1-1)*ny*nz + (nz_2-1)*ny + ny_2;
nIndex6 = (nx_2-1)*ny*nz + (nz_2-1)*ny + ny_2;
nIndex7 = (nx_2-1)*ny*nz + (nz_2-1)*ny + ny_1;
nIndex8 = (nx_1-1)*ny*nz + (nz_2-1)*ny + ny_1;

edof = [...
    3*nIndex1-2:3*nIndex1 3*nIndex2-2:3*nIndex2 ...
    3*nIndex3-2:3*nIndex3 3*nIndex4-2:3*nIndex4 ...
    3*nIndex5-2:3*nIndex5 3*nIndex6-2:3*nIndex6 ...
    3*nIndex7-2:3*nIndex7 3*nIndex8-2:3*nIndex8 ...
    ];
end

%% FUNCTION getKEPreIntegration - preintegrate KE for cell structure
function [KEpre] = getKEPreIntegration(l, gc)

ncell = 2^(l-1);
int_points = 5;
C = getC(gc.nu);

a = gc.elementSizeX * 2^(l-1);
b = gc.elementSizeY * 2^(l-1);
c = gc.elementSizeZ * 2^(l-1);

xx = [...
    -a -b -c ...
     a -b -c ...
     a  b -c ...
    -a  b -c ...
    -a -b  c ...
     a -b  c ...
     a  b  c ...
    -a  b  c ...
];

spacing = 2/ncell/int_points;
subCellVolume = spacing*spacing*spacing;

%pre- integrate matrices, to speed up product.
KEpre = cell(ncell,ncell,ncell);
for ii = 1:ncell
    for kk = 1:ncell
        for jj = 1:ncell
            KEpre{ii,jj,kk} = zeros(24);
            
            starti = -1 + spacing/2 + 2/ncell*(ii-1);
            endi   = 1 - spacing/2 - 2/ncell*(ncell-ii);
            ipts = starti:spacing:endi;
            
            startj = -1 + spacing/2 + 2/ncell*(jj-1);
            endj   = 1 - spacing/2 - 2/ncell*(ncell-jj);
            jpts = startj:spacing:endj;
            jpts = -jpts; 
            % important to flip y/eta-coordinate, due to bad numbering in
            % original code.
            
            startk = -1 + spacing/2 + 2/ncell*(kk-1);
            endk   = 1 - spacing/2 - 2/ncell*(ncell-kk);
            kpts = startk:spacing:endk;
            
            for xi = ipts
                for eta = jpts
                    for zeta = kpts
                        [B,jdet] = getB([xi,eta,zeta],xx);
                        KEpre{ii,jj,kk} = KEpre{ii,jj,kk} + (jdet * subCellVolume) * (B'*C*B);
                    end
                end
            end
        end
    end
end
end

%% FUNCTION getKEPreIntegration - preintegrate KE for cell structure
function [KE] = assembleKEFromPre(i,j,k,ncell,KEpre,gc)
KE = zeros(24);
for ii = 1:ncell
    for kk = 1:ncell
        for jj = 1:ncell
            ifine =(i-1)*ncell + ii;
            jfine =(j-1)*ncell + jj;
            kfine =(k-1)*ncell + kk;
            
            elementIndex = (ifine-1)*gc.nely*gc.nelz + (kfine-1)*gc.nely + jfine;
            localFact = (gc.Emin+gc.xPhys(elementIndex)'.^gc.penal*(gc.E0-gc.Emin));
            
            KE = KE + localFact*KEpre{ii,jj,kk};
        end
    end
end
end

% this is were boundary condtions are defined for the grid
function [fdof] = getFixedDof(nelx,nely,nelz)
    fdof = 1:3*(nely+1)*(nelz+1);
end

%% FUNCTION prepcoarse - PREPARE MG PROLONGATION OPERATOR
function [Pu] = stateProjection(nex,ney,nez)
% Assemble state variable prolongation
maxnum = nex*ney*nez*20;
iP = zeros(maxnum,1); jP = zeros(maxnum,1); sP = zeros(maxnum,1);
nexc = nex/2; neyc = ney/2; nezc = nez/2;
% Weights for fixed distances to neighbors on a structured grid 
vals = [1,0.5,0.25,0.125];
cc = 0;
for nx = 1:nexc+1
    for ny = 1:neyc+1
        for nz = 1:nezc+1
            col = (nx-1)*(neyc+1)+ny+(nz-1)*(neyc+1)*(nexc+1); 
            % Coordinate on fine grid
            nx1 = nx*2 - 1; ny1 = ny*2 - 1; nz1 = nz*2 - 1;
            % Loop over fine nodes within the rectangular domain
            for k = max(nx1-1,1):min(nx1+1,nex+1)
                for l = max(ny1-1,1):min(ny1+1,ney+1)
                    for h = max(nz1-1,1):min(nz1+1,nez+1)
                        row = (k-1)*(ney+1)+l+(h-1)*(nex+1)*(ney+1); 
                        % Based on squared dist assign weights: 1.0 0.5 0.25 0.125
                        ind = 1+((nx1-k)^2+(ny1-l)^2+(nz1-h)^2);
                        cc=cc+1; iP(cc)=3*row-2; jP(cc)=3*col-2; sP(cc)=vals(ind);
                        cc=cc+1; iP(cc)=3*row-1; jP(cc)=3*col-1; sP(cc)=vals(ind);
                        cc=cc+1; iP(cc)=3*row; jP(cc)=3*col; sP(cc)=vals(ind);
                    end
                end
            end
        end
    end
end
% Assemble matrices
Pu = sparse(iP(1:cc),jP(1:cc),sP(1:cc));
end

%% FUNCTION Ke3D - ELEMENT STIFFNESS MATRIX
function KE = Ke3DSize(nu,a,b,c)

xx = [...
    -a -b -c ...
     a -b -c ...
     a  b -c ...
    -a  b -c ...
    -a -b  c ...
     a -b  c ...
     a  b  c ...
    -a  b  c ...
];

xpts = [-1/sqrt(3), 1/sqrt(3)];
ypts = [-1/sqrt(3), 1/sqrt(3)];
zpts = [-1/sqrt(3), 1/sqrt(3)];

C = getC(nu);
KE = zeros(24);

for xi = xpts
    for eta = ypts
        for zeta = zpts
         
            [B,jdet] = getB([xi, eta, zeta], xx);

            KE = KE +jdet*(B'*C*B);
        end
    end
end
end

function [C] = getC(nu)
temp1 = (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
temp2 = nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
temp3 = 1.0 / (2.0 * (1.0 + nu));

C = zeros(6);
C(1, 1) = temp1;
C(2, 2) = temp1;
C(3, 3) = temp1;
C(4, 4) = temp3;
C(5, 5) = temp3;
C(6, 6) = temp3;
C(1, 2) = temp2;
C(2, 1) = temp2;
C(1, 3) = temp2;
C(3, 1) = temp2;
C(2, 3) = temp2;
C(3, 2) = temp2;
end

function [B,jdet] = getB(iso,xe)

    xi = iso(1);
    eta = iso(2);
    zeta = iso(3);

    n1xi   = -0.125 * (1 - eta) * (1 - zeta);
    n1eta  = -0.125 * (1 - xi) * (1 - zeta);
    n1zeta = -0.125 * (1 - xi) * (1 - eta);
    n2xi   = 0.125 * (1 - eta) * (1 - zeta);
    n2eta  = -0.125 * (1 + xi) * (1 - zeta);
    n2zeta = -0.125 * (1 + xi) * (1 - eta);

    n3xi   = 0.125 * (1 + eta) * (1 - zeta);
    n3eta  = 0.125 * (1 + xi) * (1 - zeta);
    n3zeta = -0.125 * (1 + xi) * (1 + eta);
    n4xi   = -0.125 * (1 + eta) * (1 - zeta);
    n4eta  = 0.125 * (1 - xi) * (1 - zeta);
    n4zeta = -0.125 * (1 - xi) * (1 + eta);

    n5xi   = -0.125 * (1 - eta) * (1 + zeta);
    n5eta  = -0.125 * (1 - xi) * (1 + zeta);
    n5zeta = 0.125 * (1 - xi) * (1 - eta);
    n6xi   = 0.125 * (1 - eta) * (1 + zeta);
    n6eta  = -0.125 * (1 + xi) * (1 + zeta);
    n6zeta = 0.125 * (1 + xi) * (1 - eta);

    n7xi   = 0.125 * (1 + eta) * (1 + zeta);
    n7eta  = 0.125 * (1 + xi) * (1 + zeta);
    n7zeta = 0.125 * (1 + xi) * (1 + eta);
    n8xi   = -0.125 * (1 + eta) * (1 + zeta);
    n8eta  = 0.125 * (1 - xi) * (1 + zeta);
    n8zeta = 0.125 * (1 - xi) * (1 + eta);

    L = zeros(6,9);
    jac = zeros(3);
    jacinvt = zeros(9);
    Nt = zeros(9,24);
    
    L(1, 1) = 1.0;
    L(2, 5) = 1.0;
    L(3, 9) = 1.0;
    L(4, 2) = 1.0;
    L(4, 4) = 1.0;
    L(5, 6) = 1.0;
    L(5, 8) = 1.0;
    L(6, 3) = 1.0;
    L(6, 7) = 1.0;

    Nt(1, 1)  = n1xi;
    Nt(2, 1)  = n1eta;
    Nt(3, 1)  = n1zeta;
    Nt(1, 4)  = n2xi;
    Nt(2, 4)  = n2eta;
    Nt(3, 4)  = n2zeta;
    Nt(1, 7)  = n3xi;
    Nt(2, 7)  = n3eta;
    Nt(3, 7)  = n3zeta;
    Nt(1, 10)  = n4xi;
    Nt(2, 10)  = n4eta;
    Nt(3, 10)  = n4zeta;
    Nt(1, 13) = n5xi;
    Nt(2, 13) = n5eta;
    Nt(3, 13) = n5zeta;
    Nt(1, 16) = n6xi;
    Nt(2, 16) = n6eta;
    Nt(3, 16) = n6zeta;
    Nt(1, 19) = n7xi;
    Nt(2, 19) = n7eta;
    Nt(3, 19) = n7zeta;
    Nt(1, 22) = n8xi;
    Nt(2, 22) = n8eta;
    Nt(3, 22) = n8zeta;

    Nt(4, 2)  = n1xi;
    Nt(5, 2)  = n1eta;
    Nt(6, 2)  = n1zeta;
    Nt(4, 5)  = n2xi;
    Nt(5, 5)  = n2eta;
    Nt(6, 5)  = n2zeta;
    Nt(4, 8)  = n3xi;
    Nt(5, 8)  = n3eta;
    Nt(6, 8)  = n3zeta;
    Nt(4, 11) = n4xi;
    Nt(5, 11) = n4eta;
    Nt(6, 11) = n4zeta;
    Nt(4, 14) = n5xi;
    Nt(5, 14) = n5eta;
    Nt(6, 14) = n5zeta;
    Nt(4, 17) = n6xi;
    Nt(5, 17) = n6eta;
    Nt(6, 17) = n6zeta;
    Nt(4, 20) = n7xi;
    Nt(5, 20) = n7eta;
    Nt(6, 20) = n7zeta;
    Nt(4, 23) = n8xi;
    Nt(5, 23) = n8eta;
    Nt(6, 23) = n8zeta;

    Nt(7, 3)  = n1xi;
    Nt(8, 3)  = n1eta;
    Nt(9, 3)  = n1zeta;
    Nt(7, 6)  = n2xi;
    Nt(8, 6)  = n2eta;
    Nt(9, 6)  = n2zeta;
    Nt(7, 9)  = n3xi;
    Nt(8, 9)  = n3eta;
    Nt(9, 9)  = n3zeta;
    Nt(7, 12) = n4xi;
    Nt(8, 12) = n4eta;
    Nt(9, 12) = n4zeta;
    Nt(7, 15) = n5xi;
    Nt(8, 15) = n5eta;
    Nt(9, 15) = n5zeta;
    Nt(7, 18) = n6xi;
    Nt(8, 18) = n6eta;
    Nt(9, 18) = n6zeta;
    Nt(7, 21) = n7xi;
    Nt(8, 21) = n7eta;
    Nt(9, 21) = n7zeta;
    Nt(7, 24) = n8xi;
    Nt(8, 24) = n8eta;
    Nt(9, 24) = n8zeta;


    jac(1, 1) = n1xi * xe(1) + n2xi * xe(4) + n3xi * xe(7) + n4xi * xe(10) +...
        n5xi * xe(13) + n6xi * xe(16) + n7xi * xe(19) + n8xi * xe(22);
    jac(2, 1) = n1eta * xe(1) + n2eta * xe(4) + n3eta * xe(7) + n4eta * xe(10) +...
        n5eta * xe(13) + n6eta * xe(16) + n7eta * xe(19) + n8eta * xe(22);
    jac(3, 1) = n1zeta * xe(1) + n2zeta * xe(4) + n3zeta * xe(7) + n4zeta * xe(10) + n5zeta * xe(13) + n6zeta * xe(16) +n7zeta * xe(19) + n8zeta * xe(22);

    jac(1, 2) = n1xi * xe(2) + n2xi * xe(5) + n3xi * xe(8) + n4xi * xe(11) + n5xi * xe(14) + n6xi * xe(17) +n7xi * xe(20) + n8xi * xe(23);
    jac(2, 2) = n1eta * xe(2) + n2eta * xe(5) + n3eta * xe(8) + n4eta * xe(11) + n5eta * xe(14) + n6eta * xe(17) +n7eta * xe(20) + n8eta * xe(23);
    jac(3, 2) = n1zeta * xe(2) + n2zeta * xe(5) + n3zeta * xe(8) + n4zeta * xe(11) + n5zeta * xe(14) + n6zeta * xe(17) +n7zeta * xe(20) + n8zeta * xe(23);

    jac(1, 3) = n1xi * xe(3) + n2xi * xe(6) + n3xi * xe(9) + n4xi * xe(12) + n5xi * xe(15) + n6xi * xe(18) +n7xi * xe(21) + n8xi * xe(24);
    jac(2, 3) = n1eta * xe(3) + n2eta * xe(6) + n3eta * xe(9) + n4eta * xe(12) + n5eta * xe(15) + n6eta * xe(18) +n7eta * xe(21) + n8eta * xe(24);
    jac(3, 3) = n1zeta * xe(3) + n2zeta * xe(6) + n3zeta * xe(9) + n4zeta * xe(12) + n5zeta * xe(15) + n6zeta * xe(18) +n7zeta * xe(21) + n8zeta * xe(24);

    jdet = det(jac);
    ijac = inv(jac);
    
    jacinvt(1:3,1:3) = ijac;
    jacinvt(4:6,4:6) = ijac;
    jacinvt(7:9,7:9) = ijac;

    B = (L * jacinvt * Nt);
end
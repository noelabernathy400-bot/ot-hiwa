function generate_synthetic_baselines(max_trials)
% The only missing configuration; original author generator and baselines.
if nargin<1, max_trials=50; end
here=fileparts(mfilename('fullpath')); root=fileparts(fileparts(fileparts(here)));
workspace=fileparts(fileparts(fileparts(root)));
code=fullfile(workspace,'HiWA','hiwa-matlab','code');
addpath(fullfile(code,'toolbox')); addpath(fullfile(code,'third party tools'));
out=fullfile(root,'data','processed','synthetic');
if ~exist(out,'dir'), mkdir(out); end
S=5; D=5; d=2; N=50; Nvar=0; delta=0;
methods={'WA','SA','ICP','CORAL'};
wp.maxiter=200; wp.limit_gamma=.1; wp.sh_maxiter=1000;
wp.gamma=100; wp.alpha_gamma=.95; wp.display=0;
for seed=1:max_trials
    destination=fullfile(out,sprintf('S%02d_D%02d_seed%02d.mat',S,D,seed));
    if isfile(destination), continue; end
    rng(seed);
    [~,~,~,~,XX,YY,Rgt,~,~]=GenerateSyntheticSubspaceData(S,D,d,N,Nvar,delta,0);
    baseline_rng=rng;
    aligned=nan(D,size(XX,2),4);
    R=WA(XX,YY,Rgt,wp); aligned(:,:,1)=R*XX;
    R=SA(XX',YY')'; aligned(:,:,2)=R*XX;
    R=ICP(XX',YY'); aligned(:,:,3)=R*XX;
    aligned(:,:,4)=CORAL(XX',YY')';
    truth=Rgt*XX;
    baseline_errors=arrayfun(@(k) norm(aligned(:,:,k)-truth,'fro')^2/norm(truth,'fro')^2,1:4);
    temporary=[destination '.tmp.mat'];
    save(temporary,'XX','YY','Rgt','S','D','d','N','seed','Nvar','delta',...
         'methods','baseline_errors','aligned','baseline_rng','-v7');
    movefile(temporary,destination,'f');
    fprintf('done S=%d D=%d seed=%d\n',S,D,seed);
end
end

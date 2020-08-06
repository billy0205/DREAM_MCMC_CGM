using Pkg
using DataFrames
using Distributions
using StatsBase
using Random

### function ###
function dream_mcmc(prior_dream,pdfDream,N,T,d,dataW)
    delta,c,c_star,n_CR,p_g=3,0.1,1e-12,3,0.2 #Default of algorithmic parameter
    x = zeros(T,d,N) #zeros(Array{Float64}(T,d,N))          #Preallocate chain and density
    p_x=zeros(T,N)#zeros(Array{Float64}(T,N))
    rmse_x=zeros(T,N)#zeros(Array{Float64}(T,N))
    std_X=zeros(d,1)#zeros(Array{Float64}(d,1))
    R=fill(0,N,N-1)#zeros(Array{Int}(N,N-1))
    J=zeros(n_CR)#Variables selection prob. crossover
    n_id=zeros(n_CR)
    for idx=1:N#R-matrix: index of chains for DE
        R[idx,:]=setdiff(1:N,idx)
    end
    CR=collect(1:n_CR)/n_CR   #Crossover values and select prob.
    pCR=repeat([1],inner=n_CR)
    X=rand(prior_dream,N)     #Create initial population
    Xp=zeros(size(X))
    p_X=zeros(N,1)[:,1]
    rmse_X=zeros(N,1)[:,1]
    for k=1:N
        (p_X[k,1],rmse_X[k,1])=pdfDream(dataW,X[1:d,k],parModel)
    end
    p_Xp=zeros(N,1)
    rmse_Xp=zeros(N,1) #RMSE_Xp(just for recroding)
    x[1,:,:]=X  #Store initial states and density
    p_x[1,1:N] =transpose(p_X)
    rmse_x[1,1:N] =transpose(rmse_X) #RMSE(just for recroding)
    #Dynamic part: Evolution of N chains
    for idxt=2:T
        draw= fill(0,N-1,N)#Array{Int}(N-1,N)   #permute[1,...,N] N times
        for idx=1:N
            draw[:,idx]=Random.randperm(N-1)
        end
        dX=zeros(N,d)                    #Set N jump vectors to zero
        lambda = rand(Uniform(-c,c), N,1)#Draw N lambda values
        for idx=1:d                      #Compute std each dimension
            std_X[idx,:] .= std(X[idx,:])
        end
        for i=1:N#Create prpposals and accept/reject
            D=rand(1:delta)#Select delta (equal select. prob.)
            a=R[i,draw[1:D,i]]#Extract vectors a and b not equal i
            b=R[i,draw[(D+1):(2*D),i]]
            id=StatsBase.sample(collect(1:n_CR),pweights(pCR))#Select index of crossover value
            z=rand(Uniform(0,1),d,1)#draw d values fomr U[0,1]
            A=findall(x->x<CR[id],vec(z))#findall(z. <CR[id])#Derive subset A selected dimensions
            d_star = size(A)[1]#The sampled dimensions
            if d_star == 0 #A must contain at least one value
                zmin=minimum(z)
                A= findall(x->x==zmin,vec(z))
                d_star=1
            end
            gamma_d=2.38/sqrt(2*D*d_star)#Calculate jump rate
            g=StatsBase.sample([gamma_d, 1],pweights([1-p_g,p_g]))#Select gamma: 80/20 mix [defaule 1]
            dXs2=(1+lambda[i])*g*sum(reshape(X[A,a]-X[A,b],d_star,length(a)),dims=2)#
            dXs1=c_star*rand(Normal(0, 1),d_star)
            dXs12=copy(dXs1+vec(dXs2))
            dX[i,A]=copy(dXs12)
            Xp[1:d,i]=X[1:d,i]+dX[i,1:d]#compute ith proposal

            idx_outlier=checkPar(Xp[1:d,i],XB)
            Xp[idx_outlier,i]=X[idx_outlier,i]
            dX[i,idx_outlier] .=0.
            (p_Xp[i,1],rmse_Xp[i,1])=pdfDream(dataW,Xp[1:d,i],parModel)
            alpha_h=exp((p_Xp[i,1]-p_X[i,1])) #*10^6
            p_acc=min(1,alpha_h)#alpha_h*prior_ratio)#
            if p_acc>rand(1)[1]#p_acc larger than U[0,1] #20180519
                X[1:d,i]=copy(Xp[1:d,i])#True accept proposal
                p_X[i,1]=copy(p_Xp[i,1])
                rmse_X[i,1]=copy(rmse_Xp[i,1])
            else
                #dX[i,1:d]=0#Set jump back to zero for pCR
                dX[i,1:d].=0#Set jump back to zero for pCR
            end
            J[id]=J[id]+sum((dX[i,1:d]./std_X).^2)
            n_id[id]=n_id[id]+1#How many times idx crossover used
        end # End of [for i=1:N]
        x[idxt,1:d,1:N]=X#Append current X and density
        p_x[idxt,1:N]=p_X
        rmse_x[idxt,1:N]=rmse_X
        if idxt < (T/10)#Update selection prob. crossover
            pCR = J./n_id
            pCR = pCR/sum(pCR)
        end
        #X,p_X=checkIQRSimriw(X,p_x,idxt)#Outlier detection and correction
    end
    return(x,p_x,rmse_x)
end

function loglik(x)
    n=size(x,1)
    #a=(-n/2)*log(2*pi*std(x))+(-1/(2*var(x)))*(sum(x.^2))
    a=(-n/2)*log(2*pi)-sum(log(sigma_prior))+(-(1/2)*sum((x./sigma_prior)^2)
    return(a)
end

function lik(x)
    n=size(x,1)
    a=(1/((2*pi*std(x))^(n/2)))*exp((-1/(2*var(x)))*(sum(x.^2)))
    return(a)
end

function checkPar(V,bound)
    Vu=sign.(bound[:,2].-V)#bound[,2]=upper bound
    Vl=sign.(-bound[:,1].+V)
    Vc=Vu+Vl
    index=findall(Vc.==0)
    return(index)
end

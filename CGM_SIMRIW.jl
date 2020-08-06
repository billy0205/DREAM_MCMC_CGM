function MCMC_SIMRIW_Heading_main(n_iteration,n_chain,data_w,par_variety_prior,par_model,chain_dif)
    n_par=size(par_variety_prior,1)   # Number of SIMRIW parameters (Heading date)
    n_obs=size(data_w,1)
    record_sample=zeros(n_iteration*n_par,n_chain)  #SIMRIW parameters related to DVI calculation (heading)
    record_accepted=zeros(n_iteration*n_par,n_chain)   #record the accepted or not in each turn
    record_sigma=zeros(n_iteration*n_par,n_chain)      #model performance (not necessary)
    record_md=zeros(n_iteration*n_par,n_chain)
    for chain=1:n_chain
        par_heading=zeros(n_iteration*n_par)
        ra=zeros(n_iteration*n_par)
        rsg=zeros(n_iteration*n_par)
        md=zeros(n_iteration*n_par)
        counti=1
        if (chain==1) & (n_chain>1)
            mu_prior_chain=par_variety_prior[:,1]-chain_dif*par_variety_prior[:,2]
        end
        if (chain==2) | (n_chain==1)
            mu_prior_chain=par_variety_prior[:,1]
        end
        if (chain==3) & (n_chain>=3)
            mu_prior_chain=par_variety_prior[:,1]+chain_dif*par_variety_prior[:,2]
        end
        sd_prior_chain=par_variety_prior[:,2]
        bound_par_lower=par_variety_prior[:,3]
        bound_par_upper=par_variety_prior[:,4]
        for itr=1:n_iteration
            if itr==1
                par_heading[1:n_par]=mu_prior_chain
            end
            if itr>=2
                par_heading[(itr-1)*6+collect(1:n_par)]=par_heading[(itr-2)*6+collect(1:n_par)]
            end
            for i=1:n_par
                sd_sampler=par_variety_prior[:,5] #step 1
                sd_theta_i=sd_sampler[i] # sd sampler  #sd_theta_i=sd_prior_chain[i]
                theta_pre_i=par_heading[(itr-1)*6+collect(1:n_par)] #step2
                mu_theta_pre_i=theta_pre_i[i]#par_heading[(itr-1)*6+i]
                d_gen=Normal(0,1) #mean=0 sd=1
                mu_theta_can_i=(mu_theta_pre_i+rand(d_gen,1)*sd_theta_i)[1]
                theta_can_i=copy(theta_pre_i)
                theta_can_i[i]=mu_theta_can_i
                theta_can_save=theta_can_i
                theta_pre_save=theta_pre_i
                d_prior=Normal(par_variety_prior[i,1],par_variety_prior[i,2])
                d_prior_t=Truncated(d_prior,par_variety_prior[i,3],par_variety_prior[i,4])
                prior_pre=pdf(d_prior_t,mu_theta_pre_i)
                prior_can=pdf(d_prior_t,mu_theta_can_i)
                prior_ratio=prior_can/prior_pre
                diff_pre=SIMRIW_DVI2(data_w,theta_pre_i,par_model)
                diff_can=SIMRIW_DVI2(data_w,theta_can_i,par_model)
                sd_diff_pre=std(diff_pre,corrected=true)
                sd_diff_can=std(diff_can,corrected=true)
                alpha_h_1=exp(loglik(diff_can)-loglik(diff_pre)) #step5
                alpha_h=alpha_h_1*prior_ratio
                if(alpha_h>=1)#step 6
                    theta_update=copy(mu_theta_can_i)
                    sd_diff_acepted=copy(sd_diff_can)
                    md_accepted=sum(abs.(diff_can))/n_obs  #mean of absolute difference
                    case=2
                end
                if(alpha_h<1)
                    d_genu=Uniform(0,1)
                    p_success=rand(d_genu,1)[1]  #notice [1]
                    if(p_success<=alpha_h)
                        theta_update=mu_theta_can_i
                        sd_diff_acepted=sd_diff_can
                        md_accepted=sum(abs.(diff_can))/n_obs
                        case=1
                    end
                    if(p_success>alpha_h)
                        theta_update=mu_theta_pre_i
                        sd_diff_acepted=sd_diff_pre
                        md_accepted=sum(abs.(diff_pre))/n_obs
                        case=0
                    end
                end  #  if(alpha_h<1)
                par_heading[counti]=theta_update
                ra[counti]=case
                rsg[counti]=sd_diff_acepted
                md[counti]=md_accepted
                counti+=1
            end  #for i
        end #for itr
        record_sample[:,chain]=par_heading
        record_accepted[:,chain]=ra
        record_sigma[:,chain]=rsg
        record_md[:,chain]=md
    end # for chain
    return record_sample,record_accepted,record_sigma,record_md
end  #end of function

function DVRt(W,parM,parD) #start from the transplanting date
    parTD=parD[1:3]
    parPD=parD[4:6]
    alpha=parM[1]
    beta=parM[2]
    G=parM[3]
    nDay=size(W)[1]
    Temp=W[:,1]
    DayL=W[:,2]
    dvsSum=0.  #start from the transplanting
    dvsV=zeros(nDay)
    for i=1:nDay
        if ((0.145+0.005*G)<=dvsSum) & (dvsSum<=(0.345+0.005*G))
            dvsi=((DVSft(Temp[i],parTD)^alpha)*
            (DVSfg(DayL[i],parPD)^beta))/G
        end
        if ((0.145+0.005*G)>dvsSum) | (dvsSum>(0.345+0.005*G))
            dvsi=(DVSft(Temp[i],parTD)^alpha)/G
        end
        dvsSum=dvsSum+dvsi
        dvsV[i]=dvsSum
    end
    return dvsV
end
function DVSft(Ti,parT)
    Tb=parT[1]
    To=parT[2]
    Tc=parT[3]
    if (Ti<=Tc) & (Ti>=Tb)
        out=(Ti-Tb)/(To-Tb)*((Tc-Ti)/(Tc-To))^((Tc-To)/(To-Tb))
    end
    if (Ti>Tc) | (Ti<Tb)
        out=0
    end
    return out
end
function DVSfg(Pi,parP)
    Pb=parP[1]
    Po=parP[2]
    Pc=parP[3]
    if (Pi>=Po)
        out=(Pi-Pb)/(Po-Pb)*((Pc-Pi)/(Pc-Po))^((Pc-Po)/(Po-Pb))
    end
    if (Pi<Po)
        out=1
    end
    return out
end

function DVRt_model(dataW,par_variety,par_model)
    extendDay=70
    n=size(dataW,1)  #the number of records
    diff=zeros(n)      #the difference between the observed value and predicted value
    for i=1:n
        dataWs=dataW[i]
        (dataT,dataDL)=[dataWs[:ave_tem],dataWs[:daylength]]#[dataWs[:,:ave_tem],dataWs[:,:daylength]]
        dataWi=hcat(dataT,dataDL)
        obsHD=size(dataT,1)-extendDay  #observed days to heading
        DVI_value=DVRt(dataWi,par_variety,par_model)
        H_days=findall(DVI_value.>=1.)
        if H_days==[]
            preHD=0
        else
            preHD=H_days[1] #predicted daysto heading
        end
        diff[i]=obsHD-preHD
    end
    return(diff)
end

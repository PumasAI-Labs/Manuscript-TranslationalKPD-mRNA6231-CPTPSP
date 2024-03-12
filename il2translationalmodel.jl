using Pumas
using PumasUtilities
using CSV
using DataFramesMeta
using AlgebraOfGraphics
using CairoMakie

il2data = CSV.read("./data/il2finaldataset.csv",
    DataFrame, missingstring=["NA", "."])

il2data = @chain il2data begin
    @rsubset !((:time == 0) && (:dv == 0) && (:evid == 0))
    @rtransform :dv = :evid == 0 ? :dv : missing
end

utimes = unique(il2data.time)

il2pop = read_pumas(il2data,
    observations=[:dv],
    covariates=[:dosegroup])
aa = observations_vs_time(il2pop, separate = true, paginate = true)

@info "IL2 K-PD model"
il2model = @model begin

    @options begin
        checklinear = true
    end
    @param begin
        tvV ∈ RealDomain(lower=0.0, init=168.455013932525,)
        tvCl ∈ RealDomain(lower=0.0, init=13.1626806857056,)
        tvKa ∈ RealDomain(lower=0.0, init=0.0718292068792162,)
        tvslope ∈ RealDomain(lower=0.0, init=4339.68619522804,)
        ecl ∈ RealDomain(lower=0.0, init=0.309455332833419,)
        tvC50 ∈ RealDomain(lower=0.0, init=0.123176390371521,)
        tvCL_cIL2_A2 ∈ RealDomain(lower=0.0, init=1.54847214781172,)
        Ω ∈ PSDDomain(init=[0.26946409 0.028705788; 0.028705788 0.056879909])
        σ ∈ RealDomain(lower=0.0001, init=0.437384839547608)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @covariates dosegroup

    @pre begin
        V = tvV
        CL = tvCl
        Ka = tvKa
        slope = tvslope * exp(η[1])
        C50 = tvC50
        CL_cIL2_A2 = tvCL_cIL2_A2 * exp(η[2]) * (dosegroup / 0.0045)^ecl
    end

    @vars begin
        cp = Central / V
        cIL2 = IL2 / V
    end

    @dynamics begin
        Depot' = -(Depot * Ka)
        Central' = -(CL * cp) + (Depot * Ka)
        IL2' = ((cp * slope) / (cp + C50)) - (CL_cIL2_A2 * cIL2)
    end

    @derived begin
        "HSA-IL2m"
        dv ~ @. Normal(cIL2, abs(cIL2) * σ)
    end
end

@info "Compute initial loglikelihood"

ll = loglikelihood(il2model,
    il2pop,
    init_params(il2model), FOCE())

fi = findinfluential(il2model,
    il2pop,
    init_params(il2model), FOCE())
DataFrame(fi)

@info "Fit using maximum loglikelihood"
il2fit = fit(il2model,
    il2pop,
    init_params(il2model), FOCE())
#
@info "Inference"
il2infer = infer(il2fit)

@info "Inpsection"
il2inspect = inspect(il2fit)

fig1 = goodness_of_fit(il2inspect,
    markercolor=:grey,
    markersize=5,
    ols=false,
    axis=(; title="",
        spinewidth=4,
        rightspinevisible=false,
        topspinevisible=false))
figurelegend(fig1; position=:t, alignmode=Outside(), orientation=:horizontal)
fig1

@info "Visual Predictive Check"
## pcvpc
vpc_il2 = vpc(il2fit,
    prediction_correction=true,
    ensemblealg=EnsembleThreads(),
    bandwidth=60.0, nnodes = 15)
fig2 = vpc_plot(vpc_il2,
    observations=true,
    markercolor=:grey,
    observed_linewidth=3,
    figurelegend=(position=:t, 
                  alignmode=Outside(), 
                  orientation=:horizontal, 
                  nbanks=3),
    markersize=12,
    axis=(yscale=Makie.pseudolog10,
        xticks=0:168:1800,
        yticks=vcat(0:20:100, 200:200:600),
        ylabel="IL2-HSM",
        xlabel="Time (hours)",
        spinewidth=4,
        rightspinevisible=false,
        topspinevisible=false),
    figure=(resolution=(1800, 1400), fontsize=40))
fig2

# individual predictions
## fine grained prediction for individual fits 
indpreds_il2 = [predict(il2fit.model,
    il2fit.data[i],
    il2fit.param;
    obstimes=0:7:maximum(il2fit.data[i].time)) for i in 1:length(il2fit.data)]
fig3_indpreds_il2_sf = subject_fits(indpreds_il2,
    figurelegend = (position=:t, 
                    alignmode=Outside(), 
                    orientation=:horizontal),
    paginate=true,
    separate=true,
    limit=9,
    rows=3,
    columns=3,
    ipred_linewidth=6,
    pred_linewidth=6,
    markersize=16,
    facet=(combinelabels=true,),
    figure=(resolution=(1400, 1000),
        fontsize=28,),
    axis=(xticks=0:168:1600, xticklabelrotation=π / 2,
        ylabel="HSA-IL2m", xlabel="Time (hours)"))
fig3_indpreds_il2_sf[4]

#### Treg model

@info "Treg model"
il2tregmodel = @model begin

    @param begin
        tvV ∈ RealDomain(lower=0.0, init=168.455013932525,)
        tvCl ∈ RealDomain(lower=0.0, init=13.1626806857056,)
        tvKa ∈ RealDomain(lower=0.0, init=0.0718292068792162,)
        tvslope ∈ RealDomain(lower=0.0, init=4339.68619522804,)
        ecl ∈ RealDomain(lower=0.0, init=0.309455332833419,)
        tvC50 ∈ RealDomain(lower=0.0, init=0.123176390371521,)
        tvCL_cIL2_A2 ∈ RealDomain(lower=0.0, init=1.54847214781172,)
        Ωil2 ∈ PSDDomain(init=[0.26946409 0.028705788; 0.028705788 0.056879909])
        # σil2 ∈ RealDomain(lower=0.0001, init=0.437384839547608)

        #Treg

        tvKe0 ∈ RealDomain(lower=0.0, init=0.0122890413556561,)
        tvKin ∈ RealDomain(lower=0.0, init=0.0587412101355083,)
        tvKout ∈ RealDomain(lower=0.0, init=0.0150924840375577,)
        tvEmax ∈ RealDomain(lower=0.0, init=2.86422914456863,)
        tvgam ∈ RealDomain(lower=0.0, init=2.36034187391134,)
        tvEC50 ∈ RealDomain(lower=0.0, init=5.10030961019806,)
        Ωtreg ∈ PSDDomain(init=[0.13159683 -0.1088289; -0.1088289 0.090004069])
        σtreg ∈ RealDomain(lower=0.0001, init=0.27538878330097)

    end

    @options begin
        # subject_t0 = true
        checklinear = true
    end

    @random begin
        ηil2 ~ MvNormal(Ωil2)
        ηtreg ~ MvNormal(Ωtreg)
    end

    @covariates dosegroup form

    @init begin
        Treg = Kin / Kout
    end

    @pre begin
        V = tvV
        CL = tvCl
        Ka = tvKa
        slope = tvslope * exp(ηil2[1])
        C50 = tvC50
        CL_cIL2_A2 = tvCL_cIL2_A2 * exp(ηil2[2]) * (dosegroup / 0.0045)^ecl

        #Treg
        Ke0 = tvKe0
        Kin = tvKin * exp(ηtreg[1])
        Kout = tvKout
        gam = tvgam
        Emax = exp(tvEmax) * exp(ηtreg[2])
        EC50 = exp(tvEC50)
    end

    @vars begin
        cp = Central / V
        cIL2 = IL2 / V
    end

    @dynamics begin
        Depot' = -(Depot * Ka)
        Central' = -(CL * cp) + (Depot * Ka)
        IL2' = ((cp * slope) / (cp + C50)) - (CL_cIL2_A2 * cIL2)
        Ce' = Ke0 * (cIL2 - Ce) + 0.00000001
        Treg' = Kin * (1 + ((Emax * abs(Ce)^gam) / (abs(EC50)^gam + (abs(Ce)^gam)))) - Kout * Treg
    end

    @derived begin
        "%FOXP3+ of CD4+ T cells"
        dv ~ @. Normal(Treg, abs(Treg) * σtreg)
    end
end

@info "treg data"
tregdata = CSV.read("./data/tregfinaldataset.csv",
    DataFrame, missingstring=["NA", "."])
@rsubset! tregdata :id != 2002
tregpop = read_pumas(tregdata,
    observations=[:dv],
    covariates=[:dosegroup, :form])

# Plots
# aa = observations_vs_time(tregpop, separate=true, paginate=true)
@info "Compute initial loglikelihood"

ll = loglikelihood(il2tregmodel,
    tregpop,
    init_params(il2tregmodel), FOCE())

@info "Fit using maximum loglikelihood"
il2tregfit = @time fit(il2tregmodel,
    tregpop,
    init_params(il2tregmodel), FOCE(),
    constantcoef=(tvV=171.5577983523755,
        tvCl=12.865014260196519,
        tvKa=0.07496438054293122,
        tvslope=4570.288914609663,
        ecl=0.3075096768027959,
        tvC50=0.13122923353002033,
        Ωil2=[0.2640513178907819 0.024990985611595563; 0.024990985611595563 0.05249457763745266])
)
#
@info "Inference"
il2treginfer = infer(il2tregfit, Pumas.Bootstrap(;samples=200))

@info "Inpsection"
il2treginspect = inspect(il2tregfit)

# res_il2reg = evaluate_diagnostics(il2treginspect)

#gof plot

fig4 = goodness_of_fit(il2treginspect,
    figurelegend = (position=:t, alignmode=Outside(), orientation=:horizontal),
    markercolor=:grey,
    markersize=5,
    ols=false,
    axis=(; title="",
        spinewidth=4,
        rightspinevisible=false,
        topspinevisible=false))
fig4

## pcvpc
vpc_treg = vpc(il2tregfit,
    prediction_correction=true,
    ensemblealg=EnsembleThreads(),
    bandwidth=50.0)
fig5 = vpc_plot(vpc_treg,
    observations=true,
    markercolor=:grey,
    markersize=12,
    observed_linewidth=3,
    figurelegend = (position=:t, alignmode=Outside(), orientation=:horizontal),
    axis=(yscale=Makie.pseudolog10,
        xticks=0:168:1800,
        yticks=0:10:70,
        ylabel="%FOXP3+ of CD4+ T cells",
        xlabel="Time (hours)",
        spinewidth=4,
        rightspinevisible=false,
        topspinevisible=false
    ),
    figure=(resolution=(1800, 1400), fontsize=40))

fig5



##
indpreds_treg = [predict(il2tregfit.model,
    il2tregfit.data[i],
    il2tregfit.param;
    obstimes=0:7:maximum(il2tregfit.data[i].time)) for i in 1:length(il2tregfit.data)]
fig6_indpreds_treg_sf = subject_fits(indpreds_treg,
    paginate=true,
    separate=true,
    limit=9,
    rows=3,
    columns=3,
    ipred_linewidth=6,
    pred_linewidth=6,
    markersize=16,
    facet=(combinelabels=true,),
    figurelegend = (position=:t, alignmode=Outside(), orientation=:horizontal),
    figure=(resolution=(1400, 1000),
        fontsize=32,),
    axis=(xticks=0:168:1600, xticklabelrotation=π / 2,
        ylabel="%FOXP3+ of CD4+ T cells",
        xlabel="Time (hours)",))
fig6_indpreds_treg_sf[4]

##
# Treg Simulation model
#### Treg model

@info "Treg model"
sim_il2tregmodel = @model begin

    @param begin
        tvV ∈ RealDomain(lower=0.0, init=168.455013932525,)
        tvCl ∈ RealDomain(lower=0.0, init=13.1626806857056,)
        tvKa ∈ RealDomain(lower=0.0, init=0.0718292068792162,)
        tvslope ∈ RealDomain(lower=0.0, init=4339.68619522804,)
        ecl ∈ RealDomain(lower=0.0, init=0.309455332833419,)
        tvC50 ∈ RealDomain(lower=0.0, init=0.123176390371521,)
        tvCL_cIL2_A2 ∈ RealDomain(lower=0.0, init=1.54847214781172,)
        Ωil2 ∈ PSDDomain(init=[0.26946409 0.028705788; 0.028705788 0.056879909])
        # σil2 ∈ RealDomain(lower=0.0001, init=0.437384839547608)

        #Treg

        tvKe0 ∈ RealDomain(lower=0.0, init=0.0122890413556561,)
        tvKin ∈ RealDomain(lower=0.0, init=0.0587412101355083,)
        tvKout ∈ RealDomain(lower=0.0, init=0.0150924840375577,)
        tvEmax ∈ RealDomain(lower=0.0, init=2.86422914456863,)
        tvgam ∈ RealDomain(lower=0.0, init=2.36034187391134,)
        tvEC50 ∈ RealDomain(lower=0.0, init=5.10030961019806,)
        Ωtreg ∈ PSDDomain(init=[0.13159683 -0.1088289; -0.1088289 0.090004069])
        σtreg ∈ RealDomain(lower=0.0001, init=0.27538878330097)

    end

    @options begin
        # subject_t0 = true
        checklinear = true
    end

    @random begin
        ηil2 ~ MvNormal(Ωil2)
        ηtreg ~ MvNormal(Ωtreg)
    end

    @covariates dosegroup

    @init begin
        Treg = Kin / Kout
    end

    @pre begin
        ala = 30.0 # (75/2.5)
        V = tvV * ala
        CL = tvCl * ala^0.65
        Ka = tvKa * ala^-0.25
        slope = tvslope * exp(ηil2[1]) * ala
        C50 = tvC50
        CL_cIL2_A2 = tvCL_cIL2_A2 * exp(ηil2[2]) * (dosegroup / 0.0045)^ecl * ala^0.75

        #Treg
        Ke0 = tvKe0
        Kin = tvKin * exp(ηtreg[1]) * ala^-0.25
        Kout = tvKout * ala^-0.25
        gam = tvgam
        Emax = exp(tvEmax) * exp(ηtreg[2])
        EC50 = exp(tvEC50)
    end

    @vars begin
        cp = Central / V
        cIL2 = IL2 / V
    end

    @dynamics begin
        Depot' = -(Depot * Ka)
        Central' = -(CL * cp) + (Depot * Ka)
        IL2' = ((cp * slope) / (cp + C50)) - (CL_cIL2_A2 * cIL2)
        Ce' = Ke0 * (cIL2 - Ce) + 0.00000001
        Treg' = Kin * (1 + ((Emax * abs(Ce)^gam) / (abs(EC50)^gam + (abs(Ce)^gam)))) - Kout * Treg
    end

    @derived begin
        itreg = @. Treg
        "%FOXP3+ of CD4+ T cells"
        dv ~ @. Normal(itreg, abs(itreg) * σtreg)
    end
end

# setup Simulation


simpop = map(dose -> Subject(id=string(dose, " mg"),
        events=DosageRegimen(dose;),
        covariates=(dosegroup=dose / 70000,)),
    [70, 210, 630, 1260, 2100])
sims = [simobs(sim_il2tregmodel, simpop[1], coef(il2tregfit), obstimes=0:7:1600) for i in 1:500]
# sim_plot(sims, observations = :itreg, separate = true)   
simdf = DataFrame(sims)


vpc_data = @chain simdf begin
    @rsubset :evid != 1
    groupby(:time)
    @combine begin
        :meantreg = mean(:itreg)
        :p05treg = quantile(:itreg, 0.05)
        :p95treg = quantile(:itreg, 0.95)

        :meancIL2 = mean(:IL2)
        :p05cIL2 = quantile(:IL2, 0.05)
        :p95cIL2 = quantile(:IL2, 0.95)
    end
    @rtransform :Day = round((:time / 24.0), digits=2)
end


### il2

#observed data 
obs = CSV.read("data/mRNA6231-P101_PD HSA IL-2_20082021.csv", DataFrame)
@rsubset! obs :DV > 0
@rtransform! obs :Day = round((:Time / 24.0), digits=2)
#% > % filter(DV > 0)


fig7 = Figure(resolution=(1200, 800), fontsize=34)

ax7 = Axis(fig7, xlabel="Time (days)", ylabel="HSM-IL2m", xticks=0:7:70,
    title="Predicted HSM-IL2m Concentrations at 70 μg dose", yscale=Makie.pseudolog10)

lines!(vpc_data.Day, vpc_data.p05cIL2, color=(:maroon, 0.4), linestyle=:dash, linewidth=4, label="5th")
lines!(vpc_data.Day, vpc_data.meancIL2, color=:black, linestyle=:solid, linewidth=4, label="50th")
lines!(vpc_data.Day, vpc_data.p95cIL2, color=(:maroon, 0.4), linestyle=:dash, linewidth=4, label="95th")

band!(vpc_data.Day, vpc_data.p05cIL2, vpc_data.p95cIL2; color=(:green, 0.2))
scatter!(obs.Day, obs.DV, color=(:blue, 0.5), markersize=18)
fig7[1, 1] = ax7
axislegend()

fig7

## treg

#observed data 
obs = CSV.read("data/mRNA6231-P101_PD IPT_26082021.csv", DataFrame)
@rsubset! obs :Units == "%"
@rsubset! obs :Analyte == "Treg (FoxP3+CD25+, CD4+)"
@rtransform! obs :Day = :TAFD
#% > % filter(DV > 0)

fig8 = Figure(resolution=(1200, 800), fontsize=34)

ax8 = Axis(fig8, xlabel="Time (days)", ylabel="%FOXP3+ of CD4+ T cells", xticks=0:7:70,
    title="%FOXP3+ of CD4+ T cells at 70 μg dose", yscale=Makie.pseudolog10)

lines!(vpc_data.Day, vpc_data.p05treg, color=(:maroon, 0.4), linestyle=:dash, linewidth=4, label="5th")
lines!(vpc_data.Day, vpc_data.meantreg, color=:black, linestyle=:solid, linewidth=4, label="50th")
lines!(vpc_data.Day, vpc_data.p95treg, color=(:maroon, 0.4), linestyle=:dash, linewidth=4, label="95th")

band!(vpc_data.Day, vpc_data.p05treg, vpc_data.p95treg; color=(:green, 0.2))
scatter!(obs.Day, obs.Result, color=(:blue, 0.5), markersize=18)
fig8[1, 1] = ax8
axislegend()

fig8


myplots = vcat([fig1...][1], [fig2...][1], [fig3_indpreds_il2_sf...], [fig4...][1],
    [fig5...][1], [fig6_indpreds_treg_sf...], [fig7, fig8])
report(myplots,
    title="HSM-IL2 Manuscript Plots",
    clean=false)

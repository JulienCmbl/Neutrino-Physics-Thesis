' Julia is supposedly the same  as Python but runs code much faster.'
' So the goal was to optimise even more to try to go faster than ~34sec, but even in Julia I did not manage to go below the 30sec'
' Although Julia has a loop time of a couple of seconds, whilst Python takes 30+ seconds '
' So the advantage of Julia here will be found if we re-run the code over and over again '

using HDF5
using LinearAlgebra
using CairoMakie

t_start = time()

const G_F = 1.1663787e-23
const CONV_KM_TO_INV_EV = 5.06773e9
const CONV_CM_TO_INV_EV = CONV_KM_TO_INV_EV * 1e-5
const CONV_GCC_TO_EV = 4.362e18
const m_N_eV = 1.675e-27 * 5.609588e35
const m_tau = 1.77686e9
const m_W = 8.0379e10

const DCP  = 0.0
const dm21 = 7.53e-5
const dm31 = 0.002455

const s12, c12 = sqrt(0.307), sqrt(1 - 0.307)
const s13, c13 = sqrt(0.0219), sqrt(1 - 0.0219)
const s23, c23 = sqrt(0.558), sqrt(1 - 0.558)

const E_nu = 1e7

const eCP  = exp(1im * DCP)
const enCP = exp(-1im * DCP)

const U_PMNS = ComplexF64[
     c12*c13                     s12*c13                     s13*enCP;
    -s12*c23 - c12*s23*s13*eCP    c12*c23 - s12*s23*s13*eCP  s23*c13;
     s12*s23 - c12*c23*s13*eCP   -c12*s23 - s12*c23*s13*eCP  c23*c13
]


const m2_diag = Diagonal(ComplexF64[0.0, dm21, dm31])
const M2_flavour = U_PMNS * m2_diag * U_PMNS'

file_path = ""

r_data, rho_1d, Y_e_1d = h5open(file_path, "r") do f
    r = read(f["r_fine"])
    rho = read(f["rho_fine"])[:, 1, 1, 1] .* CONV_GCC_TO_EV
    Y_e = read(f["y_e_fine"])[:, 1, 1, 1]
    r, rho, Y_e
end

const N = length(r_data)
const n_e = rho_1d .* Y_e_1d ./ m_N_eV
const Y_n = 1.0 .- Y_e_1d


const log_ratio = log((m_W / m_tau)^2) - 1.0

V_mu_tau = @. (3.0 * G_F * m_tau^2) / (2.0 * sqrt(2) * π^2 * Y_e_1d) *
               (log_ratio + Y_n / 3.0)

V_ee = @. sqrt(2) * G_F * n_e
V_tt = @. V_ee * V_mu_tau 

const H_vac = M2_flavour ./ (2.0 * E_nu) 

H_stack = Vector{Matrix{ComplexF64}}(undef, N)
for i in 1:N
    H_stack[i] = copy(H_vac)
    H_stack[i][1, 1] += V_ee[i]
    H_stack[i][3, 3] += V_tt[i]
end

dt = diff(r_data) .* CONV_CM_TO_INV_EV

println("Setup done")
t_loop_start = time()

function evolve_loop(H_stack, dt, psi0::Vector{ComplexF64})
    n_steps = length(dt)
    prob_e = Vector{Float64}(undef, n_steps)
    prob_mu = Vector{Float64}(undef, n_steps)
    prob_tau = Vector{Float64}(undef, n_steps)

    psi = copy(psi0)

    for i in 1:n_steps
        H = H_stack[i]

        F = eigen(Hermitian(H))
        eigenvalues = F.values 
        V = F.vectors

        phase = exp.(-1im .* eigenvalues .* dt[i])
        tmp = V' * psi
        tmp .*= phase
        psi = V * tmp

        prob_e[i] = abs2(psi[1])
        prob_mu[i] = abs2(psi[2])
        prob_tau[i] = abs2(psi[3])
    end

    return prob_e, prob_mu, prob_tau
end

psi0 = ComplexF64[0.0, 0.0, 1.0]
prob_e, prob_mu, prob_tau = evolve_loop(H_stack, dt, psi0)

t_loop_end = time()
println("Evolution loop done in $(round(t_loop_end - t_loop_start, digits=3)) seconds")

distance_km = r_data[2:end] .* 1e-5

fig = Figure(size = (1200, 700))
ax = Axis(fig[1, 1],
    xlabel = "Distance (km)",
    ylabel = "Probability",
    title = "Neutrino Oscillation Probability",
    xscale = log10,
    limits = ((distance_km[1], distance_km[end]), (0.0, 1.0)),
    xlabelsize = 16,
    ylabelsize = 16,
    titlesize = 18,
)

lines!(ax, distance_km, prob_e, label = L"P(\nu_\tau \to \nu_e)",   color = :orange)
lines!(ax, distance_km, prob_mu, label = L"P(\nu_\tau \to \nu_\mu)", color = :blueviolet)
lines!(ax, distance_km, prob_tau, label = L"P(\nu_\tau \to \nu_\tau)",color = :orangered)
axislegend(ax, position = :rt)

display(fig)
#save("neutrino_oscillation.png", fig)

t_end = time()
println("Loop time  : $(round(t_loop_end - t_loop_start, digits=3)) s")
println("Total time : $(round(t_end - t_start, digits=3)) s")

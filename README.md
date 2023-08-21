# Astrophysical GWB

This is a repo with code that's used to estimate the astrophysical gravitational wave background (GWB).
There are multiple ways to try to do this, and we've tried to construct a way to evaluate multiple methods on the same set of
assumptions so that we can compare the different methods.

## Method I: Monte Carlo Integration
The Callister method takes form in four distinct steps:
1. Define the local merger rate.
2. Calculate the merger rate.
3. Determine the mass distribution probability grid.
4. Calculate the GW energy density $\Omega_\text{GW}$.

### Define the local merger rate
The local merger rate is predefined for later merger rate normalization.

### Calculate the merger rate
The merger rate is modelled as follows:
$$\dot{N}(z)=\mathcal{C}(\alpha,\beta,z_\text{p})\frac{\dot{N}_0(1+z)^\alpha}{1+(\frac{1+z}{1+z_\text{p}})^{\alpha+\beta}}$$
$$\mathcal{C}(\alpha,\beta,z_\text{p})=1+(1+z_\text{p})^{-(\alpha+\beta)}$$

An array of merger rates is calculated for each redshift bin of $\text{d}z=0.01$.

### Determine the mass distribution probability grid
A probability grid of the mass distribution is defined in ($m_1,q$) space using `bilby` priors. The probabilities are then converted to $(\text{ln}M_\text{tot},q)$ space with the Jacobian and normalized.

### Calculate the GW energy density $\Omega_\text{GW}$
The GW spectral energy density is calculated with the following equation:
$$\Omega_\text{GW}(f)=\frac{f}{\rho_\text{c}}\int_0^{z_\text{max}}dz\frac{\dot{N}(z)}{(1+z)H(z)}\bigg\langle\frac{dE_\text{GW}}{df_\text{r}}\bigg |_{f_\text{r}=f(1+z)}\bigg\rangle$$
$$\bigg\langle\frac{dE_\text{GW}}{df_\text{r}}\bigg\rangle=\int d\theta p(\theta)\frac{dE_\text{GW}(\theta;f_\text{r})}{df_\text{r}}$$

$\frac{dE}{df}$ is determined for each point in the probability grid:
$$\frac{dE_\text{GW}}{df}=\frac{(G\pi)^{2/3}\mathcal{M}^{5/3}}{3}H(f)$$
$$\mathcal{M}=\frac{(m_1m_2)^{3/5}}{(m_1+m_2)^{1/5}}$$
$$H(f)=
\begin{cases}
    f^{-1/3} && (f<f_\text{merge}) \\
    \frac{f^{2/3}}{f_\text{merge}} && (f_\text{merge}\leq f<f_\text{ring}) \\
    \frac{1}{f_\text{merge}f_\text{ring}^{4/3}}\bigg(\frac{f}{1+(\frac{f-f_\text{ring}}{\sigma/2})^2}\bigg)^2 && (f_\text{ring} \leq f<f_\text{cutoff}) \\
    0 && (f\geq f_\text{cutoff})
\end{cases}
$$

## Method II: Numerical Integration
The numerical integration method calculates the GW energy density as follows:
$$\Omega_\text{GW}(f)=\frac{2}{T_\text{obs}}\sum_0^N\frac{2\pi^2f^3}{3H_0^2}\frac{dE}{df}$$
$$\frac{dE}{df}=|h_+|^2+|h_\times|^2$$
A list of injections is created from `bilby` priors. Instead of integrating over probability grid points, $\frac{dE}{df}$ is calculated for each injection.

## Method III: Combined
The combined method uses the same calculations as the first method but inserts injections for the chirp mass when determining $\frac{dE}{df}$.
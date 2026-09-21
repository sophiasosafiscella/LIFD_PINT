"""
main.py
=======

Fits three chromatic (frequency-dependent) profile-evolution delay models:
 - FD (NANOGrav 15yr log-polynomial),
 - IFD (power-series in inverse frequency), and
 - LIFD (Legendre series in a normalized inverse-frequency axis)
to the same dataset in turn, and produces a comparison plot of the recovered profile-evolution delay vs. observing
frequency for all three.

For each method, this script:
    1. Loads the par/tim files and computes pre-fit residuals.
    2. Optionally overrides the DM value with a previously-determined one.
    3. Swaps in the relevant chromatic-delay component (FD is already present in the par file; IFD/LIFD are added in its
     place).
    4. Runs a weighted least-squares fit.
    5. Extracts the fitted chromatic-delay coefficients and the implied delay curve, and saves both to disk.
"""

import numpy as np
from numpy.polynomial.legendre import legval
from numpy.polynomial.polynomial import polyval
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from pint.fitter import WLSFitter
from pint.models import get_model
from pint.toa import get_TOAs
from pint.residuals import Residuals
from pint import logging
logging.setup("WARNING")
import astropy.units as u
from utils import plot_residuals, get_dmx_params, freeze_parameters, get_FD_delay
import sys

#-----------------------------------------------------------------------------------------------------------
# Configuration
#-----------------------------------------------------------------------------------------------------------

PSR_name: str = "B1937+21"     # Name of the pulsar
plot_fits: bool = False        # Plot the pre-fit and post-fit residuals
order: int = 6                 # Order (number of terms) of the IFD/LIFD polynomial
change_dm: bool = True         # Override the par-file DM with a previously fitted value

# Input files
parfile: str = glob(f"./NANOGrav15yr_PulsarTiming_v2.1.0/narrowband/par/{PSR_name}_PINT_*.nb.par")[0]
timfile: str = glob(f"./NANOGrav15yr_PulsarTiming_v2.1.0/narrowband/tim/{PSR_name}_PINT_*.nb.tim")[0]

# Set up the comparison plot
sns.set_context("paper", font_scale=1.50, rc={"lines.linewidth": 2.5})
fig, ax = plt.subplots(1, 1)
ax2 = ax.twinx()  # second y-axis, shared x-axis
colors = plt.colormaps['Paired'](range(8))
colors = [colors[6], colors[7], colors[0], colors[1], colors[2], colors[3]]

#-----------------------------------------------------------------------------------------------------------
# Fit each chromatic-delay model in turn
#-----------------------------------------------------------------------------------------------------------

for i, method in enumerate(["FD", "IFD", "LIFD"]):

    print(f"Running {method}")


    timing_model = get_model(parfile)  # Ecliptic coordinates
    toas = get_TOAs(timfile, planets=True, ephem=timing_model.EPHEM.value)  # Load TOAs

    if plot_fits:
        # Pre-fit residuals, with no model changes yet
        res_pint = Residuals(toas, timing_model, subtract_mean=False).time_resids.to(u.us).value
        errs_pint = toas.get_errors().value
        freqs_pint = toas.get_freqs().value
        toas_mjd_pint = toas.get_mjds().value

        fig_aux, ax_aux = plot_residuals(toas_mjd_pint, res_pint, freqs_pint)
        ax_aux.set_title("Pre-fit", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"./results/{method}_simulation_nofreqeq_residuals_prefit.png")
        plt.show()
        plt.close(fig_aux)

    # We'll use these to store the dispersive delay, profile-evolution delay, and their sum
    new_dmx_dispersion_delays_us = np.zeros(toas.ntoas)
    prof_evol_delay_us = np.zeros(toas.ntoas)
    total_delay = np.zeros(toas.ntoas)

    # -----------------------------------------------------------------------------------------------------------
    # Optionally override the DM so that the epoch-wise corrections are smaller (see DM_search.py).
    # -----------------------------------------------------------------------------------------------------------
    if change_dm:
        DM_value = np.load(f"./results/{PSR_name}/new_DM.npy").item()
        DM_param = {"DM": (DM_value * timing_model.DM.units, 1, 0 * timing_model.DM.units)}

        for name, info in DM_param.items():
            par = getattr(timing_model, name)  # Get parameter object from name
            par.value = info[0]                # Set parameter value
            if info[1] == 1:
                par.frozen = False             # info[1] == 1 means "let this be fit"
            par.uncertainty = info[2]          # Set parameter uncertainty

    # Fix the DM value so that it is not included in the timing fit
    getattr(timing_model, 'DM').frozen = True

    #-----------------------------------------------------------------------------------------------------------
    # Swap in the IFD or LIFD chromatic-delay component (FD is already present in the par file)
    #-----------------------------------------------------------------------------------------------------------
    if method == "IFD" or method == "LIFD":
        timing_model.remove_component("FD")

        # LIFD defines its λ→x mapping at setup(), which requires TOAs, so attach them to the model beforehand.
        timing_model.toas = toas

        if method == "IFD":
            from IFD_class import IFD
            timing_model.add_component(IFD(order=order))
            freeze_parameters(timing_model, ['IFD0'])  # AIC indicates this term does not matter

        elif method == "LIFD":
            from LIFD_class import LIFD
            timing_model.add_component(LIFD(order=order))

    #-----------------------------------------------------------------------------------------------------------
    # Fit the timing model
    #-----------------------------------------------------------------------------------------------------------
    f = WLSFitter(toas, timing_model)
    f.fit_toas()
    fitted_model = f.model

    if plot_fits:
        fig_aux, ax_aux = plot_residuals(toas_mjd_pint, f.resids.time_resids, freqs_pint)
        ax_aux.set_title(f"Post-Fit | {method}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"./results/{method}_simulation_nofreqeq_residuals_postfit.png")
        plt.show()
        plt.close(fig_aux)

    # -----------------------------------------------------------------------------------------------------------
    # Extract the fitted chromatic-delay coefficients and the implied delay curve
    # -----------------------------------------------------------------------------------------------------------
    freq_GHz = toas.get_freqs().to(u.GHz)
    freq_rounded = np.round(freq_GHz, decimals=3).value
    unique_freqs = np.unique(freq_rounded)  # Sorted, de-duplicated channel frequencies
    freq_GHz = unique_freqs * u.GHz

    if method == "FD":
        x_var = np.sort(freq_GHz.value)
        ax.set_xlabel("$\\nu$ [GHz]")

        FD_coeffs = [getattr(fitted_model, FD_param).value for FD_param in
                     fitted_model.components['FD'].params]  # seconds
        np.save(f"./results/{PSR_name}/FD_coeffs.npy", FD_coeffs)

        # get_FD_delay returns microseconds already (see utils.py)
        prof_evol_delay_us = get_FD_delay(FD_coeffs, x_var)

    elif method == "IFD":
        x_var = np.sort(IFD.get_lambda_ns_from_freq(freq_GHz))  # Inverse frequencies
        ax.set_xlabel("$\lambda$")

        IFD_coeffs = [getattr(fitted_model, f"IFD{deg}").value for deg in range(0, order)]
        np.save(f"./results/{PSR_name}/IFD_coeffs.npy", IFD_coeffs)

        # Coefficients are in seconds, so the polynomial evaluates to seconds -> convert to us
        prof_evol_delay_us = (polyval(x=x_var, c=IFD_coeffs) * u.second).to(u.us)

    elif method == "LIFD":
        lambdas_sec = LIFD.get_lambda_sec_from_freq(freq_GHz)  # Inverse frequencies, in seconds
        lifd_comp = fitted_model.components["LIFD"]
        x_var = np.sort(lifd_comp.map_lambda_to_unit(lambdas_sec, lifd_comp.lmin, lifd_comp.lmax))
        ax.set_xlabel("x")

        LIFD_coeffs = [getattr(fitted_model, f"LIFD{i}").value for i in range(0, order)]
        np.save(f"./results/{PSR_name}/LIFD_coeffs.npy", LIFD_coeffs)

        prof_evol_delay_us = (legval(x_var, LIFD_coeffs) * u.second).to(u.us)

    else:
        print("Method not recognized")
        sys.exit(1)

    # Dispersive (DMX) delay at each unique frequency, mean-subtracted for comparability with prof_evol_delay_us
    new_DMX_component = fitted_model.components["DispersionDMX"]
    new_dmx_dispersion_delays_us = new_DMX_component.DMX_dispersion_delay(toas).to(u.us)
    new_dmx_dispersion_delay_means = np.array([new_dmx_dispersion_delays_us.value[freq_rounded == f].mean() for f in unique_freqs])
    new_dmx_dispersion_delay_means -= np.mean(new_dmx_dispersion_delay_means)

    #-----------------------------------------------------------------------------------------------------------
    # Save the fitted DMX parameters
    #-----------------------------------------------------------------------------------------------------------
    DMX_params = get_dmx_params(fitted_model)
    DMX_params.to_pickle(f"./results/{PSR_name}/{method}_DMX.pkl")

    # -----------------------------------------------------------------------------------------------------------
    # Compute and plot the total chromatic delay (profile evolution + dispersion) vs. frequency
    # -----------------------------------------------------------------------------------------------------------
    total_delay = prof_evol_delay_us.value + new_dmx_dispersion_delay_means
    total_delay -= np.mean(total_delay)

    # Round frequencies to nearest channel to group identical frequencies
    freq_rounded = np.round(freq_GHz, decimals=3).value
    unique_freqs = np.unique(freq_rounded)

    # Detect gaps in frequency coverage so the plotted line doesn't bridge them
    gap_threshold = 0.05  # GHz -- adjust based on receiver channel spacing
    diffs = np.diff(unique_freqs)
    gap_indices = np.where(diffs > gap_threshold)[0] + 1

    # Insert NaNs at gap locations
    freqs_with_nans = np.insert(unique_freqs, gap_indices, np.nan)

    total_delays_means = np.array([total_delay[freq_rounded == f].mean() for f in unique_freqs])
    total_delays_means_with_nans = np.insert(total_delays_means, gap_indices, np.nan)

    total_delays_stds = np.array([total_delay[freq_rounded == f].std() for f in unique_freqs])
    total_delays_stds_with_nans = np.insert(total_delays_stds, gap_indices, np.nan)

    prof_evol_delay_means = np.array([prof_evol_delay_us.value[freq_rounded == f].mean() for f in unique_freqs])
    prof_evol_delay_means -= np.mean(prof_evol_delay_means)
    prof_evol_delay_means_with_nans = np.insert(prof_evol_delay_means, gap_indices, np.nan)

    new_dmx_dispersion_delay_means_with_nans = np.insert(new_dmx_dispersion_delay_means, gap_indices, np.nan)

    # Total delay (for this method), with shaded +/- 1 std band
    ax.plot(freqs_with_nans, total_delays_means_with_nans, ls='-', color=colors[2 * i + 1], label=f"{method}",
            alpha=0.6)
    ax.fill_between(freqs_with_nans,
                    total_delays_means_with_nans - total_delays_stds_with_nans,
                    total_delays_means_with_nans + total_delays_stds_with_nans, color=colors[2 * i + 1], alpha=0.1)

    # Individual components (profile evolution, DMX), on the secondary axis
    ax2.plot(freqs_with_nans, prof_evol_delay_means_with_nans, ls='--', color=colors[2 * i])
    ax2.plot(freqs_with_nans, new_dmx_dispersion_delay_means_with_nans, ls=':', color=colors[2 * i])

#-----------------------------------------------------------------------------------------------------------
# Finalize and save the comparison plot
#-----------------------------------------------------------------------------------------------------------
ax.set_xlabel("$\\nu$ [GHz]")
ax.set_ylabel("Total delay [$\mu$s]")
ax2.set_ylabel("Individual components delays [$\mu$s]")

ax.plot([], [], ls='-', color='black', label=f"Total delay")
ax.plot([], [], ls='--', color='black', alpha=0.3, label=f"Profile evolution")
ax.plot([], [], ls=':', color='black', alpha=0.3, label=f"DMX")

ax.legend(ncol=2)
plt.tight_layout()
plt.savefig(f"./figures/{PSR_name}_profile_evl_and_DMX.pdf")
plt.show()

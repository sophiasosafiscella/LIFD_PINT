"""
utils.py
========

Helper functions for fitting and visualizing chromatic (frequency-dependent)
timing delay models -- FD, IFD, and LIFD -- against PINT timing models.

Roughly grouped into:
    - DMX-window bookkeeping (get_dmx_ranges, get_dmx_observations, get_dmx_params)
    - Frequency-axis mapping (map_domain, reverse_mapping)
    - Data assembly for GP-style fits (get_data, DataObject)
    - Plotting (plot_all_coeffs_fit, plot_FD_curve, plot_residuals)
    - Misc (epoch_scrunch, get_FD_curve_values, get_FD_delay, freeze_parameters)
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.units import Quantity
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.linalg import cho_factor
from numpy.polynomial.polynomial import Polynomial
from pint.residuals import Residuals
from pypulse.utils import weighted_moments
from pypulse.par import Par
import astropy.units as u
from dataclasses import dataclass


@dataclass
class DataObject:
    """
    Container for the broadband data products needed to fit a chromatic
    delay model with a Gaussian-process-style noise model. Populated by
    `get_data()`.
    """
    PSR_name: str
    dmx_ranges: np.ndarray            # (Nwindows, 2) array of [MJD_start, MJD_end] per DMX window
    max_inv_freq: float               # max of 1/freq [GHz^-1] across broadband TOAs
    min_inv_freq: float               # min of 1/freq [GHz^-1] across broadband TOAs
    logdet_C: float                   # log-determinant of the full noise covariance matrix
    Ndiag: np.ndarray                 # diagonal of the white-noise covariance (TOA uncertainties^2, in s^2)
    U: np.ndarray                     # noise-model (e.g. ECORR/red-noise) design matrix
    Sigma_cf: tuple                   # Cholesky factorization of Sigma (see get_data), from scipy.linalg.cho_factor
    xvals: list[np.ndarray]           # per-DMX-window normalized frequency axis values, x in [-1, 1]
    freqs: list[np.ndarray]           # per-DMX-window observing frequencies [GHz]
    resids: list[np.ndarray]          # per-DMX-window timing residuals [us]


def get_dmx_ranges(model, mjds):
    """
    Return the (Nwindows, 2) array of [MJD_start, MJD_end] for each DMX
    window that contains at least one of the given `mjds`.
    """
    dmx_model = model.components['DispersionDMX']  # names of the DMX_xxxx parameters

    # MJD range [DMXR1, DMXR2] for every DMX window in the model
    DMXR = np.fromiter(((dmx_model._parent[f"DMXR1_{idx:04d}"].value, dmx_model._parent[f"DMXR2_{idx:04d}"].value)
                        for idx in dmx_model.get_indices()), dtype=np.dtype((float, 2)))

    # Keep only windows that actually contain at least one TOA
    mask = np.any(((DMXR[:, 0, None] <= mjds) & (mjds <= DMXR[:, 1, None])), axis=1)

    return DMXR[mask]


def get_dmx_observations(observations, low_mjd, high_mjd):
    """
    Return the subset of `observations` (a PINT TOA object) whose MJDs fall
    strictly inside (low_mjd, high_mjd) -- TOAs exactly on a bin edge are
    excluded.
    """
    mjds = observations.get_mjds().value
    mask = (low_mjd < mjds) & (mjds < high_mjd)

    return observations[mask]


def map_domain(frequencies, max_inv_freq, min_inv_freq):
    """
    Map observing frequencies to a normalized axis x in [-1, 1], via
    λ = 1/frequencies affinely scaled by the given (min, max) inverse-
    frequency range. `frequencies`, `max_inv_freq`, and `min_inv_freq` must
    all be in consistent units (e.g. GHz / GHz^-1).
    """
    lambdas = np.power(frequencies, -1.0)                                # Inverse of the frequencies
    x_aux_values = (lambdas - min_inv_freq)/(max_inv_freq-min_inv_freq)  # Between 0 and 1
    x_values = np.subtract(np.multiply(x_aux_values, 2.0), 1.0)          # Between -1 and 1

    return np.asarray(x_values, dtype=np.float64)


def reverse_mapping(x_values, max_inv_freq, min_inv_freq):
    """
    Inverse of `map_domain`: convert normalized axis values x in [-1, 1]
    back to observing frequencies, given the same (min, max) inverse-
    frequency range used to construct x.
    """
    aux = np.divide(np.add(x_values, 1.0), 2.0)                 # Between 0 and 1
    lambdas = aux * (max_inv_freq-min_inv_freq) + min_inv_freq  # Inverse of the frequencies
    frequencies = np.power(lambdas, -1.0)

    return frequencies


def get_data(psr_name, toas, timing_model):
    """
    Given TOAs and a timing model, extract broadband observations (i.e.
    drop narrowband-only receivers), strip the DMX and FD components from
    the model to get "raw" chromatic residuals, and assemble the noise
    covariance pieces (Ndiag, U, Sigma_cf, logdet_C) needed for a GP-style
    fit. Residuals, frequencies, and normalized x-values are grouped by
    DMX window.

    Note: this mutates `timing_model` in place by removing its DispersionDMX
    and FD components.

    Returns
    -------
    DataObject
    """
    # Filter out narrowband-only receivers (GASP/ASP)
    backends = np.array([toas.table["flags"][obs]["be"] for obs in range(len(toas.table["flags"]))])
    broadband_TOAs = toas[~np.isin(backends, ["GASP", "ASP"])]

    # Extract relevant information from the broadband TOAs
    broadband_mjds = broadband_TOAs.get_mjds().value                     # MJDs of the broadband TOAs
    broadband_dmx_ranges = get_dmx_ranges(timing_model, broadband_mjds)  # DMX windows containing broadband TOAs
    freqs_GHz = broadband_TOAs.get_freqs().to(u.GHz).value               # Frequencies in GHz
    inverse_freqs = np.power(freqs_GHz, -1)                              # Inverse frequencies in GHz^-1
    max_inv_freq, min_inv_freq = np.amax(inverse_freqs), np.amin(inverse_freqs)
    full_xvals = map_domain(freqs_GHz, max_inv_freq, min_inv_freq)       # Inverse frequencies, mapped to [-1, 1]

    # Strip the DMX and FD parameters to get the simplified (chromatic-
    # delay-free) timing model
    timing_model.remove_component("DispersionDMX")
    timing_model.remove_component("FD")

    # Residuals under the simplified model
    res_object = Residuals(broadband_TOAs, timing_model)
    residuals = np.asarray(res_object.time_resids.to(u.us).value, dtype=np.float64)

    # Noise covariance pieces. See Eq. 13 of
    # https://iopscience.iop.org/article/10.3847/1538-4357/ad59f7/pdf
    Ndiag = res_object.model.scaled_toa_uncertainty(broadband_TOAs).to_value(u.s) ** 2
    U = res_object.model.noise_model_designmatrix(res_object.toas)
    Phidiag = res_object.model.noise_model_basis_weight(res_object.toas)

    Sigma = np.diag(1.0 / Phidiag) + (U.T / Ndiag) @ U
    Sigma_cf = cho_factor(Sigma)

    logdet_N = np.sum(np.log(Ndiag))
    logdet_Phi = np.sum(np.log(Phidiag))
    _, logdet_Sigma = np.linalg.slogdet(Sigma.astype(float))
    logdet_C = np.asarray(logdet_N + logdet_Phi + logdet_Sigma, dtype=np.float64)

    # Group residuals/x-values/frequencies by DMX window
    window_masks = [(broadband_mjds > window[0]) & (broadband_mjds < window[1]) for window in broadband_dmx_ranges]
    resids = [residuals[mask] for mask in window_masks]
    xvals = [full_xvals[mask] for mask in window_masks]
    freqs = [freqs_GHz[mask] for mask in window_masks]

    return DataObject(PSR_name=timing_model.PSR.value, dmx_ranges=broadband_dmx_ranges,
                       max_inv_freq=max_inv_freq, min_inv_freq=min_inv_freq,
                       logdet_C=logdet_C, Ndiag=Ndiag, U=U, Sigma_cf=Sigma_cf,
                       xvals=xvals, freqs=freqs, resids=resids)


def plot_all_coeffs_fit(PSR_name, df):
    """
    Plot the fitted monomial coefficients b0..b5 (in x) versus DMX window
    center, one subplot per coefficient, with the across-window mean
    overplotted. Assumes a 5th-order polynomial fit (six coefficients,
    a0..a5) and saves the figure to '<PSR_name>_results.png'.
    """
    windows_centers = df["DMXR1"] + (df["DMXR2"] - df["DMXR1"]) / 2.0

    sns.set_style("ticks")
    sns.set_context("paper", font_scale=3.0)
    fig, ax = plt.subplots(nrows=6, ncols=1, figsize=(12, 24), sharex=True,
                           gridspec_kw={'hspace': 0})
    fig.suptitle(PSR_name + " - Monomial Coefficients in $x$ \n $t_\\nu = b_0 + b_1 x + b_2 x^2 + b_3 x^3 + b_4 x^4 + b_5 x^5$")

    means = df[['a0', 'a1', 'a2', 'a3', 'a4', 'a5']].mean(axis=0)
    print(f"a1 = {means['a1']}")
    print(f"a3 = {means['a3']}")
    print(f"a5 = {means['a5']}")

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    for i in range(6):
        ax[i].scatter(windows_centers, df[f'a{i}'], color=colors[i])
        ax[i].axhline(y=means[f'a{i}'], color='black', lw=4, linestyle='--')
        ax[i].set_ylabel(f"$b_{i} [ns]$")
        ax[i].grid(True)
        ax[i].label_outer()  # Hide inner x labels/ticks shared across subplots

        ax[i].text(0.2, 0.1, f'Mean = {round(means[f"a{i}"], 4)}', horizontalalignment='center', verticalalignment='center',
                 transform=ax[i].transAxes)

    ax[5].set_xlabel("Window Center [MJD]")

    plt.tight_layout()
    plt.savefig('./' + PSR_name + '_results.png')
    plt.show()


def epoch_scrunch(toas, data=None, errors=None, epochs=None, decimals=0, getdict=False, weighted=False, harmonic=False):
    """
    Group TOAs (and optionally associated data/errors) into epochs by
    rounding to `decimals` decimal places, merging consecutive epochs that
    are closer together than the rounding resolution.

    If `data` is None, only the array of epoch MJDs is returned. Otherwise,
    `data` is averaged within each epoch (weighted by 1/errors**2 if
    `weighted=True` and `errors` is given) and (epochs, values, errors) is
    returned -- or a dict keyed by epoch if `getdict=True`.

    Note: kept for backwards compatibility with earlier pipeline scripts;
    not currently called elsewhere in this codebase.
    """
    if epochs is None:
        epochsize = 10 ** (-decimals)
        bins = np.arange(np.around(min(toas), decimals=decimals) - epochsize,
                         np.around(max(toas), decimals=decimals) + 2 * epochsize,
                         epochsize)  # 2 allows for the extra bin to get chopped by np.histogram
        freq, bins = np.histogram(toas, bins)
        validinds = np.where(freq != 0)[0]

        epochs = np.sort(bins[validinds])
        diffs = np.array(list(map(lambda x: np.around(x, decimals=decimals), np.diff(epochs))))
        epochs = np.append(epochs[np.where(diffs > epochsize)[0]], [epochs[-1]])
    else:
        epochs = np.array(epochs)
    reducedTOAs = np.array(list(map(lambda toa: epochs[np.argmin(np.abs(epochs - toa))], toas)))

    if data is None:
        return epochs

    Nepochs = len(epochs)

    if weighted and errors is not None:
        averaging_func = lambda x, y: weighted_moments(x, 1.0 / y ** 2, unbiased=True, harmonic=harmonic)
    else:
        averaging_func = lambda x, y: (np.mean(x), np.std(y))

    if getdict:
        retval = dict()
        retvalerrs = dict()
    else:
        retval = np.zeros(Nepochs)
        retvalerrs = np.zeros(Nepochs)
    for i in range(Nepochs):
        epoch = epochs[i]
        inds = np.where(reducedTOAs == epoch)[0]
        if getdict:
            retval[epoch] = data[inds]
            if errors is not None:
                retvalerrs[epoch] = errors[inds]
        else:
            if errors is None:
                retval[i] = np.mean(data[inds])
                retvalerrs[i] = np.std(data[inds])
            else:
                retval[i], retvalerrs[i] = averaging_func(data[inds], errors[inds])
    if getdict and errors is None:
        return epochs, retval
    return epochs, retval, retvalerrs


def get_FD_curve_values(p, freqs, DM0=0.0):
    """
    Evaluate a libstempo/PSRCHIVE-style FD model (read from a `Par` object
    `p`) on a dense frequency grid spanning the range of `freqs`, mean-
    subtracted. Returns (grid_freqs, delay_values_in_us), or None if `p`
    has no FD model.
    """
    FDfunc = p.getFDfunc()  # Frequency is in GHz, returns values in microseconds
    if FDfunc is None:
        return

    F1s = np.amin(freqs)
    F2s = np.amax(freqs)
    fs = np.arange(F1s, F2s, 0.001)

    ys = FDfunc(fs)
    ys -= np.mean(ys)

    return fs, ys


def plot_FD_curve(PSR_name, parfile, data_obj, samples):
    """
    Plot the fitted odd-order IFD profile-evolution curve (a1*x + a3*x^3 +
    a5*x^5, using the posterior median of `samples`) alongside the
    NANOGrav 15yr FD curve read from `parfile`, as a function of observing
    frequency. Saves to './results/<PSR_name>/<PSR_name>_FD_curve.png'.
    """
    fig, ax = plt.subplots()
    fig.suptitle(PSR_name)

    a1a3a5 = np.median(samples, axis=0)  # Posterior median coefficients
    poly = Polynomial([0.0, a1a3a5[0], 0.0, a1a3a5[1], 0.0, a1a3a5[2]])  # Odd powers only: a1 x + a3 x^3 + a5 x^5
    x_vals = np.arange(-1.0, 1.0, 0.001)  # Normalized inverse-frequency axis
    ys = poly(x_vals)
    ys -= np.mean(ys)

    # Convert the normalized x-axis back to physical frequency (GHz)
    frequencies = reverse_mapping(x_vals, data_obj.max_inv_freq, data_obj.min_inv_freq)
    ax.plot(frequencies, ys * 10**(-3), label="$a_1 x + a_3 x^3 + a_5 x^5$")

    # NANOGrav 15yr FD model, for comparison
    p = Par(parfile, numwrap=float)
    DM = p.getDM()
    fs, ys_FD = get_FD_curve_values(p, frequencies, DM0=DM)
    ax.plot(fs, ys_FD * 10**(-3), 'k', label="NG15's FD model")

    F1, F2 = frequencies[0], frequencies[-1]
    Fdiff = F2 - F1
    ax.set_xlim(F1 - 0.1 * Fdiff, F2 + 0.1 * Fdiff)

    ax.set_xlabel(r"Frequency (GHz)")
    ax.set_ylabel(r"Residual ($\mu$s)")

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./results/{PSR_name}/{PSR_name}_FD_curve.png")
    plt.show()


def plot_residuals(toas, residuals, freqs, toaerrs=None, title=None, errs_smoothened=False):
    """
    Plot timing residuals (in microseconds) versus calendar year, optionally
    color-coded by observing frequency and/or with error bars.

    Parameters
    ----------
    toas : array-like
        TOAs in MJD.
    residuals : array-like
        Timing residuals, in microseconds.
    freqs : array-like or None
        Observing frequencies (MHz); if given, points are colored by
        frequency with a colorbar.
    toaerrs : array-like or None
        TOA uncertainties, in microseconds; if given, plotted as error bars.
    title : str or None
        Plot title.
    errs_smoothened : bool or int
        If an int N is given (and `toaerrs` is provided), points with
        uncertainty greater than N standard deviations above the mean
        uncertainty are dropped before plotting. If False (default), no
        filtering is applied.

    Returns
    -------
    (fig, ax) : the created matplotlib Figure and Axes.
    """
    toas = np.asarray(toas)
    residuals = np.asarray(residuals)

    if freqs is not None:
        freqs = np.asarray(freqs)

    if toaerrs is not None:
        toaerrs = np.asarray(toaerrs)

    if toaerrs is not None and type(errs_smoothened) == int:
        errs_std = np.std(toaerrs)
        indexes = np.where(toaerrs < errs_smoothened * errs_std)

        residuals = residuals[indexes]
        toas = toas[indexes]
        toaerrs = toaerrs[indexes]

        if freqs is not None:
            freqs = freqs[indexes]

    # Convert MJD to fractional calendar year (1858.878 ~= the decimal year
    # corresponding to MJD 0, i.e. 1858-11-17)
    years = 1858.878 + toas / 365.25

    fig, ax = plt.subplots(figsize=(10, 6))

    if freqs is not None:
        vmin = freqs.min() - freqs.min() * 0.1
        vmax = freqs.max() + freqs.max() * 0.1

        cmap = LinearSegmentedColormap.from_list("", ["blue", "yellow"])
        norm = Normalize(vmin=vmin, vmax=vmax)
        colors = cmap(norm(freqs))

        if toaerrs is not None:
            ax.errorbar(years, residuals, yerr=toaerrs, fmt="none", ecolor=colors, elinewidth=1.1)
        else:
            ax.errorbar(years, residuals, yerr=toaerrs, fmt=".", ecolor="blue", label="Timing Residuals")

        sc = ax.scatter(years, residuals, c=freqs, cmap=cmap, norm=norm, label="Timing Residuals")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Frequency (MHz)", fontsize=12)

    elif toaerrs is not None:
        ax.errorbar(years, residuals, yerr=toaerrs, fmt=".", ecolor="blue", label="Timing Residuals")

    else:
        ax.scatter(years, residuals, edgecolor="k", label="Timing Residuals")

    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Residuals (μs)", fontsize=12)

    if title is not None:
        ax.set_title(title, fontsize=14)

    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    return fig, ax


def freeze_parameters(model, params_to_freeze):
    """Zero out, freeze, and set uncertainty to 0 for a list of parameter names."""
    for p in params_to_freeze:
        param = getattr(model, p)
        param.value = 0.0
        param.uncertainty_value = 0.0
        param.frozen = True


def get_FD_delay(FD_params, freqs_GHz):
    """
    Evaluate the NANOGrav-style FD (logarithmic-polynomial) profile-
    evolution delay at the given frequencies.

    Delay(ν) = 1e6 * Σ_k FD_params[k] * (ln ν[GHz])^(k+1)   [microseconds]

    (the trailing zero constant term means there is no contribution at
    ln ν = 0, i.e. at 1 GHz).
    """
    FD_params = FD_params[::-1]  # np.polyval expects highest-order coefficient first
    FD_params = np.concatenate((FD_params, [0]))  # constant term = 0, see https://numpy.org/doc/stable/reference/generated/numpy.polyval.html
    FD_func = lambda nu: np.polyval(FD_params, np.log(nu))  # seconds -> microseconds
    FD_delays = (FD_func(freqs_GHz) * u.s).to(u.us)
    return FD_delays


def get_dmx_params(timing_model):
    """
    Collect the value and uncertainty of every DispersionDMX parameter in
    `timing_model` into a DataFrame indexed by parameter name.
    """
    dmx_params = timing_model.components["DispersionDMX"].params
    names = []
    values = np.empty(len(dmx_params))
    errors = np.full(len(dmx_params), np.nan) * u.pc / u.cm**3

    for i, par in enumerate(dmx_params):
        names.append(getattr(timing_model, par).name)
        values[i] = getattr(timing_model, par).value

        if isinstance(getattr(timing_model, par).uncertainty, Quantity):
            errors[i] = getattr(timing_model, par).uncertainty
        else:
            errors[i] = np.nan * u.pc / u.cm**3

    return pd.DataFrame({'Value': values, 'Error': errors}, index=names)
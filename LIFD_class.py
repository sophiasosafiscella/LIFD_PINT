from pint.models.timing_model import DelayComponent
from pint.models.parameter import prefixParameter
import astropy.units as u
import numpy as np

class LIFD(DelayComponent):
    """
    LIFD chromatic delay:
      Δt(ν, t) =  Σ_{k=0..N-1}  c_k L_k(x)

    - Global Legendre terms c_k on x ∈ [-1,1], where x is a *fixed* affine map of λ=1/ν.

    Notes:
      * The λ→x mapping is fixed at setup() using the model's current TOAs, so c_k have a stable definition.
    """

    register = True
    category = "LIFD"

    def __init__(self, order=4):
        """
        Parameters
        ----------
        order : int
            Highest Legendre degree.
        """

        super().__init__()
        if order < 0:
            raise ValueError("order must be >= 0")
        self.order = int(order)

        # Global Legendre coefficients c_0..c_order (seconds)
        for deg in range(0, self.order):
            self.add_param(
                prefixParameter(
                    name=f"LIFD{deg}",
                    units="second",
                    value=0.0,
                    description=f"Global Inverse Legendre FD coefficient degree {deg}",
                    type_match="float",
                    convert_tcb2tdb=False,
                    frozen=False,  # <--- important
                )
            )

        self.delay_funcs_component += [self._delay]


 # ------------------------------------------------ Helper functions ------------------------------------------------

    @staticmethod
    def get_freq_hz_from_toas(model, toas):
        tbl = toas.table
        try:
            bfreq = model.barycentric_radio_freq(toas)  # astropy Quantity
            return bfreq.to(u.Hz)
        except Exception:
            # Fallback to topocentric freq column
            col = tbl["freq"]
            return (col * u.MHz).to(u.Hz) if getattr(col, "unit", None) is None else col.to(u.Hz)

    @staticmethod
    def get_lambda_sec_from_freq(freqs):
        # Input: frequencies (in any unit)
        # Output: inverse frequencies (in seconds)
        # λ = 1/ν  would have units of seconds, but here we use it as a scalar axis for mapping only.
        freq_hz = freqs.to(u.Hz).value
        return np.power(freq_hz, -1.0)

    @staticmethod
    def _legendre_val(deg, x):
        coeffs = np.zeros(deg + 1, dtype=float)    # Create a Legendre polynomial where all the coefficients are zero
        coeffs[deg] = 1.0                          # except for the coefficient c_k of the degree k in question
        return np.polynomial.legendre.legval(x, coeffs)

    @staticmethod
    def map_lambda_to_unit(lambdas, lambda_min, lambda_max):
        # Map to [-1, 1] using global min/max values over the *total* array of lambdas over ALL the TOAs
        if not np.isfinite(lambda_min) or not np.isfinite(lambda_max) or lambda_max == lambda_min:
            # degenerate case; just return zeros
            return np.zeros_like(lambdas)
        else:
            # Equivalent to 2*((lambdas - min_inv_freq)/(max_inv_freq-min_inv_freq)) - 1
            return 2.0 * (lambdas - 0.5 * (lambda_min + lambda_max)) / (lambda_max - lambda_min)


    # ------------------------------------------------ PINT functions ------------------------------------------------
    def setup(self):
        """Fix λ→x mapping."""
        super().setup()

        # Define a FIXED λ→x mapping using model TOAs
        toas = getattr(self._parent, "toas", None)
        if toas is None:
            raise ValueError("Parent model has no TOAs attached at setup().")

        # Set lambda_min, lambda_max from ALL TOAs
        freq_hz = self.get_freq_hz_from_toas(self._parent, toas)  # Frequencies in Hz
        lambdas = self.get_lambda_sec_from_freq(freq_hz)  # Inverse frequencies (would be in seconds)
        self.lmin = lambdas.min()  # lambda_min
        self.lmax = lambdas.max()  # lambda_max


        # Register derivative functions:
        # - Global Legendre coefficients LIFDk: derivative = L_k(x) (for all TOAs)
        for deg in range(0, self.order):
            self.register_deriv_funcs(self._deriv_global_LFDk, f"LIFD{deg}")

    def validate(self):
        super().validate()
        pass

    # This is the core delay!
    def _delay(self, toas, acc_delay=None):
        tbl = toas.table

        freq_hz = self.get_freq_hz_from_toas(self._parent, toas)
        lambdas = self.get_lambda_sec_from_freq(freq_hz)
        x = self.map_lambda_to_unit(lambdas, self.lmin, self.lmax)  # fixed mapping set in setup()

        delay_sec = np.zeros(len(tbl), dtype=float) * u.second

        # Global Legendre sum
        for deg in range(0, self.order):
            coeff = getattr(self, f"LIFD{deg}").value
#            if coeff != 0.0:
            delay_sec += coeff * self._legendre_val(deg, x) * u.second   # This is basically summing the c_k * L_k(x)

        return delay_sec

    def print_par(self, format="pint"):
        result = ""
        LIFD_mapping = self.get_prefix_mapping_component("LIFD")
        for LIFD in LIFD_mapping.values():
            LIFD_par = getattr(self, LIFD)
            result += LIFD_par.as_parfile_line(format=format)
        return result

    # ------------------------------------------------ Derivatives ------------------------------------------------

    def _deriv_global_LFDk(self, toas, param, acc_delay=None):
        # --------------------------------------------------------------------------------------------
        # Δt = Σ_{i=1..N}  c_i L_i(x), and we're taking the derivative with respect to LFD_k = c_k, so
        #
        #                             d(Δt)/d(LFDk) = L_k(x)   (dimensionless)
        #
        #--------------------------------------------------------------------------------------------
        deg = int(param.replace("LIFD", ""))  # For example, "LIFD3" -> 3
        freq_hz = self.get_freq_hz_from_toas(self._parent, toas)
        lambdas = self.get_lambda_sec_from_freq(freq_hz)
        x = self.map_lambda_to_unit(lambdas, self.lmin, self.lmax)
        deriv = self._legendre_val(deg, x)
        return deriv * (u.second / u.second)

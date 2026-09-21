from pint.models.timing_model import DelayComponent
from pint.models.parameter import prefixParameter
import astropy.units as u
import numpy as np

class IFD(DelayComponent):
    """
    IFD chromatic delay:
      Δt(ν, t) =  Σ_{k=1..N}  a_k (lambda/lambda_0)^k
    - Global terms a_k on lambda, where λ=1/ν and lambda_0 = (1 GHz)^{-1}
    IMPORTANT: \lambda_0 is set to 1 GHZ^{-1}, please do not forget
    """

    register = True
    category = "IFD"

    def __init__(self, order=4):
        """
        Parameters
        ----------
        order : int
            Highest degree.
        """

        super().__init__()
        if order < 0:
            raise ValueError("order must be >= 0")
        self.order = int(order)

        # Global coefficients a_0..a_order (seconds)
        for deg in range(0, self.order):
            self.add_param(
                prefixParameter(
                    name=f"IFD{deg}",
                    units="second",
                    value=0.0,
                    description=f"Global Inverse FD coefficient degree {deg}",
                    type_match="float",
                    convert_tcb2tdb=False,
                    frozen=False,  # <--- important
                )
            )

        self.delay_funcs_component += [self._delay]


 # ------------------------------------------------ Helper functions ------------------------------------------------

    @staticmethod
    def get_freq_GHz_from_toas(model, toas):
        tbl = toas.table
        try:
            bfreq = model.barycentric_radio_freq(toas)  # astropy Quantity
            return bfreq.to(u.GHz)
        except Exception:
            # Fallback to topocentric freq column
            col = tbl["freq"]
            return (col * u.MHz).to(u.GHz) if getattr(col, "unit", None) is None else col.to(u.GHz)

    @staticmethod
    def get_lambda_ns_from_freq(freqs):
        # Input: frequencie (in any units, as a quantity object, will be converted to GHz)
        # Output: inverse frequencies (would be in units of nanoseconds)
        # λ = 1/ν  would have units of (GHz)^{-1}, but here we use it as a scalar axis for mapping only.
        freq_GHz = freqs.to(u.GHz).value
        return np.power(freq_GHz, -1.0)

    # ------------------------------------------------ PINT functions ------------------------------------------------
    def setup(self):
        super().setup()

        toas = getattr(self._parent, "toas", None)
        if toas is None:
            raise ValueError("Parent model has no TOAs attached at setup().")

        # Register derivative functions:
        for deg in range(0, self.order):
            self.register_deriv_funcs(self._deriv_global_IFDk, f"IFD{deg}")

    def validate(self):
        super().validate()
        pass

    # This is the core delay!
    def _delay(self, toas, acc_delay=None):
        tbl = toas.table  # Barycentric TOAs
        freq_Ghz = self.get_freq_GHz_from_toas(self._parent, toas)  # GHz
        lambdas = self.get_lambda_ns_from_freq(freq_Ghz)           # GHz^{-1} = ns

        # Global sum
        coeffs = [getattr(self, f"IFD{deg}").value for deg in range(0, self.order)]
        delay_s = np.polynomial.polynomial.polyval(x=lambdas, c=coeffs) * u.second

        return delay_s

    def print_par(self, format="tempo2"):
        result = ""
        IFD_mapping = self.get_prefix_mapping_component("IFD")
        for IFD in IFD_mapping.values():
            IFD_par = getattr(self, IFD)
            result += IFD_par.as_parfile_line(format=format)
        return result

    # ------------------------------------------------ Derivatives ------------------------------------------------

    def _deriv_global_IFDk(self, toas, param, acc_delay=None):
        # --------------------------------------------------------------------------------------------
        # param: IFD parameter we're taking the derivative with respect to
        # Δt = Σ_{i=1..N}  a_i (lambda/lambda_0)^{i}, and we're taking the derivative with respect to IFD_k = a_k, so
        #
        #                     d(Δt)/d(IFDk) = (lambda/lambda_0)^{k}   (dimensionless)
        #
        # --------------------------------------------------------------------------------------------
        deg = int(param.replace("IFD", ""))  # For example, "IFD3" -> 3
        freq_GHz = self.get_freq_GHz_from_toas(self._parent, toas)
        lambdas = self.get_lambda_ns_from_freq(freq_GHz)
        deriv = np.power(lambdas, deg)
        return deriv * (u.second / u.second)

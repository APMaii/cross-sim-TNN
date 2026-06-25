from ..idevice import EmptyDevice
from ...backend import ComputeBackend

xp = ComputeBackend()


class APM_SINW_SBFET(EmptyDevice):
    """
    Empirical 1-bit APM SiNW SBFET device model.

    States:
        ON / LRS
        OFF / HRS

    Models:
        programming_error(): binary state-placement error
        drift_error(): D3-based stochastic time drift
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Gmin = 1 / self.device_params.Rmax
        self.Gmax = 1 / self.device_params.Rmin

        self.Imin = self.Gmin * self.device_params.Vread * 1e9  # nA
        self.Imax = self.Gmax * self.device_params.Vread * 1e9  # nA

    def _calculate_current(self, input_):
        """
        Convert CrossSim normalized conductance to physical current in nA.
        """
        I = self.Imin + (self.Imax - self.Imin) * (
            input_ - self.Gmin_norm
        ) / self.Grange_norm

        return I

    def _current_to_input(self, I):
        """
        Convert physical current in nA back to CrossSim normalized conductance.
        """
        input_ = self.Gmin_norm + self.Grange_norm * (
            (I - self.Imin) / (self.Imax - self.Imin)
        )

        return input_.clip(0, 1)

    def programming_error(self, input_):
        """
        APM_SINW_SBFET binary programming-error model.

        Target states are defined from D3 fixed-read Ron-off measurements
        at Vread = 1 V.

        Actual programmed states are modeled using the relative error
        distribution extracted from 50 I-V hysteresis sweeps at Vread = 1 V.
        """

        I = self._calculate_current(input_)

        I_mid = 0.5 * (self.Imax + self.Imin)

        on_mask = I >= I_mid
        off_mask = I < I_mid

        I_programmed = xp.array(I, copy=True)

        # Units: nA
        I_ON_TARGET_NA = 0.711996800000
        I_OFF_TARGET_NA = 0.227967200000

        ON_SIGMA_REL = 0.048277631988
        OFF_SIGMA_REL = 0.053932935780

        ON_ERROR_MIN_REL = -0.044405817554
        ON_ERROR_MAX_REL = 0.247758416892

        OFF_ERROR_MIN_REL = -0.246295081047
        OFF_ERROR_MAX_REL = 0.128188616608

        if on_mask.any():
            eps_on = xp.random.normal(
                loc=0.0,
                scale=ON_SIGMA_REL,
                size=I[on_mask].shape
            )

            eps_on = xp.clip(
                eps_on,
                ON_ERROR_MIN_REL,
                ON_ERROR_MAX_REL
            )

            I_programmed[on_mask] = I_ON_TARGET_NA * (1.0 + eps_on)

        if off_mask.any():
            eps_off = xp.random.normal(
                loc=0.0,
                scale=OFF_SIGMA_REL,
                size=I[off_mask].shape
            )

            eps_off = xp.clip(
                eps_off,
                OFF_ERROR_MIN_REL,
                OFF_ERROR_MAX_REL
            )

            I_programmed[off_mask] = I_OFF_TARGET_NA * (1.0 + eps_off)

        return self._current_to_input(I_programmed)

    def drift_error(self, input_, time):
        """
        APM_SINW_SBFET D3 empirical stochastic drift model.

        Drift law:
            I(t) = I0 * (1 + t/t0)^a

        Time unit: seconds.
        """

        if time == 0:
            return input_

        I = self._calculate_current(input_)

        t0 = 1.0

        A_ON_MEAN = 0.020880982522
        A_ON_STD = 0.006651255288
        A_ON_MIN = 0.014268920746
        A_ON_MAX = 0.028880097408

        A_OFF_MEAN = -0.021035482875
        A_OFF_STD = 0.006880647250
        A_OFF_MIN = -0.028015050237
        A_OFF_MAX = -0.009688333594

        I_mid = 0.5 * (self.Imax + self.Imin)

        on_mask = I >= I_mid
        off_mask = I < I_mid

        I_drifted = xp.array(I, copy=True)

        if on_mask.any():
            a_on = xp.random.normal(
                loc=A_ON_MEAN,
                scale=A_ON_STD,
                size=I[on_mask].shape
            )

            a_on = xp.clip(a_on, A_ON_MIN, A_ON_MAX)

            I_drifted[on_mask] = I[on_mask] * (1.0 + time / t0) ** a_on

        if off_mask.any():
            a_off = xp.random.normal(
                loc=A_OFF_MEAN,
                scale=A_OFF_STD,
                size=I[off_mask].shape
            )

            a_off = xp.clip(a_off, A_OFF_MIN, A_OFF_MAX)

            I_drifted[off_mask] = I[off_mask] * (1.0 + time / t0) ** a_off

        return self._current_to_input(I_drifted)

    def read_noise(self, input_):
        """
        No read-noise model is currently included.
        """
        
        return input_
        #or
        #raise ValueError(
        #    "APM_SINW_SBFET is not a valid read model. It can only be used to simulate programming error.",
        #)
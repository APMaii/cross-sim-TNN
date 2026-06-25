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
        drift_error(): empirical stochastic drift from D1, D2, D3 cycles
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

        I_actual = I_target * (1 + epsilon)

        epsilon is state-dependent and includes both:
            - measured mean relative error
            - measured standard deviation
        """

        I = self._calculate_current(input_)

        I_mid = 0.5 * (self.Imax + self.Imin)

        on_mask = I >= I_mid
        off_mask = I < I_mid

        I_programmed = xp.array(I, copy=True)

        # Units: nA
        I_ON_TARGET_NA = 0.709038666667
        I_OFF_TARGET_NA = 0.225772066667

        # Relative error mean
        ON_MEAN_REL = 0.010450
        OFF_MEAN_REL = -0.127216

        # Relative error standard deviation
        ON_SIGMA_REL = 0.048479047904
        OFF_SIGMA_REL = 0.054457314136

        # Relative error clipping range from measured data
        ON_ERROR_MIN_REL = -0.040419046258
        ON_ERROR_MAX_REL = 0.252964107270

        OFF_ERROR_MIN_REL = -0.238966969932
        OFF_ERROR_MAX_REL = 0.139157752317

        if on_mask.any():
            eps_on = xp.random.normal(
                loc=ON_MEAN_REL,
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
                loc=OFF_MEAN_REL,
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
        APM_SINW_SBFET empirical stochastic drift model.

        ON / LRS:
            I(t) = I0 * (1 + t/t0)^a

        OFF / HRS:
            Empirical joint sampling from fitted OFF curves:
            I(t) = I0 * [1 + A*(1 - exp(-t/tau1)) + B*(1 - exp(-t/tau2))]

        Time unit:
            seconds
        """
        MAX_DRIFT_TIME_SEC = 101.0
        if time > MAX_DRIFT_TIME_SEC:
            raise ValueError(
                f"APM_SINW_SBFET drift model is only validated "
                f"for retention times up to {MAX_DRIFT_TIME_SEC} seconds."
            )

        if time == 0:
            return input_



        I = self._calculate_current(input_)

        t0 = 1.0

        # ON / LRS power-law parameters
        A_ON_MEAN = 0.025780
        A_ON_STD = 0.002856
        A_ON_MIN = 0.020074
        A_ON_MAX = 0.032939

        # OFF / HRS empirical fitted parameter table
        # Each row is one real fitted OFF drift curve: [A, tau1, B, tau2]
        OFF_FIT_TABLE = xp.array([
            [-0.279823, 15.791254, 0.626822, 300.000000],
            [-1.000000, 23.326950, 0.946585,  33.256634],
            [-1.000000, 24.873327, 0.962306,  36.445650],
            [-0.304159, 12.693863, 0.231730,  56.836447],
            [-0.996635, 21.183390, 0.941414,  30.905514],
            [-0.327217, 16.405843, 0.331286,  79.954135],
            [-0.271670, 13.500160, 0.704667, 300.000000],
            [-0.305352, 15.754444, 0.276956,  74.756819],
            [-0.294003, 17.096792, 0.767971, 300.000000],
            [-0.274824, 12.745730, 0.633352, 300.000000],
            [-0.206250, 13.058626, 0.383317, 300.000000],
            [-0.323915, 15.817618, 0.817728, 300.000000],
            [-0.214140, 16.471436, 0.439525, 300.000000],
            [-0.399050, 16.531542, 1.000000, 241.815029],
            [-0.241828, 12.260364, 0.401551, 300.000000],
        ])

        I_mid = 0.5 * (self.Imax + self.Imin)

        on_mask = I >= I_mid
        off_mask = I < I_mid

        I_drifted = xp.array(I, copy=True)

        # ON / LRS drift
        if on_mask.any():
            a_on = xp.random.normal(
                loc=A_ON_MEAN,
                scale=A_ON_STD,
                size=I[on_mask].shape
            )

            a_on = xp.clip(
                a_on,
                A_ON_MIN,
                A_ON_MAX
            )

            I_drifted[on_mask] = (
                I[on_mask]
                * (1.0 + time / t0) ** a_on
            )

        # OFF / HRS drift
        if off_mask.any():
            n_off = I[off_mask].size

            row_idx = xp.random.randint(
                0,
                OFF_FIT_TABLE.shape[0],
                size=n_off
            )

            sampled_params = OFF_FIT_TABLE[row_idx]

            A_off = sampled_params[:, 0]
            tau1_off = sampled_params[:, 1]
            B_off = sampled_params[:, 2]
            tau2_off = sampled_params[:, 3]

            off_factor = (
                1.0
                + A_off * (1.0 - xp.exp(-time / tau1_off))
                + B_off * (1.0 - xp.exp(-time / tau2_off))
            )

            I_drifted[off_mask] = I[off_mask] * off_factor

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
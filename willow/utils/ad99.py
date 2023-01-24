from typing import Literal, Optional, Union, overload

import numpy as np

from .mima import R_dry, c_p, grav

class AlexanderDunkerton:
    """Vectorized implementation of the AD99 parameterization."""

    def __init__(self) -> None:
        """Initialize an AlexanderDunkerton instance."""

        self.bk = self._get_bk()
        self.p_ref = self._get_vertical_coords(100000)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute parameterized drags for an array of profiles.

        Parameters
        ----------
        X : Array of profiles. The first 40 columns correspond to wind, the
            second 40 to temperature, and the last two to surface pressure and
            latitude, respectively.

        Returns
        -------
        drag : Array of parameterized accelerations with 40 columns and the same
            number of rows as `X`.

        """

        drag = np.zeros((X.shape[0], 40))
        for i, row in enumerate(X):
            wind, T = row[:40], row[40:80]
            p_surf, lat = row[-2:]
            drag[i] = self._predict(wind, T, p_surf, lat)

        return drag

    def _get_buoyancy_frequency(
        self,
        T: np.ndarray,
        z: np.ndarray
    ) -> np.ndarray:
        """
        Compute buoyancy frequency at each level.

        Parameters
        ----------
        T : Array of temperatures.
        z : Array of heights.

        Returns
        -------
        N : Array of buoyancy frequencies.

        """
        
        dTdz = self._add_zero_index()
        dTdz[1] = (T[1] - T[2]) / (z[1] - z[2])
        dTdz[2:-1] = (T[1:-2] - T[3:]) / (z[1:-2] - z[3:])

        N = (grav / T) * (dTdz + grav / c_p)
        N = np.sqrt(np.maximum(N, 2.5e-5))
        N[0] = N[1]

        return N

    def _get_density(self, p: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Compute density at each vertical level.
        
        Parameters
        ----------
        p : Array of pressures.
        T : Array of temperatures.

        Returns
        -------
        rho : Array of densities calculated with the ideal gas law.

        """

        rho = p / (R_dry * T)
        rho[0] = (rho[1] ** 2) / (rho[2])

        return rho

    def _get_k_source(self, lat: float) -> int:
        """
        Get the index of the source level.

        Parameters
        ----------
        lat : Latitude, in degrees.

        Returns
        -------
        k_source : Index of the source level, to be used in profile arrays that
            have had a dummy top zero index added.

        """

        tropopause = np.argmax(self.p_ref > 31500)
        k = int(40.5 - (41 - tropopause) * np.cos(np.radians(lat)))

        return min(k, 39)

    @overload
    def _get_vertical_coords(
        self,
        p_surf: float,
        T: Literal[None]=...
    ) -> np.ndarray:
        ...

    @overload
    def _get_vertical_coords(
        self,
        p_surf: float,
        T: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ...

    def _get_vertical_coords(
        self,
        p_surf: float,
        T: Optional[np.ndarray]=None
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Compute pressures and potentially heights using the hybrid coordinate.

        Parameters
        ----------
        p_surf : Surface pressure.
        T : Temperature profile for use in computing geopotentials. If `None`,
            heights will not be returned.

        Returns
        -------
        p_full : Pressures at full vertical levels.
        z : Heights at full vertical levels. Only returned if `T is not None`.

        """
        
        p_half = self.bk * p_surf
        log_p_half = np.ma.log(p_half).filled(0)
        dlogp = np.diff(log_p_half) / np.diff(p_half)

        log_p_full = log_p_half[1:] - 1 + p_half[:-1] * dlogp
        p_full = self._add_zero_index(np.exp(log_p_full))

        if T is None:
            return p_full

        steps = R_dry * T[-1:1:-1] * (log_p_half[-1:1:-1] - log_p_half[-2:0:-1])
        geo_half = np.append(np.cumsum(steps)[::-1], 0)

        geo_full = geo_half + R_dry * T[1:] * (log_p_half[1:] - log_p_full)
        z = self._add_zero_index(geo_full / grav)
        z[0] = 2 * z[1] - z[2]

        return p_full, z

    def _predict(
        self,
        wind: np.ndarray,
        T: np.ndarray,
        p_surf: float,
        lat: float
    ) -> np.ndarray:
        """
        Compute a single parameterized drag profile.

        Parameters
        ----------
        wind : Array of velocities.
        T : Array of temperatures.
        p_surf : Surface pressure.
        lat : Latitude, in degrees.

        Returns
        -------
        drag : Array of parameterized accelerations.

        """

        wind = self._add_zero_index(wind)
        T = self._add_zero_index(T)
        wind[0] = 2 * wind[1] - wind[2]

        p, z = self._get_vertical_coords(p_surf, T)
        k_source = self._get_k_source(lat)
        dz = np.diff(z)

        rho = self._get_density(p, T)
        N = self._get_buoyancy_frequency(T, z)

        cs = self._get_phase_speeds()
        amp, Bs = self._get_amp_and_Bs(lat, cs, wind[k_source])
        eps = amp / (abs(Bs).sum() * rho[k_source])

        wvn = 2 * np.pi / 300e3
        H = dz / np.diff(np.log(rho))
        alpha = 1 / ((2 * H) ** 2)

        omc = self._add_zero_index()
        omc[:-1] = wvn * N[:-1] / np.sqrt((wvn ** 2) + alpha)
        fac = (wvn * rho) / (2 * N * rho[k_source])

        wind = wind[:(k_source + 1)]
        omc = omc[:(k_source + 1)]
        fac = fac[:(k_source + 1)]

        cs, Bs = cs[:, None], Bs[:, None]
        breaking = Bs / ((cs - wind) ** 3) >= fac
        sign_change = (cs - wind) * (cs - wind[k_source]) <= 0

        remove = (cs == wind) | (wvn * abs(cs - wind) >= omc)
        deposit = breaking | sign_change
        deposit[:, 0] = True

        k_remove = k_source - np.argmax(remove[:, ::-1], axis=1)
        k_deposit = k_source - np.argmax(deposit[:, ::-1], axis=1)
        k_remove[~np.any(remove, axis=1)] = -1

        idx = (k_remove < k_deposit) & (k_deposit < k_source)
        cs, Bs = cs[idx, 0], Bs[idx, 0]
        k_deposit = k_deposit[idx]

        drag = self._add_zero_index()
        drag[0] = 0

        np.add.at(drag, k_deposit, Bs)
        rbh = np.sqrt(rho[:k_source] * rho[1:(k_source + 1)])
        drag[:k_source] *= (eps * rho[k_source]) / (rbh * -dz[:k_source])

        drag[1:] /= 2
        drag[2:(k_source + 1)] += drag[1:k_source]
        drag[1:4] += drag[0] / 3

        return drag[1:]

    @staticmethod
    def _add_zero_index(a: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Add an initial `nan` to represent the temporary zero index in Fortran.

        Parameters
        ----------
        a : Array to add a slot to. If `None`, a zero array will be created.

        Returns
        -------
        a : Passed array with a prepended `nan`.
        
        """

        if a is None:
            a = np.zeros(40)

        return np.insert(a, 0, np.nan)

    @staticmethod
    def _get_amp_and_Bs(
        lat: float,
        cs: np.ndarray,
        wind: float
    ) -> tuple[float, np.ndarray]:
        """
        Compute the amplitude and B parameters as defined in the Fortran.

        Parameters
        ----------
        lat : Latitude in degrees.
        cs : Array of phase speeds
        wind : Wind velocity at the source level.

        Returns
        -------
        amp : Amplitude of source spectrum as determined by the latitude.
        Bs : Array of fluxes with the same shape as `cs`.

        """

        Bt_0, Bt_eq = 0.0043, 0.0048
        Bt_nh, Bt_sh = 0.0035, 0.0035
        phi0, dphi = 15, 10

        if abs(lat) > phi0:
            amp = Bt_0 + (
                Bt_nh * 0.5 * (1 + np.tanh((lat - phi0) / dphi)) +
                Bt_sh * 0.5 * (1 + np.tanh((lat + phi0) / -dphi))
            )

        elif abs(lat) <= dphi:
            amp = Bt_eq

        elif dphi < lat <= phi0:
            amp = Bt_0 + (Bt_eq - Bt_0) * (phi0 - lat) / (phi0 - dphi)

        elif -phi0 <= lat < -dphi:
            amp = Bt_0 + (Bt_eq - Bt_0) * (-phi0 - lat) / (dphi - phi0)

        diff = cs - wind
        if abs(lat) < dphi:
            cs = diff

        Bs = 0.4 * np.exp(-np.log(2) * ((cs / 35) ** 2))
        Bs[diff < 0] = -1 * Bs[diff < 0]

        return amp, Bs

    @staticmethod
    def _get_bk() -> np.ndarray:
        """Compute the bk coefficients used in the hybrid coordinate."""

        bk = np.zeros(41)
        bk[0], bk[-1] = 0, 1

        zeta = 1 - np.arange(1, 40) / 40
        z = 0.1 * zeta + 0.9 * (zeta ** 1.4)
        bk[1:40] = np.exp(-7.9 * z)

        return bk

    @staticmethod
    def _get_phase_speeds(c_max: float=99.6, dc: float=1.2) -> np.ndarray:
        """
        Compute the gravity wave phase speeds.
        
        Parameters
        ----------
        c_max : Norm of the maximum phase speed.
        dc : Phase speed spacing.

        Returns
        -------
        cs : Array of phase speeds.

        """

        return np.linspace(-c_max, c_max, int(2 * c_max / dc + 1))
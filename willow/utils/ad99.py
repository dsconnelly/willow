import numpy as np

from .mima import R_dry, c_p, grav

class AlexanderDunkerton:
    def __init__(self):
        self.bk = self.get_bk()
        self.p_ref = self.get_vertical_coords(100000)

    def predict(self, X):
        out = np.zeros((X.shape[0], 40))
        for i, row in enumerate(X):
            wind, T = row[:40], row[40:80]
            p_surf, lat = row[-2:]
            out[i] = self._predict(wind, T, p_surf, lat)

        return out

    def _predict(self, wind, T, p_surf, lat):
        wind = self.add_zero_index(wind)
        T = self.add_zero_index(T)
        wind[0] = 2 * wind[1] - wind[2]

        p, z = self.get_vertical_coords(p_surf, T)
        k_source = self.get_k_source(lat)
        dz = np.diff(z)

        rho = self.get_density(p, T)
        N = self.get_buoyancy_frequency(T, z)

        cs = self.get_phase_speeds()
        amp, Bs = self.get_amp_and_Bs(lat, cs, wind[k_source])
        eps = amp / (abs(Bs).sum() * rho[k_source])

        wvn = 2 * np.pi / 300e3
        H = dz / np.diff(np.log(rho))
        alpha = 1 / ((2 * H) ** 2)

        omc = self.add_zero_index()
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

        drag = self.add_zero_index()
        drag[0] = 0

        np.add.at(drag, k_deposit, Bs)
        rbh = np.sqrt(rho[:k_source] * rho[1:(k_source + 1)])
        drag[:k_source] *= (eps * rho[k_source]) / (rbh * -dz[:k_source])

        drag[1:] /= 2
        drag[2:(k_source + 1)] += drag[1:k_source]
        drag[1:4] += drag[0] / 3

        return drag[1:]

    def get_buoyancy_frequency(self, T, z):
        dTdz = self.add_zero_index()
        dTdz[1] = (T[1] - T[2]) / (z[1] - z[2])
        dTdz[2:-1] = (T[1:-2] - T[3:]) / (z[1:-2] - z[3:])

        N = (grav / T) * (dTdz + grav / c_p)
        N = np.sqrt(np.maximum(N, 2.5e-5))
        N[0] = N[1]

        return N

    def get_k_source(self, lat):
        k = np.argmax(self.p_ref > 31500)
        k = int(40.5 - (41 - k) * np.cos(np.radians(lat)))

        return min(k, 39)

    def get_density(self, p, T):
        rho = p / (R_dry * T)
        rho[0] = (rho[1] ** 2) / (rho[2])

        return rho

    def get_vertical_coords(self, p_surf, T=None):
        p_half = self.bk * p_surf
        log_p_half = np.ma.log(p_half).filled(0)
        dlogp = np.diff(log_p_half) / np.diff(p_half)

        log_p_full = log_p_half[1:] - 1 + p_half[:-1] * dlogp
        p_full = self.add_zero_index(np.exp(log_p_full))

        if T is None:
            return p_full

        steps = R_dry * T[-1:1:-1] * (log_p_half[-1:1:-1] - log_p_half[-2:0:-1])
        geo_half = np.append(np.cumsum(steps)[::-1], 0)

        geo_full = geo_half + R_dry * T[1:] * (log_p_half[1:] - log_p_full)
        z = self.add_zero_index(geo_full / grav)
        z[0] = 2 * z[1] - z[2]

        return p_full, z

    @staticmethod
    def add_zero_index(a=None, mode=None):
        if a is None:
            a = np.zeros(40)

        return np.insert(a, 0, np.nan)

    @staticmethod
    def get_amp_and_Bs(lat, cs, wind):
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
    def get_bk():
        bk = np.zeros(41)
        bk[0], bk[-1] = 0, 1

        zeta = 1 - np.arange(1, 40) / 40
        z = 0.1 * zeta + 0.9 * (zeta ** 1.4)
        bk[1:40] = np.exp(-7.9 * z)

        return bk

    @staticmethod
    def get_phase_speeds(c_max=99.6, dc=1.2):
        return np.linspace(-c_max, c_max, int(2 * c_max / dc + 1))

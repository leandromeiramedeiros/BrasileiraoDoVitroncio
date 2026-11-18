"""Goal-observation builders (the 'adjusted performance' targets)."""
import numpy as np


def fit_xg_linear(d, Y):
    m = (d.season < Y) & d.statsOk & d.post
    if m.sum() < 300:
        return None
    X = np.concatenate([np.c_[np.ones(m.sum()), d.st_sot[m, s], np.maximum(0, d.st_sh[m, s] - d.st_sot[m, s])] for s in (0, 1)])
    y = np.concatenate([d.hg[m], d.ag[m]]).astype(float)
    return np.linalg.lstsq(X, y, rcond=None)[0]


def obs_v1(d, hp, Y, aux=None):
    """v1 observation: goals with man-advantage goals down-weighted (wRed), blended with the linear shots proxy (w)."""
    wRed = hp.get('wRed', 1.0); w = hp.get('w', 1.0)
    g = np.where(d.evOk[:, None], d.g_all - (1 - wRed) * d.g_adv, np.c_[d.hg, d.ag]).astype(float)
    b = aux if aux is not None else fit_xg_linear(d, Y)
    if b is not None and w < 0.99999:
        x = np.maximum(0.05, b[0] + b[1] * d.st_sot + b[2] * np.maximum(0, d.st_sh - d.st_sot))
        g = np.where(d.statsOk[:, None], w * g + (1 - w) * x, g)
    return g, np.ones((d.n, 2))


def obs_adj(d, hp, Y, aux=None):
    """Goal observation with category weights and (optionally) red-card exposure.
    y = all - (1-wRed) adv - (1-wPen) pen - (1-wOwn) own - (1-wLead) late goals by a side already 2+ up
            - (1-wTrail) late consolation goals by a side 2+ down
    E = exp(expo * (0.40 f_adv_own - 0.45 f_adv_opp))  (share of the match played a man up / a man down)
    """
    g = d.g_all.copy()
    for key, cat in (('wRed', 'adv'), ('wPen', 'pen'), ('wOwn', 'own'), ('wLead', 'late_lead2'), ('wTrail', 'late_trail2')):
        w = hp.get(key, 1.0)
        if w < 0.99999:
            g = g - (1 - w) * getattr(d, 'g_' + cat)
    g = np.where(d.evOk[:, None], g, np.c_[d.hg, d.ag]).astype(float)
    ex = hp.get('expo', 0.0)
    E = np.ones((d.n, 2))
    if ex > 0:
        E = np.exp(ex * (0.40 * d.fadv - 0.45 * d.fadv[:, ::-1]))
        E = np.where(d.evOk[:, None], E, 1.0)
    return g, E

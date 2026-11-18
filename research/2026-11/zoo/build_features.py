"""Ratings zoo: strictly walk-forward (online) feature generators over the match stream.

Every generator exposes  roll(team, info) / pre(m) -> dict / post(m).
The driver computes the features of a match from the state BEFORE kick-off; a finished match only
updates the state once  kickoff + 2.5h <= kickoff of the match being predicted  (results of games that are
still being played are never used).

All hyper-parameters in this file are fixed a priori (literature defaults or simple statistics of the
2006-2012 seasons, i.e. before DEV); nothing here is fitted on 2013+.

Run:  /home/user/bdv-lab/venv/bin/python build_features.py      -> features_zoo.csv (+ features_all.pkl)
"""
import math
import sys
from collections import defaultdict
import numpy as np
import pandas as pd
from lib import load_matches, load_base, ChainGrid, HZ, Z

DELAY_H = 2.5

# ----------------------------------------------------------------------------- static geography (hand-entered)
CITY = {
    'rio': (-22.91, -43.17), 'sp': (-23.55, -46.63), 'poa': (-30.03, -51.23), 'cwb': (-25.43, -49.27),
    'bh': (-19.92, -43.94), 'santos': (-23.96, -46.33), 'recife': (-8.05, -34.88), 'campinas': (-22.91, -47.06),
    'fortaleza': (-3.73, -38.53), 'salvador': (-12.97, -38.51), 'floripa': (-27.59, -48.55), 'goiania': (-16.69, -49.26),
    'braganca': (-22.95, -46.54), 'criciuma': (-28.68, -49.37), 'maceio': (-9.67, -35.74), 'chapeco': (-27.10, -52.62),
    'caxias': (-29.17, -51.18), 'natal': (-5.79, -35.21), 'barueri': (-23.51, -46.88), 'saoluis': (-2.53, -44.30),
    'londrina': (-23.31, -51.16), 'cuiaba': (-15.60, -56.10), 'saocaetano': (-23.62, -46.57), 'belem': (-1.46, -48.50),
    'pontagrossa': (-25.09, -50.16), 'pelotas': (-31.77, -52.34), 'ribeirao': (-21.18, -47.81), 'joinville': (-26.30, -48.85),
    'novohorizonte': (-21.47, -49.22), 'ipatinga': (-19.47, -42.54), 'arapiraca': (-9.75, -36.66), 'juazeiro': (-7.21, -39.32),
    'guaratingueta': (-22.82, -45.19), 'varginha': (-21.55, -45.43), 'lucas': (-13.06, -55.91), 'mirassol': (-20.82, -49.52),
    'brasilia': (-15.83, -48.06), 'santoandre': (-23.66, -46.53), 'duque': (-22.79, -43.31), 'brusque': (-27.10, -48.92),
    'itu': (-23.26, -47.30), 'sorocaba': (-23.50, -47.46), 'aracaju': (-10.91, -37.07), 'muriae': (-21.13, -42.37),
    'manaus': (-3.10, -60.02), 'sjdelrei': (-21.14, -44.26), 'marilia': (-22.21, -49.95), 'campinagrande': (-7.23, -35.88),
    'salgueiro': (-8.07, -39.12), 'mogi': (-22.43, -46.96), 'macae': (-22.37, -41.79), 'juizdefora': (-21.76, -43.35),
    'voltaredonda': (-22.52, -44.10), 'araraquara': (-21.79, -48.18), 'saobernardo': (-23.69, -46.56), 'itapolis': (-21.60, -48.81),
    'setelagoas': (-19.47, -44.25), 'ituiutaba': (-18.97, -49.46), 'americana': (-22.74, -47.33), 'cariacica': (-20.26, -40.42),
    'uberlandia': (-18.92, -48.28), 'itajai': (-26.91, -48.66), 'novohamburgo': (-29.68, -51.13), 'feira': (-12.27, -38.97),
    'paragominas': (-2.99, -47.35), 'campogrande': (-20.44, -54.65), 'maringa': (-23.42, -51.94), 'jundiai': (-23.19, -46.88),
    'joaopessoa': (-7.12, -34.86), 'erechim': (-27.63, -52.27), 'teresina': (-5.09, -42.80), 'catanduva': (-21.14, -48.97),
}
# normalised ESPN city string -> key above (clearly wrong ESPN cities such as 'Kaiserslautern' are left unmapped)
CITY_STR = {
    'rio de janeiro': 'rio', 'ilha do governador': 'rio', 'edson passos': 'rio', 'curitiba': 'cwb', 'sao paulo': 'sp',
    'belo horizonte': 'bh', 'porto alegre': 'poa', 'recife': 'recife', 'goiania': 'goiania', 'aparecida de goiania': 'goiania',
    'salvador': 'salvador', 'florianopolis': 'floripa', 'fortaleza': 'fortaleza', 'horizonte': 'fortaleza', 'santos': 'santos',
    'braganca paulista': 'braganca', 'maceio, alagoas': 'maceio', 'maceio': 'maceio', 'criciuma': 'criciuma',
    'caxias do sul': 'caxias', 'chapeco': 'chapeco', 'ponte preta': 'campinas', 'campinas': 'campinas', 'indaiatuba': 'campinas',
    'barueri': 'barueri', 'osasco, sao paulo': 'barueri', 'belem': 'belem', 'cuiaba': 'cuiaba', 'natal': 'natal',
    'rio grande do norte': 'natal', 'goianinha': 'natal', 'londrina': 'londrina', 'sao luis': 'saoluis',
    'sao caetano do sul': 'saocaetano', 'sete lagoas': 'setelagoas', 'ponta grossa': 'pontagrossa', 'ribeirao preto': 'ribeirao',
    'pelotas': 'pelotas', 'joinville': 'joinville', 'novo horizonte': 'novohorizonte', 'ipatinga': 'ipatinga',
    'varginha': 'varginha', 'virginha': 'varginha', 'mirassol': 'mirassol', 'lucas do rio verde': 'lucas', 'itu': 'itu',
    'volta redonda': 'voltaredonda', 'taguatinga': 'brasilia', 'brasilia': 'brasilia', 'gama': 'brasilia', 'paranoa': 'brasilia',
    'guaratingueta': 'guaratingueta', 'manaus': 'manaus', 'brusque': 'brusque', 'itapolis': 'itapolis', 'sorocaba': 'sorocaba',
    'aracaju, sergipe': 'aracaju', 'aracaju': 'aracaju', 'araraquara': 'araraquara', 'arapiraca': 'arapiraca',
    'santo andre': 'santoandre', 'muriae, minas gerais': 'muriae', 'tombos': 'muriae', 'duque de caxias': 'duque',
    'sao joao del rei': 'sjdelrei', 'juiz de fora': 'juizdefora', 'sao bernardo do campo': 'saobernardo',
    'sao bernanrdo do campo': 'saobernardo', 'marilia': 'marilia', 'campina grande': 'campinagrande', 'ituiutaba': 'ituiutaba',
    'juazeiro do norte': 'juazeiro', 'americana': 'americana', 'salgueiro': 'salgueiro', 'espirito santo': 'cariacica',
    'uberlandia': 'uberlandia', 'itajai': 'itajai', 'macae': 'macae', 'novo hamburgo': 'novohamburgo', 'campo bom': 'novohamburgo',
    'feira de santana': 'feira', 'paragominas': 'paragominas', 'campo grande': 'campogrande', 'mogi mirim': 'mogi',
    'maringa': 'maringa', 'jundiai': 'jundiai', 'joao pessoa': 'joaopessoa', 'erechim, rs': 'erechim', 'teresina': 'teresina',
    'catanduva': 'catanduva',
}


def norm_city(c):
    if not c: return None
    import unicodedata
    c = ''.join(ch for ch in unicodedata.normalize('NFD', c) if unicodedata.category(ch) != 'Mn').lower().strip()
    return CITY_STR.get(c)
TEAM_CITY = {
    '3454': 'rio', '6086': 'rio', '3445': 'rio', '819': 'rio', '2026': 'sp', '2029': 'sp', '874': 'sp', '4773': 'sp',
    '6273': 'poa', '1936': 'poa', '3458': 'cwb', '3456': 'cwb', '3453': 'cwb', '2674': 'santos', '2022': 'bh', '7632': 'bh',
    '6154': 'bh', '7635': 'recife', '7633': 'recife', '4929': 'recife', '3459': 'campinas', '3448': 'campinas',
    '9969': 'fortaleza', '6272': 'fortaleza', '9967': 'salvador', '3457': 'salvador', '9966': 'floripa', '3461': 'floripa',
    '10357': 'goiania', '9973': 'goiania', '3395': 'goiania', '6079': 'braganca', '9971': 'criciuma', '9970': 'maceio',
    '7531': 'maceio', '9318': 'chapeco', '6270': 'caxias', '9965': 'natal', '7634': 'natal', '10263': 'barueri',
    '7375': 'barueri', '4873': 'saoluis', '17333': 'londrina', '17313': 'cuiaba', '6084': 'saocaetano', '15424': 'belem',
    '4936': 'belem', '18187': 'pontagrossa', '17134': 'pelotas', '10281': 'ribeirao', '13143': 'joinville',
    '18127': 'novohorizonte', '4905': 'ipatinga', '6196': 'arapiraca', '10983': 'juazeiro', '7376': 'guaratingueta',
    '15560': 'varginha', '15576': 'lucas', '9169': 'mirassol', '9968': 'brasilia', '9972': 'brasilia', '4813': 'santoandre',
    '9175': 'duque', '7486': 'varginha', '18428': 'brusque', '4774': 'itu', '6083': 'sorocaba', '4875': 'aracaju',
    '17339': 'muriae', '21888': 'manaus', '20851': 'sjdelrei', '4782': 'marilia', '7509': 'campinagrande',
    '1361': 'salgueiro', '4777': 'mogi', '9176': 'macae', '15575': 'juizdefora', '4806': 'voltaredonda',
    '18126': 'araraquara', '11268': 'saobernardo',
}


def km(c1, c2):
    (a1, o1), (a2, o2) = CITY[c1], CITY[c2]
    p1, p2 = math.radians(a1), math.radians(a2)
    x = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(o2 - o1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(x))


# ----------------------------------------------------------------------------- per-match derived quantities
def _compile():
    g = HZ['goal']
    F = np.zeros((8, 5, 3))
    for b in range(8):
        for d in range(5):
            for mi in range(3):
                e = g['bin'][b] + g['diff'][d] + g['men'][mi]
                if b >= 6 and d < 2: e += g.get('lateTrail', 0) or 0
                F[b, d, mi] = math.exp(e) / 90 * (g.get('norm', 1) or 1)
    bins = []
    for m in range(1, 46): bins.append(0 if m <= 15 else 1 if m <= 30 else 2)
    bins += [3] * HZ['S1']
    for m in range(46, 91): bins.append(4 if m <= 60 else 5 if m <= 75 else 6)
    bins += [7] * HZ['S2']
    return F, bins


GOALF, BINS = _compile()
NSTEP = len(BINS)


def step_of(minute, add):
    S1, S2 = HZ['S1'], HZ['S2']
    if minute <= 45:
        if minute < 45 or not add: return max(0, min(44, minute - 1))
        return 45 + max(0, min(S1 - 1, add - 1))
    if minute < 90 or not add: return S1 + max(45, min(89, minute - 1))
    return 90 + S1 + max(0, min(S2 - 1, add - 1))


def derive(m):
    """Post-match quantities of a finished match (never used before it is over)."""
    d = {}
    if m['status'] != 'post': return d
    if m['statsOk']:
        for s in 'ha':
            st = m['stats'][s]
            d['sh_' + s] = st['sh']; d['sot_' + s] = st['sot']
            d['co_' + s] = st['co']; d['po_' + s] = st['po']; d['fo_' + s] = st['fo']
            d['xg_' + s] = max(0.05, 0.555 + 0.247 * st['sot'] - 0.049 * max(0, st['sh'] - st['sot']))
    d['evOk'] = bool(m['evOk'])
    if m['evOk']:
        evs = sorted(((step_of(e['m'], e['x']), i, e) for i, e in enumerate(m['ev'])), key=lambda z: (z[0], z[1]))
        hg = ag = rh = ra = 0
        Eh = Ea = 0.0; lead_h = lead_a = 0; madv = 0; pen_h = pen_a = 0; comp_h = comp_a = 0; k = 0
        ht_h = ht_a = 0; first = 0; H1 = 45 + HZ['S1']
        for t in range(NSTEP):
            b = BINS[t]
            dH = max(-2, min(2, hg - ag)) + 2; dA = 4 - dH
            mH = 2 if ra > rh else (0 if rh > ra else 1); mA = 2 - mH
            Eh += GOALF[b, dH, mH]; Ea += GOALF[b, dA, mA]
            if hg > ag: lead_h += 1
            elif ag > hg: lead_a += 1
            madv += 1 if ra > rh else (-1 if rh > ra else 0)
            while k < len(evs) and evs[k][0] == t:
                e = evs[k][2]; k += 1
                if e['t'] == 'R':
                    if e['s'] == 'h': rh += 1
                    else: ra += 1
                elif e['t'] in 'GPO':
                    close = abs(hg - ag) <= 1
                    if hg + ag == 0: first = 1 if e['s'] == 'h' else -1
                    if e['s'] == 'h':
                        hg += 1; pen_h += e['t'] == 'P'; comp_h += close; ht_h += t < H1
                    else:
                        ag += 1; pen_a += e['t'] == 'P'; comp_a += close; ht_a += t < H1
        d.update(Eh=Eh, Ea=Ea, dom=(lead_h - lead_a) / NSTEP, madv=madv / NSTEP, pen_h=pen_h, pen_a=pen_a,
                 comp_h=comp_h, comp_a=comp_a, red_h=rh, red_a=ra, ht_h=ht_h, ht_a=ht_a, first=first)
    return d


# ----------------------------------------------------------------------------- small helpers
class EW:
    """Exponentially weighted, shrunk-to-zero mean per key."""

    def __init__(self, hl, k0=3.0):
        self.d = 0.5 ** (1.0 / hl); self.k0 = k0
        self.S = defaultdict(float); self.W = defaultdict(float)

    def get(self, key):
        return self.S[key] / (self.W[key] + self.k0)

    def add(self, key, x, w=1.0):
        self.S[key] = self.S[key] * self.d + w * x; self.W[key] = self.W[key] * self.d + w

    def scale(self, key, f):
        self.S[key] *= f; self.W[key] *= f


def roll_scalar(r, info, carry, dpromo, dnew, dnewA):
    """Season roll-over of a 'strength' scalar expressed relative to the league average."""
    if info['new']:
        if info['lg'] == 'A': return 0.0 if info['season'] <= 2007 else -dnewA
        return -dnew
    r = r * carry ** info['gap']
    if info['from'] != info['lg']:
        r += -dpromo if info['lg'] == 'A' else dpromo
    elif info['gap'] > 1:
        r = min(r, -dnew)
    return r


# ----------------------------------------------------------------------------- rating systems
class Elo:
    """Elo with home advantage and margin-of-victory multiplier (World Football Elo style)."""

    def __init__(self, name, K, H=96.0, carry=0.85, dpromo=115.0, dnew=0.0, dnewA=45.0, mode='goals', scale=1.0):
        self.name, self.K, self.H, self.carry, self.dpromo, self.dnew, self.dnewA, self.mode, self.scale = name, K, H, carry, dpromo, dnew, dnewA, mode, scale
        self.R = defaultdict(float)

    def roll(self, t, info):
        self.R[t] = roll_scalar(self.R[t], info, self.carry, self.dpromo, self.dnew, self.dnewA)

    def pre(self, m):
        return {self.name + '_d': (self.R[m['h']] - self.R[m['a']]) / 100.0}

    def post(self, m):
        d = m['d']
        E = 1.0 / (1.0 + 10 ** (-(self.R[m['h']] - self.R[m['a']] + self.H) / 400.0))
        if self.mode == 'goals':
            gd = m['hg'] - m['ag']
            S = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
            n = abs(gd); mult = 1.0 if n <= 1 else (1.5 if n == 2 else 1.75 + (n - 3) / 8.0)
        else:  # soft result from the shot-based xG proxy
            if 'xg_h' not in d: return
            S = 1.0 / (1.0 + math.exp(-(d['xg_h'] - d['xg_a']) / self.scale)); mult = 1.0
        delta = self.K * mult * (S - E)
        self.R[m['h']] += delta; self.R[m['a']] -= delta


class Pi:
    """pi-ratings (Constantinou & Fenton 2013): home and away rating per team with cross learning."""

    def __init__(self, name, lam, gam=0.7, carry=0.85, dpromo=0.65, dnew=0.0, dnewA=0.27, mode='goals'):
        self.name, self.lam, self.gam, self.carry, self.dpromo, self.dnew, self.dnewA, self.mode = name, lam, gam, carry, dpromo, dnew, dnewA, mode
        self.RH = defaultdict(float); self.RA = defaultdict(float)

    def roll(self, t, info):
        for R in (self.RH, self.RA):
            R[t] = roll_scalar(R[t], info, self.carry, self.dpromo, self.dnew, self.dnewA)

    @staticmethod
    def g(R):
        return math.copysign(10 ** (abs(R) / 3.0) - 1.0, R)

    def pre(self, m):
        h, a = m['h'], m['a']
        return {self.name + '_d': self.g(self.RH[h]) - self.g(self.RA[a]),
                self.name + '_o': (self.RA[h] - self.RH[a])}  # "other venue" ratings: extra info on venue split

    def post(self, m):
        h, a = m['h'], m['a']; d = m['d']
        if self.mode == 'goals': gd = m['hg'] - m['ag']
        else:
            if 'xg_h' not in d: return
            gd = d['xg_h'] - d['xg_a']
        exp = self.g(self.RH[h]) - self.g(self.RA[a])
        e = abs(gd - exp); psi = 3.0 * math.log10(1.0 + e)
        ph = psi if gd > exp else -psi
        self.RH[h] += ph * self.lam; self.RA[h] += ph * self.lam * self.gam
        self.RA[a] -= ph * self.lam; self.RH[a] -= ph * self.lam * self.gam


class Berrar:
    """Berrar/Lopes/Dubitzky-style goal-expectation ratings: home/away attack and defence, bounded logistic link."""

    def __init__(self, name, w, cross=0.5, alpha=5.0, beta=1.0, carry=0.85, dpromo=0.3):
        self.name, self.w, self.cross, self.alpha, self.beta, self.carry, self.dpromo = name, w, cross, alpha, beta, carry, dpromo
        self.gh = math.log(1.62 / (alpha - 1.62)); self.ga = math.log(1.07 / (alpha - 1.07))  # 2006-2012 goal means
        self.r = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])  # hatt, hdef, aatt, adef (def: higher = leakier)

    def roll(self, t, info):
        r = self.r[t]
        for i in range(4):
            sgn = 1 if i in (0, 2) else -1
            r[i] = sgn * roll_scalar(sgn * r[i], info, self.carry, self.dpromo, 0.0, 0.12)

    def exp(self, h, a):
        H, A = self.r[h], self.r[a]
        gh = self.alpha / (1 + math.exp(-self.beta * (H[0] + A[3]) - self.gh))
        ga = self.alpha / (1 + math.exp(-self.beta * (A[2] + H[1]) - self.ga))
        return gh, ga

    def pre(self, m):
        gh, ga = self.exp(m['h'], m['a'])
        return {self.name + '_d': math.log(gh / ga), self.name + '_s': math.log(gh * ga) / 2 - 0.137}

    def post(self, m):
        h, a = m['h'], m['a']
        gh, ga = self.exp(h, a)
        eh = min(m['hg'], 5) - gh; ea = min(m['ag'], 5) - ga
        H, A = self.r[h], self.r[a]; w, c = self.w, self.cross
        H[0] += w * eh; A[3] += w * eh; A[2] += w * ea; H[1] += w * ea
        H[2] += c * w * eh; A[1] += c * w * eh; A[0] += c * w * ea; H[3] += c * w * ea


class GAP:
    """GAP ratings (Wheatcroft 2020) on a count statistic (shots, shots on target, corners)."""

    def __init__(self, name, key, lam, mh, ma, phi=0.6, carry=0.7, dpromo=0.06):
        self.name, self.key, self.lam, self.mh, self.ma, self.phi, self.carry, self.dpromo = name, key, lam, mh, ma, phi, carry, dpromo
        self.r = {}  # team -> [Ha, Hd, Aa, Ad]
        self.base = math.log(mh / ma)

    def roll(self, t, info):
        if t not in self.r: self.r[t] = [self.mh, self.ma, self.ma, self.mh]
        r = self.r[t]; init = [self.mh, self.ma, self.ma, self.mh]
        c = self.carry ** info['gap']
        for i in range(4): r[i] = init[i] + c * (r[i] - init[i])
        if not info['new'] and info['from'] != info['lg']:
            s = -1 if info['lg'] == 'A' else 1
            r[0] *= 1 + s * self.dpromo; r[2] *= 1 + s * self.dpromo; r[1] *= 1 - s * self.dpromo; r[3] *= 1 - s * self.dpromo

    def exp(self, h, a):
        H, A = self.r[h], self.r[a]
        return (H[0] + A[3]) / 2, (A[2] + H[1]) / 2

    def pre(self, m):
        sh, sa = self.exp(m['h'], m['a'])
        return {self.name + '_d': math.log(max(sh, 0.5) / max(sa, 0.5)) - self.base}

    def post(self, m):
        d = m['d']
        if self.key + '_h' not in d or d[self.key + '_h'] is None or d[self.key + '_a'] is None: return
        H, A = self.r[m['h']], self.r[m['a']]
        sh, sa = self.exp(m['h'], m['a'])
        eh = d[self.key + '_h'] - sh; ea = d[self.key + '_a'] - sa
        l, p = self.lam, self.phi
        H[0] = max(H[0] + l * p * eh, 0); H[2] = max(H[2] + l * (1 - p) * eh, 0)
        H[1] = max(H[1] + l * p * ea, 0); H[3] = max(H[3] + l * (1 - p) * ea, 0)
        A[2] = max(A[2] + l * p * ea, 0); A[0] = max(A[0] + l * (1 - p) * ea, 0)
        A[3] = max(A[3] + l * p * eh, 0); A[1] = max(A[1] + l * (1 - p) * eh, 0)


class PossRating:
    """Elo-like rating on possession share (not used by v1)."""

    def __init__(self, k=0.08, carry=0.7):
        self.k, self.carry = k, carry; self.P = defaultdict(float); self.hfa = defaultdict(lambda: 1.0)

    def roll(self, t, info):
        self.P[t] = 0.0 if info['new'] else self.P[t] * self.carry ** info['gap']

    def pre(self, m):
        return {'pos_d': self.P[m['h']] - self.P[m['a']]}

    def post(self, m):
        d = m['d']
        if not d.get('po_h'): return
        e = d['po_h'] - (50 + self.hfa[m['lg']] + self.P[m['h']] - self.P[m['a']])
        self.P[m['h']] += self.k * e; self.P[m['a']] -= self.k * e; self.hfa[m['lg']] += 0.005 * e


# ----------------------------------------------------------------------------- form (EW means of v1 residuals)
class Form:
    HLS = (4, 12, 40)
    RES = ('gd_r', 'pts_r', 'gda_r', 'gnp_r', 'xgd_r', 'shr_r', 'sotr_r', 'cor_r', 'dom_r', 'gfc_r', 'ht_r', 'first_r')
    RAW = ('gd_raw', 'pts_raw', 'xgd_raw', 'pos_raw')
    VEN = ('gd_r', 'xgd_r', 'pts_r')
    VHL = (10, 40)

    def __init__(self, season_carry=0.6):
        self.carry = season_carry
        self.all = {(k, hl): EW(hl) for k in self.RES + self.RAW + ('gf_r', 'ga_r') for hl in self.HLS}
        self.ven = {(k, hl): EW(hl) for k in self.VEN for hl in self.VHL}
        self.lgm = {}  # league means of raw metrics (home perspective), slow EW
        self.statw = EW(12, k0=0.0)

    def roll(self, t, info):
        f = self.carry ** info['gap']
        for (k, hl), e in self.all.items():
            e.scale(t, 0.0 if (k in self.RAW and (info['new'] or info['from'] != info['lg'])) else f)
        for e in self.ven.values():
            e.scale((t, 'H'), f); e.scale((t, 'A'), f)

    def pre(self, m):
        h, a = m['h'], m['a']; o = {}
        for (k, hl), e in self.all.items():
            if k in ('gf_r', 'ga_r'): continue
            o[f'f_{k}_{hl}'] = e.get(h) - e.get(a)
        for hl in self.HLS:  # attack/defence split (for the Poisson learner)
            gf, ga = self.all[('gf_r', hl)], self.all[('ga_r', hl)]
            o[f'f_hatt_{hl}'] = gf.get(h) + ga.get(a); o[f'f_aatt_{hl}'] = gf.get(a) + ga.get(h)
            o[f'f_tot_r_{hl}'] = gf.get(h) + ga.get(a) + gf.get(a) + ga.get(h)   # both teams' recent total goals vs v1 expectation
        for (k, hl), e in self.ven.items():
            o[f'fv_{k}_{hl}'] = e.get((h, 'H')) - e.get((a, 'A'))
        o['f_statw'] = min(self.statw.W[h], self.statw.W[a]) / 17.0
        return o

    def _lg(self, lg, key, x, init):
        s = self.lgm.setdefault((lg, key), init)
        self.lgm[(lg, key)] = s + 0.004 * (x - s)
        return s

    def post(self, m):
        h, a = m['h'], m['a']; d = m['d']; v = m['v']; lg = m['lg']
        lh, la = v['lh'], v['la']; hg, ag = m['hg'], m['ag']
        P = {}  # home-perspective antisymmetric metrics
        P['gd_r'] = (hg - ag) - (lh - la)
        p = m['pv1']; pts = 3 if hg > ag else (1 if hg == ag else 0)
        ptsa = 3 if ag > hg else (1 if hg == ag else 0)
        P['pts_r'] = ((pts - (3 * p[0] + p[1])) - (ptsa - (3 * p[2] + p[1]))) / 2
        P['gd_raw'] = (hg - ag) - self._lg(lg, 'gd', hg - ag, 0.55)
        P['pts_raw'] = (pts - ptsa) - self._lg(lg, 'pts', pts - ptsa, 0.8)
        if d.get('evOk'):
            P['gda_r'] = (hg - lh * d['Eh']) - (ag - la * d['Ea'])           # red-card & score-state adjusted residual
            P['gnp_r'] = (hg - d['pen_h']) - (ag - d['pen_a']) - 0.92 * (lh - la)  # non-penalty goals
            P['dom_r'] = d['dom'] - self._lg(lg, 'dom', d['dom'], 0.1) - 0.25 * (lh - la - 0.5)  # share of time leading
            P['gfc_r'] = (d['comp_h'] - d['comp_a']) - 0.85 * (lh - la)      # goals scored while the game was close (|diff|<=1)
            P['ht_r'] = (d['ht_h'] - d['ht_a']) - 0.44 * (lh - la)             # half-time goal difference
            P['first_r'] = d['first'] - (p[0] - p[2]) * 1.15                 # who scored first
        if 'xg_h' in d:
            xr = (d['xg_h'] - d['xg_a']) - (lh - la)
            P['xgd_r'] = xr - self._lg(lg, 'xgdr', xr, -0.1)
            P['xgd_raw'] = (d['xg_h'] - d['xg_a']) - self._lg(lg, 'xgd', d['xg_h'] - d['xg_a'], 0.4)
            P['shr_r'] = math.log((d['sh_h'] + 1) / (d['sh_a'] + 1)) - math.log((v['sh_lh'] + 1) / (v['sh_la'] + 1))
            P['sotr_r'] = math.log((d['sot_h'] + 1) / (d['sot_a'] + 1)) - math.log((v['sot_lh'] + 1) / (v['sot_la'] + 1))
            if d.get('co_h') is not None and d.get('co_a') is not None:
                P['cor_r'] = math.log((d['co_h'] + 1) / (d['co_a'] + 1)) - math.log((v['co_lh'] + 1) / (v['co_la'] + 1))
            if d.get('po_h'):
                P['pos_raw'] = (d['po_h'] - 50) - self._lg(lg, 'pos', d['po_h'] - 50, 1.0)
            self.statw.add(h, 1.0); self.statw.add(a, 1.0)
        for (k, hl), e in self.all.items():
            if k == 'gf_r': e.add(h, hg - lh); e.add(a, ag - la)
            elif k == 'ga_r': e.add(h, ag - la); e.add(a, hg - lh)
            elif k in P: e.add(h, P[k]); e.add(a, -P[k])
        for (k, hl), e in self.ven.items():
            if k in P: e.add((h, 'H'), P[k]); e.add((a, 'A'), -P[k])


# ----------------------------------------------------------------------------- novelty / schedule
class Novelty:
    def __init__(self):
        self.n = defaultdict(int); self.last_t = {}; self.flag = {}; self.hist = defaultdict(list); self.nseas = defaultdict(int)

    def roll(self, t, info):
        self.n[t] = 0; self.nseas[t] += 1
        self.flag[t] = dict(promo=float((not info['new'] and info['from'] == 'B' and info['lg'] == 'A') or (info['new'] and info['lg'] == 'A' and info['season'] > 2007)),
                            releg=float(not info['new'] and info['from'] == 'A' and info['lg'] == 'B'),
                            new=float(info['new'] or info['gap'] > 1))
        self.last_t.pop(t, None)

    def pre(self, m):
        h, a = m['h'], m['a']; o = {}
        nh, na = self.n[h], self.n[a]
        o['n_h'] = nh; o['n_a'] = na
        o['n_early'] = math.exp(-min(nh, na) / 6.0)
        fh, fa = self.flag[h], self.flag[a]
        for k in ('promo', 'releg', 'new'):
            o[f'n_{k}_d'] = fh[k] - fa[k]
            o[f'n_{k}e_d'] = fh[k] * math.exp(-nh / 10.0) - fa[k] * math.exp(-na / 10.0)
        rh = (m['t'] - self.last_t[h]) / 24.0 if h in self.last_t else None
        ra = (m['t'] - self.last_t[a]) / 24.0 if a in self.last_t else None
        c = lambda r: 7.0 if r is None else min(r, 12.0)
        o['n_rest_d'] = c(rh) - c(ra)
        o['n_short_d'] = float(rh is not None and rh < 3.6) - float(ra is not None and ra < 3.6)
        o['n_long_d'] = float(rh is not None and rh > 13) - float(ra is not None and ra > 13)
        cnt = lambda t: sum(1 for x in self.hist[t][-6:] if m['t'] - x <= 15 * 24)
        o['n_cong_d'] = cnt(h) - cnt(a)
        o['n_exp_d'] = min(self.nseas[h], 8) - min(self.nseas[a], 8)   # seasons of history in the data base
        ts = pd.Timestamp(m['t'] * 3600, unit='s')
        o['n_midweek'] = float(ts.day_of_week in (1, 2, 3))
        return o

    def post(self, m):
        for t in (m['h'], m['a']):
            self.n[t] += 1; self.last_t[t] = m['t']; self.hist[t].append(m['t'])


# ----------------------------------------------------------------------------- head-to-head and venue
class H2HVenue:
    def __init__(self):
        self.h2h = EW(6, k0=3.0); self.h2hh = EW(4, k0=3.0)
        self.tv = EW(30, k0=8.0)          # (home team, venue) residual
        self.recent = defaultdict(list)
        self.hl = EW(60, k0=10.0); self.al = EW(60, k0=10.0)  # long-run home / away residual per team
        self.hl2 = EW(150, k0=15.0); self.al2 = EW(150, k0=15.0)

    def roll(self, t, info): pass

    def pre(self, m):
        h, a = m['h'], m['a']; o = {}
        key = (min(h, a), max(h, a)); sgn = 1 if h < a else -1
        o['h2h_r'] = sgn * self.h2h.get(key)
        o['h2h_home_r'] = self.h2hh.get((h, a))
        ven = m.get('venue')
        o['ven_team_r'] = self.tv.get((h, ven)) if ven else 0.0
        rec = self.recent[h][-20:]
        o['ven_fam'] = (sum(1 for x in rec if x == ven) / len(rec) - 0.7) if (ven and len(rec) >= 8) else 0.0
        o['ven_hlong'] = self.hl.get(h); o['ven_along'] = self.al.get(a)
        o['ven_long_d'] = self.hl.get(h) - self.al.get(a)
        o['ven_split_d'] = (self.hl2.get(h) - self.al2.get(h)) + (self.hl2.get(a) - self.al2.get(a))  # both teams' home-vs-away asymmetry
        ch, ca = TEAM_CITY.get(h), TEAM_CITY.get(a)
        cm = norm_city(m.get('city')) or ch          # where the game is actually played (fallback: home base)
        dist = km(cm, ca) if cm and ca else 800.0     # trip of the away team
        dhome = km(cm, ch) if cm and ch else 0.0      # trip of the home team (sold / moved home games)
        o['ven_dist'] = math.log1p(dist / 100.0) - 2.0
        o['ven_derby'] = float(dist < 60)
        o['ven_far'] = float(dist > 1500)
        o['ven_neutral'] = float(dhome > 150)
        return o

    def post(self, m):
        h, a = m['h'], m['a']; v = m['v']
        r = (m['hg'] - m['ag']) - (v['lh'] - v['la'])
        r = max(-3.0, min(3.0, r))
        key = (min(h, a), max(h, a)); sgn = 1 if h < a else -1
        self.h2h.add(key, sgn * r); self.h2hh.add((h, a), r)
        ven = m.get('venue')
        if ven: self.tv.add((h, ven), r)
        self.recent[h].append(ven)
        self.hl.add(h, r); self.al.add(a, -r); self.hl2.add(h, r); self.al2.add(a, -r)


# ----------------------------------------------------------------------------- discipline / suspensions from the card stream
class Cards:
    def __init__(self):
        self.yel = defaultdict(int); self.goals = defaultdict(int); self.tg = defaultdict(int)
        self.susp = defaultdict(list); self.red = EW(12, k0=3.0)

    def roll(self, t, info):
        self.susp[t] = []

    def pre(self, m):
        h, a = m['h'], m['a']; s = m['season']
        def w(t):
            return sum(self.goals[(t, s, p)] for p in self.susp[t]) / (self.tg[(t, s)] + 5.0)
        return {'mn_susp_d': len(self.susp[h]) - len(self.susp[a]), 'mn_suspg_d': w(h) - w(a),
                'mn_red_d': self.red.get(h) - self.red.get(a)}

    def post(self, m):
        s = m['season']
        for side, t in (('h', m['h']), ('a', m['a'])):
            out = set(); nred = 0
            for e in m['ev']:
                if e['s'] != side: continue
                p = e.get('p')
                if e['t'] == 'R':
                    nred += 1
                    if p: out.add(p)
                elif e['t'] == 'Y' and p:
                    self.yel[(t, s, p)] += 1
                    if self.yel[(t, s, p)] % 3 == 0: out.add(p)
                elif e['t'] in 'GP' and p:
                    self.goals[(t, s, p)] += 1
            self.tg[(t, s)] += m['hg'] if side == 'h' else m['ag']
            self.susp[t] = list(out)
            self.red.add(t, nred - 0.12)


# ----------------------------------------------------------------------------- standings / stakes
class Standings:
    def __init__(self, teamsets):
        self.ts = teamsets; self.tab = {}

    def roll(self, t, info): pass

    def _status(self, lg, season, t, k):
        tab = self.tab.setdefault((lg, season), {x: [0, 0] for x in self.ts[(lg, season)]})
        if t not in tab: return 0.0, 0.0, 0.0
        P, n = tab[t]; R = 38 - n
        if R <= 0 or n < 10: return 0.0, 0.0, 0.0
        order = sorted(tab.values(), key=lambda x: -x[0])
        nt = len(order)
        top = 6 if lg == 'A' else 4

        def alive(cut):  # is the boundary between rank cut and cut+1 still (realistically) open for this team?
            above, below = order[cut - 1], order[cut]
            if P >= above[0]: return below[0] + k * (38 - below[1]) >= P
            return P + k * R >= above[0]
        rel = alive(nt - 4); topz = alive(top); title = alive(1)
        dead = not (rel or topz or title)
        late = n / 38.0
        return float(dead) * late, float(rel) * late, float(title or topz) * late

    def pre(self, m):
        o = {}
        for k, nm in ((3.0, 'h'), (1.8, 's')):
            dh = self._status(m['lg'], m['season'], m['h'], k); da = self._status(m['lg'], m['season'], m['a'], k)
            o[f'st_dead{nm}_d'] = dh[0] - da[0]; o[f'st_rel{nm}_d'] = dh[1] - da[1]; o[f'st_top{nm}_d'] = dh[2] - da[2]
            if nm == 'h': o['_dead_h'] = dh[0]; o['_dead_a'] = da[0]
        return o

    def post(self, m):
        tab = self.tab.setdefault((m['lg'], m['season']), {x: [0, 0] for x in self.ts[(m['lg'], m['season'])]})
        hg, ag = m['hg'], m['ag']
        for t, p in ((m['h'], 3 if hg > ag else (1 if hg == ag else 0)), (m['a'], 3 if ag > hg else (1 if hg == ag else 0))):
            if t in tab: tab[t][0] += p; tab[t][1] += 1


# ----------------------------------------------------------------------------- league-level drift of v1 residuals
class LeagueDrift:
    def __init__(self, hl=150):
        self.d = 0.5 ** (1.0 / hl); self.S = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    def roll(self, t, info): pass

    def pre(self, m):
        s = self.S[m['lg']]; w = s[3] + 40.0
        return {'lgd_hfa': s[0] / w, 'lgd_tot': s[1] / w, 'lgd_draw': s[2] / w}

    def post(self, m):
        s = self.S[m['lg']]; v = m['v']; hg, ag = m['hg'], m['ag']
        x = ((hg - ag) - (v['lh'] - v['la']), (hg + ag) - (v['lh'] + v['la']), float(hg == ag) - m['pv1'][1], 1.0)
        for i in range(4): s[i] = s[i] * self.d + x[i]


# ----------------------------------------------------------------------------- driver
def build(hide_from=None):
    """hide_from: ISO date; results of all matches kicking off at/after it are hidden (status -> 'pre'), for leak tests."""
    M = load_matches()
    if hide_from:
        for m in M:
            if m['date'] >= hide_from and m['status'] == 'post':
                m.update(status='pre', hg=None, ag=None, ev=[], stats=None, evOk=False, statsOk=False)
    B = load_base().set_index('id')
    grid = ChainGrid()
    M = [m for m in M if m['status'] in ('post', 'pre') and m['id'] in B.index]
    ids = [m['id'] for m in M]
    Bv = B.loc[ids]
    pv1 = grid(Bv.lh.values, Bv.la.values)
    vcols = ['lh', 'la', 'g_lh', 'g_la', 'g_Sh', 'g_Sa', 'sh_lh', 'sh_la', 'sot_lh', 'sot_la', 'co_lh', 'co_la', 'yc_lh', 'yc_la', 'fo_lh', 'fo_la', 'mu', 'hfa']
    Vv = Bv[vcols].to_dict('records')
    teamsets = defaultdict(set)
    for i, m in enumerate(M):
        m['t'] = pd.Timestamp(m['date']).timestamp() / 3600.0
        m['v'] = Vv[i]; m['pv1'] = pv1[i]; m['d'] = derive(m)
        teamsets[(m['lg'], m['season'])].add(m['h']); teamsets[(m['lg'], m['season'])].add(m['a'])

    gens = [Elo('elo10', 10), Elo('elo20', 20), Elo('elo35', 35),
            Elo('eloxg', 20, mode='xg', scale=1.0),
            Pi('pi035', 0.035), Pi('pi07', 0.07), Pi('pixg', 0.05, mode='xg'),
            Berrar('ber05', 0.05), Berrar('ber12', 0.12),
            GAP('gapsh', 'sh', 0.15, 13.8, 11.3), GAP('gapsot', 'sot', 0.15, 4.8, 3.85), GAP('gapco', 'co', 0.15, 5.6, 4.6),
            GAP('gapsh4', 'sh', 0.4, 13.8, 11.3), GAP('gapsot4', 'sot', 0.4, 4.8, 3.85),
            PossRating(), Form(), Novelty(), H2HVenue(), Cards(), Standings(teamsets), LeagueDrift()]

    last = {}  # team -> (season, lg)
    nstat = defaultdict(int)
    STAT_RATINGS = ['v_shd', 'v_sotd', 'v_cod', 'v_fod', 'eloxg_d', 'pixg_d', 'pixg_o', 'gapsh_d', 'gapsot_d', 'gapco_d', 'gapsh4_d', 'gapsot4_d', 'pos_d']
    rows = []; pending = []; k = 0
    for m in M:
        while k < len(pending) and pending[k]['t'] + DELAY_H <= m['t']:
            for g in gens: g.post(pending[k])
            if 'xg_h' in pending[k]['d']: nstat[pending[k]['h']] += 1; nstat[pending[k]['a']] += 1
            k += 1
        for t in (m['h'], m['a']):
            ls = last.get(t)
            if ls is None or ls[0] < m['season']:
                info = dict(new=ls is None, gap=(m['season'] - ls[0]) if ls else 0, lg=m['lg'], season=m['season'], **{'from': ls[1] if ls else None})
                for g in gens: g.roll(t, info)
                last[t] = (m['season'], m['lg'])
        v = m['v']
        f = dict(id=m['id'], lg=m['lg'], season=m['season'], date=m['date'], h=m['h'], a=m['a'], status=m['status'], hg=m['hg'], ag=m['ag'])
        # v1-derived base features
        f['v_lh'] = v['lh']; f['v_la'] = v['la']
        f['v_dl'] = math.log(v['lh'] / v['la']); f['v_sl'] = math.log(v['lh'] * v['la']) / 2
        f['v_gdl'] = math.log(v['g_lh'] / v['g_la']); f['v_gsl'] = math.log(v['g_lh'] * v['g_la']) / 2
        f['v_shd'] = math.log(v['sh_lh'] / v['sh_la']) - 0.2; f['v_sotd'] = math.log(v['sot_lh'] / v['sot_la']) - 0.22
        f['v_cod'] = math.log(v['co_lh'] / v['co_la']) - 0.2; f['v_ycd'] = math.log(v['yc_lh'] / v['yc_la']) + 0.1
        f['v_fod'] = math.log(v['fo_lh'] / v['fo_la'])
        f['v_S'] = v['g_Sh'] + v['g_Sa'] - 0.09; f['v_Sd'] = v['g_Sh'] - v['g_Sa']
        f['v_mu'] = v['mu'] - 0.2; f['v_slc'] = f['v_sl'] - v['mu']   # league scoring level / match-up 'openness'
        f['v_pH'], f['v_pD'], f['v_pA'] = m['pv1']
        for g in gens: f.update(g.pre(m))
        if nstat[m['h']] + nstat[m['a']] == 0:
            for c in STAT_RATINGS: f[c] = 0.0
        f['st_deadh_h'] = f.pop('_dead_h'); f['st_deadh_a'] = f.pop('_dead_a')
        rows.append(f)
        if m['status'] == 'post': pending.append(m)
    F = pd.DataFrame(rows)
    F['phase'] = (F.n_h + F.n_a) / 76.0
    return F


def leakcheck(F):
    """Hide every result from a cut date on and rebuild: rows of the matches kicking off at the cut must not change."""
    fc = [c for c in F.columns if c not in ('status', 'hg', 'ag')]
    for cut in ('2013-09-14T21:30Z', '2019-07-14T19:00Z', '2024-06-19T23:00Z', '2026-04-05T19:00Z'):
        G = build(hide_from=cut)
        a = F[(F.date >= cut)].head(8); b = G.set_index('id').loc[a.id].reset_index()
        same = (a[fc].reset_index(drop=True).select_dtypes('number') - b[fc].select_dtypes('number')).abs().max().max()
        first = a.date.iloc[0]
        print(f'leakcheck cut {cut}: first kick-off {first}, rows compared {len(a)}, max |diff| on rows at the cut = '
              f'{(a[a.date == first][fc].reset_index(drop=True).select_dtypes("number") - b[b.date == first][fc].reset_index(drop=True).select_dtypes("number")).abs().max().max():.2e}')


if __name__ == '__main__':
    F = build()
    if '--leakcheck' in sys.argv: leakcheck(F)
    F.to_pickle(Z + 'features_all.pkl')
    drop = ['lg', 'season', 'date', 'h', 'a', 'status', 'hg', 'ag']
    F.drop(columns=drop).round(5).to_csv(Z + 'features_zoo_all.csv', index=False)
    print(F.shape)
    print(F.describe().T[['count', 'mean', 'std', 'min', 'max']].round(3).to_string())

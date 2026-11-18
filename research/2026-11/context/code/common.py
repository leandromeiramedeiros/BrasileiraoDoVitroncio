"""Shared helpers for the context track: loading the merged match stream, team-id aliases,
geography (city -> coordinates, team home base), haversine distances."""
import json, os, math, unicodedata, collections
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED = '/home/user/bdv-lab/shared/'
DOMESTIC = {'A', 'B', 'C', 'D', 'SP', 'RJ', 'RS', 'MG', 'CDB', 'CNE', 'SUP'}
STATE_CAP = {'SP': 'sao paulo', 'RJ': 'rio de janeiro', 'RS': 'porto alegre', 'MG': 'belo horizonte'}

# ESPN uses more than one id for a few clubs (and, worse, re-uses an id for different clubs).
# (id, comp-or-None) -> canonical id
ALIAS = {
    ('5266', None): '3395',    # Goiás (id changed in 2014)
    ('6274', 'A'): '4929',     # Santa Cruz (Recife) in Série A 2006; 6274 in the Gaúcho is Santa Cruz-RS
    ('6080', None): '3448',    # Guarani (Campinas) 2006-2013
    ('7385', 'B'): '3448',     # Guarani (Campinas) Série B 2011; 7385 elsewhere = Guaraní (PAR) / Guarani-VA
    ('15577', None): '4875',   # Confiança
    ('17326', None): '18126',  # Ferroviária
    ('17328', None): '7375',   # Grêmio Barueri
    ('2256', None): '900',     # Boavista-RJ
    ('7385', 'RS'): '7385rs',  # Guarani de Venâncio Aires (keep apart from the Paraguayan club)
    ('6274', 'RS'): '6274rs',  # Santa Cruz-RS
}

def canon(tid, comp):
    tid = str(tid)
    return ALIAS.get((tid, comp)) or ALIAS.get((tid, None)) or tid

def norm(s):
    if s is None or (isinstance(s, float) and math.isnan(s)): return ''
    s = str(s).split(',')[0]
    s = ''.join(ch for ch in unicodedata.normalize('NFD', s.lower()) if unicodedata.category(ch) != 'Mn')
    return ' '.join(s.replace('ò', 'o').split())

def load_cities():
    c = pd.read_csv(os.path.join(HERE, 'city_coords.csv'), keep_default_na=False)
    return {r.key: (float(r.lat), float(r.lon), r.uf, r.country, r.region) for r in c.itertuples()}

CITIES = load_cities()

VENUE_CITY = [  # (substring of normalised venue, city key) used when the city is missing or wrong
    ('brinco de ouro', 'campinas'), ('moises lucarelli', 'campinas'), ('serra dourada', 'goiania'),
    ('mineirao', 'belo horizonte'), ('estadio independencia', 'belo horizonte'), ('claudio moac', 'macae'),
    ('prudentao', 'presidente prudente'), ('eduardo jose farah', 'presidente prudente'),
    ('fumeirao', 'arapiraca'), ('romeirao', 'juazeiro do norte'), ('vail chaves', 'mogi mirim'),
    ('ibere ferreira', 'santa cruz rn'), ('luiz jose de lacerda', 'caruaru'), ('lamengo', 'ipatinga'),
    ('nazarenao', 'goianinha'), ('kleber andrade', 'cariacica'),
]
BAD_CITY = {'rio grande do sul', 'goias', 'genova', 'campo bom_AB', 'santa maria_AB'}

def resolve_city(venue, city, country, comp):
    """-> city key or None. Applies venue overrides and rejects foreign venues in domestic competitions."""
    v, c = norm(venue), norm(city)
    for sub, key in VENUE_CITY:
        if sub in v: return key
    if v == 'jk' and comp in ('A', 'B'): return 'itumbiara'
    if v == 'arena da fonte': return None   # ESPN mixes Arena Fonte Nova (Salvador) with Arena da Fonte (Araraquara): use the home base
    if c == 'santa maria' and 'presidente vargas' in v: return 'fortaleza'
    if c == 'campo bom' and comp in ('A', 'B'): return None
    if c in BAD_CITY or not c: return None
    if c == 'santa cruz' and comp in DOMESTIC: return None
    if c not in CITIES: return None
    if comp in DOMESTIC and CITIES[c][3] != 'BR': return None
    return c

def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1; dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))

def load_all():
    """Merged all-competitions stream as a DataFrame with canonical team ids, timestamps and resolved venues."""
    M = pd.DataFrame(json.load(open(os.path.join(HERE, 'all_matches.json'))))
    M['h'] = [canon(t, c) for t, c in zip(M.h, M.comp)]
    M['a'] = [canon(t, c) for t, c in zip(M.a, M.comp)]
    M['ts'] = pd.to_datetime(M.date, utc=True)
    M['day'] = (M.ts - pd.Timestamp('2000-01-01', tz='UTC')).dt.total_seconds() / 86400.0
    M['year'] = M.ts.dt.year
    # ESPN season labels are off for two competitions: Libertadores editions up to 2015 carry year-1, and the
    # Carioca 2020 is labelled 2019 (preliminary rounds are played in December of the previous year).
    M['season_espn'] = M.season
    lib_fix = (M.comp == 'LIB') & (M.season <= 2014) & (M.year == M.season + 1)
    M.loc[lib_fix, 'season'] = M.loc[lib_fix, 'season'] + 1
    rj = M.comp == 'RJ'
    M.loc[rj, 'season'] = (M.year + (M.ts.dt.month == 12).astype(int))[rj]
    # league "season" for A/B (the 2020 season ended in Feb 2021); calendar year elsewhere
    M['ckey'] = [resolve_city(v, c, k, comp) for v, c, k, comp in zip(M.venue, M.city, M.country, M.comp)]
    # home base per team: most frequent resolved home city (per season, then overall, then state capital)
    known = M[M.ckey.notna() & ~M.neutral & ~M.comp.isin(['CWC', 'ICC', 'SUP', 'REC'])]
    base_all = known.groupby('h').ckey.agg(lambda s: s.value_counts().index[0]).to_dict()
    # season-specific base only for genuine relocations (>= 75 % of that season's home league matches in one city);
    # otherwise the all-time mode (a club that sells some home matches to another city keeps its base)
    lg = known[known.comp.isin(['A', 'B', 'C'])]
    base_season = {}
    for (t, y), sub in lg.groupby(['h', 'season']):
        vc = sub.ckey.value_counts()
        if vc.iloc[0] >= 0.75 * len(sub) and len(sub) >= 10: base_season[(t, y)] = vc.index[0]
    state_of = {}
    for comp, cap in STATE_CAP.items():
        for t in set(M[M.comp == comp].h) | set(M[M.comp == comp].a):
            state_of.setdefault(t, cap)
    def base(t, season):
        return base_season.get((t, season)) or base_all.get(t) or state_of.get(t)
    M['hbase'] = [base(t, s) for t, s in zip(M.h, M.season)]
    M['abase'] = [base(t, s) for t, s in zip(M.a, M.season)]
    # venue fallback: home team's base (unknown/implausible venue)
    M['vkey'] = M.ckey.where(M.ckey.notna(), M.hbase)
    M = M.sort_values(['day', 'id']).reset_index(drop=True)
    return M

def coords(key):
    if key is None or (isinstance(key, float) and math.isnan(key)) or key not in CITIES: return (np.nan, np.nan)
    return CITIES[key][0], CITIES[key][1]

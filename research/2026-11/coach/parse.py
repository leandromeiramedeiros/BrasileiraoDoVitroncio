"""Parse Wikipedia "Managerial changes" / "Coaching changes" (en) and "Mudança(s) de técnicos" (pt) tables.
   /home/user/bdv-lab/venv/bin/python -I code/parse.py          (run from /home/user/bdv-lab/coach)
Reads raw/{en,pt}_{A,B}_YYYY.json, writes coach_changes.csv (+ coach_changes_pt.csv, parse_log.txt).
English is the primary source; Portuguese is used for seasons/divisions that have no English table and for spot checks."""
import json, re, os, sys, unicodedata, datetime as dt
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED = '/home/user/bdv-lab/shared/'
LOG = []
def log(*a):
    s = ' '.join(str(x) for x in a); LOG.append(s); print(s)

# ------------------------------------------------------------------ wikitext helpers
def strip_balanced(s, open_, close, keep=None):
    """Remove (or transform with keep(inner)) balanced open_...close_ spans, innermost first."""
    pat = re.compile(re.escape(open_) + r'((?:(?!' + re.escape(open_) + r'|' + re.escape(close) + r').)*)' + re.escape(close), re.S)
    for _ in range(20):
        new = pat.sub(lambda m: keep(m.group(1)) if keep else '', s)
        if new == s: break
        s = new
    return s

def tmpl(inner):
    """Template -> visible text (rough)."""
    parts = [p.strip() for p in inner.split('|')]
    name = parts[0].strip(); low = name.lower()
    if low in ('flagicon', 'fb', 'flag icon', 'flagdeco', 'brab', 'bra', 'por', 'arg', 'uru', 'esp', 'col', 'chi', 'par', 'ref_label', 'ref label', 'efn', 'refn', 'sfn', 'cite web', 'cite news', 'citar web', 'cite', 'nota', 'notelist'):
        return ''
    if re.fullmatch(r'[A-Z]{3}b?', name): return ''          # {{BRAb}}, {{ARGb}}, {{POR}} flag templates (pt)
    if low.startswith('futebol '):                             # {{Futebol Vitória|estado=antes}} (pt team template)
        return '@@TEAM:' + name[8:].strip() + '@@'
    if low in ('nowrap', 'small', 'nobr', 'center', 'big', 'sortname', 'tooltip', 'abbr', 'nowrap begin'):
        vis = [p for p in parts[1:] if '=' not in p]
        if low == 'sortname': return ' '.join(vis[:2])
        if low in ('tooltip', 'abbr'): return vis[0] if vis else ''
        return ' '.join(vis)
    if low in ('dts', 'date', 'dtsort', 'sort'):
        vis = [p for p in parts[1:] if '=' not in p]
        return ' '.join(vis[-1:]) if low == 'sort' else ' '.join(vis)
    if low in ('start date', 'start date and age'):
        vis = [p for p in parts[1:] if '=' not in p]
        return '-'.join(vis[:3]) if len(vis) >= 3 else ''
    return ''

def clean(s):
    s = re.sub(r'<ref[^>]*/>', '', s)
    s = re.sub(r'<ref[^>]*>.*?</ref>', '', s, flags=re.S)
    s = re.sub(r'<!--.*?-->', '', s, flags=re.S)
    s = re.sub(r'<sup>.*?</sup>', '', s, flags=re.S)
    s = strip_balanced(s, '{{', '}}', tmpl)
    s = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]]*)\]\]', r'\1', s)
    s = re.sub(r'<br\s*/?>', ' ', s)
    s = re.sub(r'<[^>]+>', '', s)
    s = s.replace("'''", '').replace("''", '').replace('&nbsp;', ' ')
    return re.sub(r'\s+', ' ', s).strip()

def split_attr(cell):
    """'rowspan=2 | text' -> (attrs, text). Ignore pipes inside [[..]] / {{..}}."""
    depth = 0; i = 0
    while i < len(cell):
        two = cell[i:i + 2]
        if two in ('[[', '{{'): depth += 1; i += 2; continue
        if two in (']]', '}}'): depth -= 1; i += 2; continue
        if cell[i] == '|' and depth == 0 and cell[i:i + 2] != '||':
            attr = cell[:i]
            if re.search(r'(rowspan|colspan|align|style|width|class|bgcolor|data-sort-value)\s*=', attr, re.I) or not attr.strip():
                return attr, cell[i + 1:]
            return '', cell
        i += 1
    return '', cell

def split_inline(line, sep):
    """Split a row line on '||' (or '!!') outside [[..]]/{{..}}."""
    out, depth, cur, i = [], 0, '', 0
    while i < len(line):
        two = line[i:i + 2]
        if two in ('[[', '{{'): depth += 1; cur += two; i += 2; continue
        if two in (']]', '}}'): depth -= 1; cur += two; i += 2; continue
        if two == sep and depth == 0: out.append(cur); cur = ''; i += 2; continue
        cur += line[i]; i += 1
    out.append(cur); return out

def parse_table(tab):
    """Wikitext table -> (header list, list of row lists of raw cell text) with rowspans expanded."""
    tab = re.sub(r'<ref[^>]*/>', '', tab)                         # self-closing first, else it swallows text
    tab = re.sub(r'<ref[^>]*>.*?</ref>', '', tab, flags=re.S)       # refs can contain newlines and pipes
    body = tab.split('\n', 1)[1].rsplit('|}', 1)[0]
    rows = re.split(r'\n\|-[^\n]*', '\n' + body)
    header, data, pending = None, [], {}
    for r in rows:
        lines = [l for l in r.split('\n') if l.strip()]
        cells, is_head = [], False
        for l in lines:
            if l.startswith('|+'): continue
            if l.startswith('!'):
                is_head = True; cells += split_inline(l[1:], '!!')
            elif l.startswith('|'):
                cells += split_inline(l[1:], '||')
            elif cells:
                cells[-1] += '\n' + l
        if not cells: continue
        if is_head and header is None:
            header = [clean(split_attr(c)[1]) for c in cells]; continue
        # expand rowspans
        row, ci, k = [], 0, 0
        ncols = len(header) if header else 99
        parsed = [split_attr(c) for c in cells]
        while k < len(parsed) or ci in pending:
            if ci in pending:
                txt, left = pending[ci]; row.append(txt)
                if left <= 1: del pending[ci]
                else: pending[ci] = (txt, left - 1)
                ci += 1; continue
            if k >= len(parsed): break
            attr, txt = parsed[k]; k += 1
            m = re.search(r'rowspan\s*=\s*"?(\d+)', attr, re.I)
            if m and int(m.group(1)) > 1: pending[ci] = (txt, int(m.group(1)) - 1)
            cs = re.search(r'colspan\s*=\s*"?(\d+)', attr, re.I)
            row.append(txt); ci += 1
            if cs:
                for _ in range(int(cs.group(1)) - 1): row.append(txt); ci += 1
            if ci > ncols + 3: break
        data.append(row)
    return header or [], data

def find_tables(w, lang):
    pat = r'^(=+)\s*(Managerial changes|Coaching changes|Head coach changes|Manager changes|Mudanças? de t[ée]cnicos?|Mudanças? de treinadores?|Troca de t[ée]cnicos)\s*=+\s*$'
    m = re.search(pat, w, re.M | re.I)
    if not m: return None, []
    lvl = len(m.group(1)); rest = w[m.end():]
    nxt = re.search(r'^={1,%d}[^=].*?=+\s*$' % lvl, rest, re.M)
    sec = rest[:nxt.start()] if nxt else rest
    tabs, i = [], 0
    while True:
        a = sec.find('{|', i)
        if a < 0: break
        depth, j = 0, a
        while j < len(sec):
            if sec.startswith('{|', j): depth += 1; j += 2; continue
            if sec.startswith('|}', j):
                depth -= 1; j += 2
                if depth == 0: break
                continue
            j += 1
        tabs.append(sec[a:j]); i = j
    return m.group(2), tabs

# ------------------------------------------------------------------ columns / dates
COLS = {
    'team': r'^(team|club|clube|equipe|time)\b',
    'outgoing': r'outgoing|antecessor|sa[ií]da|t[ée]cnico que saiu|deixou|^saiu',
    'manner': r'manner|motivo|forma|raz[aã]o',
    'date_vacancy': r'date of vacancy|^date$|^data$|data da sa[ií]da|data de sa[ií]da|^data \(sa',
    'position': r'position|table|^pos|posi[cç][aã]o',
    'incoming': r'incoming|sucessor|substituto|entrada|novo t[ée]cnico|^chegou',
    'date_appointment': r'date of appointment|data da contrata|data de chegada|data da entrada|data de entrada|^data \(che',
    'round': r'^round$|^rod',
}
def map_cols(header):
    out = {}
    for i, h in enumerate(header):
        hl = h.lower().strip()
        for k, p in COLS.items():
            if k in out: continue
            if re.search(p, hl):
                if k == 'date_vacancy' and re.search('appoint|contrata|chegada|entrada', hl): continue
                out[k] = i; break
    return out

MON = {m: i + 1 for i, m in enumerate(['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'])}
MON.update({m: i + 1 for i, m in enumerate(['janeiro', 'fevereiro', 'março', 'abril', 'maio', 'junho', 'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro'])})
MON.update({'marco': 3, 'sept': 9})
MON.update({k[:3]: v for k, v in list(MON.items())})

def parse_date(s, season, span):
    s0 = s; s = s.lower().replace('º', '').replace('°', '')
    m = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', s)
    if m: return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))), True
    m = re.search(r'(\d{1,2})(?:st|nd|rd|th)?\s*(?:de\s+)?([a-zç]+)\.?(?:\s*(?:de\s+)?,?\s*(\d{4}))?', s)
    if not m:
        m2 = re.search(r'([a-z]+)\s+(\d{1,2}),?\s*(\d{4})?', s)       # "May 16, 2009"
        if not m2 or m2.group(1) not in MON: return None, False
        mon, day, yr = MON[m2.group(1)], int(m2.group(2)), m2.group(3)
    else:
        if m.group(2) not in MON: return None, False
        mon, day, yr = MON[m.group(2)], int(m.group(1)), m.group(3)
    if yr: y, explicit = int(yr), True
    else:
        explicit = False; y = season
        first, last = span
        try: cand = dt.date(season, mon, day)
        except ValueError: return None, False
        # seasons that run into the next calendar year (2020: Aug 2020 - Feb 2021)
        if cand < first - dt.timedelta(days=150) and last.year > season: y = season + 1
    try: return dt.date(y, mon, day), explicit
    except ValueError: return None, False

def manner_class(s):
    s = s.lower()
    if re.search(r'caretaker|interim|interin|tamp[aã]o|end of (temporary|caretaker|interim)|fim (do|da) interin|remanejad|returned to|voltou|retorno', s): return 'interim end'
    if re.search(r'mutual|comum acordo|consent', s): return 'mutual'
    if re.search(r'sack|dismiss|fired|demit|demiss|replaced|released|desligad|afastad', s): return 'sacked'
    if re.search(r'resign|pediu|demiss[aã]o volunt|deixou|left|quit|saiu|stepped', s): return 'resigned'
    if re.search(r'signed|hired|accepted|contratado|acerto|transfer|join|assumir|seleç|national team|appointed as', s): return 'poached'
    if re.search(r'end of contract|contract expired|fim de contrato|t[ée]rmino|contract end|not renew|n[aã]o renov', s): return 'contract end'
    if re.search(r'health|sa[uú]de|death|faleci|retire|aposent|ill', s): return 'other'
    return 'other' if s.strip() else ''

INTERIM_RE = re.compile(r'\b(caretaker|interim|interino|interina|tamp[aã]o)\b|\(c\)|\(i\)', re.I)

# ------------------------------------------------------------------ team mapping
def norm(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9]', '', s)

ALIASES = {  # normalized wiki name -> canonical key
    'athleticoparanaense': 'cap', 'atleticoparanaense': 'cap', 'athleticopr': 'cap', 'atleticopr': 'cap', 'athletico': 'cap', 'clubeatleticoparanaense': 'cap',
    'atleticomineiro': 'cam', 'atleticomg': 'cam', 'clubeatleticomineiro': 'cam', 'galo': 'cam',
    'atleticogoianiense': 'ago', 'atleticogo': 'ago', 'atleticoclubegoianiense': 'ago',
    'americamineiro': 'amg', 'americamg': 'amg', 'americafuteboclube': 'amg', 'americafc': 'amg', 'americafutebolclube': 'amg',
    'americarn': 'arn', 'americadenatal': 'arn', 'americafutebolclubern': 'arn',
    'redbullbragantino': 'bra', 'bragantino': 'bra', 'rbbragantino': 'bra', 'clubeatleticobragantino': 'bra',
    'vascodagama': 'vas', 'vasco': 'vas', 'crvascodagama': 'vas',
    'botafogo': 'bot', 'botafogorj': 'bot', 'botafogodefutebolceregatas': 'bot',
    'botafogosp': 'bfc', 'botafogoribeiraopreto': 'bfc', 'botafogofutebolclube': 'bfc',
    'gremiobarueri': 'gba', 'gremioprudente': 'gba', 'prudente': 'gba', 'gremiobar': 'gba', 'barueri': 'gba',
    'gremio': 'gre', 'gremiofootballportoalegrense': 'gre',
    'guarani': 'gua', 'guaranisp': 'gua', 'guaranifc': 'gua', 'guaranifutebolclube': 'gua',
    'operario': 'ope', 'operarioferroviario': 'ope', 'operariopr': 'ope', 'operarioferroviarioesporteclube': 'ope',
    'novorizontino': 'nov', 'gremionovorizontino': 'nov',
    'brasildepelotas': 'brs', 'brasilpelotas': 'brs', 'brasil': 'brs',
    'boaesporte': 'boa', 'boa': 'boa',
    'sampaiocorrea': 'sam', 'santacruz': 'stc', 'santoandre': 'sta',
    'duquedecaxias': 'dqc', 'ituiutaba': 'itb', 'ituiutabaesporteclube': 'itb', 'guaratingueta': 'grt',
    'saocaetano': 'sca', 'saobento': 'sbn', 'saobernardo': 'ber', 'saopaulo': 'sao',
    'athletic': 'ath', 'athleticclub': 'ath', 'ferroviaria': 'fer', 'voltaredonda': 'vre',
    'campinense': 'cmp', 'campinensepb': 'cmp', 'mogimirim': 'mog', 'marilia': 'mar', 'mariliaac': 'mar',
    'chapecoense': 'cha', 'associacaochapecoensedefutebol': 'cha', 'ceara': 'cea', 'cuiaba': 'cui',
    'figueirense': 'fig', 'internacional': 'int', 'inter': 'int', 'portuguesa': 'ptg', 'paysandu': 'pay', 'parana': 'par', 'paranaclube': 'par',
    'pontepreta': 'pon', 'luverdense': 'luv', 'londrina': 'lon', 'oeste': 'oes', 'macae': 'mac', 'tombense': 'tom', 'tupi': 'tup',
    'confianca': 'con', 'brusque': 'brus', 'ituano': 'itu', 'icasa': 'ica', 'ipatinga': 'ipa', 'joinville': 'jec', 'juventude': 'juv',
    'mirassol': 'mir', 'nautico': 'nau', 'remo': 'remo', 'salgueiro': 'sal', 'amazonas': 'ama', 'abc': 'abc', 'asa': 'asa',
    'avai': 'ava', 'bahia': 'bah', 'brasiliense': 'bras', 'crb': 'crb', 'csa': 'csa', 'corinthians': 'cor', 'coritiba': 'cfc',
    'criciuma': 'cri', 'cruzeiro': 'cru', 'flamengo': 'fla', 'fluminense': 'flu', 'fortaleza': 'for', 'gama': 'gama', 'goias': 'goi',
    'palmeiras': 'pal', 'santos': 'san', 'sport': 'spo', 'sportrecife': 'spo', 'vilanova': 'vil', 'vitoria': 'vit',
}
TEAM_KEY = {  # ESPN id -> canonical key (duplicates share a key)
    '9965': 'abc', '6196': 'asa', '21888': 'ama', '10982': 'amg', '6154': 'amg', '7634': 'arn', '20851': 'ath', '3458': 'cap', '10357': 'ago',
    '7632': 'cam', '9966': 'ava', '9967': 'bah', '15560': 'boa', '6086': 'bot', '10281': 'bfc', '6079': 'bra', '17134': 'brs', '9968': 'bras',
    '18428': 'brus', '9970': 'crb', '7531': 'csa', '7509': 'cmp', '9969': 'cea', '9318': 'cha', '4875': 'con', '874': 'cor', '3456': 'cfc',
    '9971': 'cri', '2022': 'cru', '17313': 'cui', '9175': 'dqc', '18126': 'fer', '3461': 'fig', '819': 'fla', '3445': 'flu', '6272': 'for',
    '9972': 'gama', '3395': 'goi', '5266': 'goi', '7375': 'gba', '6273': 'gre', '3448': 'gua', '6080': 'gua', '7385': 'gua', '7376': 'grt',
    '10983': 'ica', '1936': 'int', '4905': 'ipa', '4774': 'itu', '7486': 'itb', '13143': 'jec', '6270': 'juv', '17333': 'lon', '15576': 'luv',
    '9176': 'mac', '4782': 'mar', '9169': 'mir', '4777': 'mog', '18127': 'nov', '7633': 'nau', '10263': 'oes', '18187': 'ope', '2029': 'pal',
    '3453': 'par', '15424': 'pay', '3459': 'pon', '4773': 'ptg', '4936': 'remo', '1361': 'sal', '4873': 'sam', '4929': 'stc', '6274': 'stc',
    '4813': 'sta', '2674': 'san', '7635': 'spo', '6083': 'sbn', '11268': 'ber', '6084': 'sca', '2026': 'sao', '17339': 'tom', '15575': 'tup',
    '3454': 'vas', '9973': 'vil', '3457': 'vit', '4806': 'vre',
}
CANON = {'5266': '3395', '6080': '3448', '6274': '4929'}

def team_key(raw):
    m = re.search(r'@@TEAM:(.*?)@@', raw)
    s = m.group(1) if m else raw
    n = norm(s)
    if n in ALIASES: return ALIASES[n]
    s = re.sub(r'\(.*?\)', '', s)
    n = norm(s)
    if n in ALIASES: return ALIASES[n]
    for k in sorted(ALIASES, key=len, reverse=True):
        if len(k) >= 4 and n.startswith(k): return ALIASES[k]
    return None

def main():
    M = pd.DataFrame(json.load(open(SHARED + 'matches.json')))
    M['d'] = (pd.to_datetime(M.date, utc=True) - pd.Timedelta(hours=3)).dt.date
    span = {(r.lg, r.season): (g.min(), g.max()) for r, g in [(type('k', (), {'lg': k[0], 'season': k[1]}), v) for k, v in M.groupby(['lg', 'season']).d]}
    present = {}
    for (lg, s), g in M.groupby(['lg', 'season']):
        ids = set(g.h.astype(str)) | set(g.a.astype(str))
        present[(lg, s)] = {TEAM_KEY.get(i): i for i in ids if TEAM_KEY.get(i)}
    allp = {}
    for (lg, s), d in present.items():
        for k, i in d.items(): allp.setdefault(s, {}).setdefault(k, (lg, i))
    recs = []
    for lang in ('en', 'pt'):
        for lg in ('A', 'B'):
            for season in range(2009, 2027):
                f = os.path.join(HERE, 'raw', f'{lang}_{lg}_{season}.json')
                if not os.path.exists(f): log(f'{lang} {lg} {season}: not fetched'); continue
                try: J = json.load(open(f))
                except Exception: log(f'{lang} {lg} {season}: not JSON (rate-limited / refused)'); continue
                if 'error' in J: log(f'{lang} {lg} {season}: page missing ({J["error"].get("code")})'); continue
                w = J['parse']['wikitext']
                title, tabs = find_tables(w, lang)
                if not tabs: log(f'{lang} {lg} {season}: no coach-change table'); continue
                n0 = len(recs); unk = set()
                if (lg, season) not in span: continue
                for tab in tabs:
                    header, rows = parse_table(tab)
                    cm = map_cols(header)
                    if 'team' not in cm or 'outgoing' not in cm:
                        log(f'  {lang} {lg} {season}: table skipped, header={header}'); continue
                    for r in rows:
                        g = lambda k: r[cm[k]] if k in cm and cm[k] < len(r) else ''
                        traw = clean(g('team')) if '@@TEAM' not in clean(g('team')) else clean(g('team'))
                        key = team_key(traw)
                        tid = present.get((lg, season), {}).get(key)
                        if tid is None and key in allp.get(season, {}): tid = allp[season][key][1]
                        if tid is None: unk.add(traw); continue
                        dv, ev = parse_date(clean(g('date_vacancy')), season, span[(lg, season)])
                        da, ea = parse_date(clean(g('date_appointment')), season, span[(lg, season)]) if 'date_appointment' in cm else (None, False)
                        out_s, in_s = clean(g('outgoing')), clean(g('incoming'))
                        pos = clean(g('position'))
                        recs.append(dict(src=lang, lg=lg, season=season, team_id=CANON.get(tid, tid), team_name=traw.replace('@@TEAM:', '').replace('@@', ''),
                                         date_vacancy=dv, date_appointment=da, outgoing=INTERIM_RE.sub('', out_s).strip(' ()'),
                                         incoming=INTERIM_RE.sub('', in_s).strip(' ()'), manner_raw=clean(g('manner')), manner=manner_class(clean(g('manner'))),
                                         out_interim=int(bool(INTERIM_RE.search(out_s))), interim=int(bool(INTERIM_RE.search(in_s))),
                                         position=pos, pre_season=int(bool(re.search(r'pre|pré', pos, re.I))), vacancy_year_explicit=int(ev)))
                log(f'{lang} {lg} {season}: section "{title}", {len(tabs)} table(s), {len(recs) - n0} rows' + (f'; unmapped teams: {sorted(unk)}' if unk else ''))
    C = pd.DataFrame(recs)
    C['outgoing'] = C.outgoing.str.replace(r'\s*\(\s*\)', '', regex=True).str.strip()
    C['incoming'] = C.incoming.str.replace(r'\s*\(\s*\)', '', regex=True).str.strip()
    # an "outgoing" caretaker leaving is the end of an interim spell, not a new vacancy
    C.loc[(C.out_interim == 1) & (C.manner == ''), 'manner'] = 'interim end'
    C.to_csv(os.path.join(HERE, 'coach_changes_all_sources.csv'), index=False)
    # primary: en; pt fills (lg, season) that en lacks
    have_en = set(map(tuple, C[C.src == 'en'][['lg', 'season']].drop_duplicates().to_numpy()))
    P = C[(C.src == 'en') | ~C.apply(lambda r: (r.lg, r.season) in have_en, axis=1)].copy()
    P = P.drop_duplicates(['lg', 'season', 'team_id', 'date_vacancy', 'outgoing', 'incoming'])
    P = P.sort_values(['date_vacancy', 'team_id'], na_position='last')
    cols = ['season', 'lg', 'team_id', 'team_name', 'date_vacancy', 'date_appointment', 'outgoing', 'incoming', 'manner', 'interim', 'out_interim', 'pre_season', 'position', 'manner_raw', 'src']
    P[cols].to_csv(os.path.join(HERE, 'coach_changes.csv'), index=False)
    log('\nrows per (src, lg, season):'); log(C.groupby(['src', 'lg', 'season']).size().unstack([0, 1]).fillna(0).astype(int).to_string())
    log('\nfinal coach_changes.csv rows per (lg, season):'); log(P.groupby(['season', 'lg']).size().unstack().fillna(0).astype(int).to_string())
    log('missing vacancy dates:', int(P.date_vacancy.isna().sum()))
    open(os.path.join(HERE, 'parse_log.txt'), 'w').write('\n'.join(LOG) + '\n')

if __name__ == '__main__':
    main()

import json, glob, os, sys, collections
RAW = sys.argv[1]
rows = collections.defaultdict(dict)
for f in sorted(glob.glob(os.path.join(RAW, '*.json'))):
    b = os.path.basename(f)[:-5]; slug, y = b.rsplit('_', 1)
    try:
        d = json.load(open(f))
        rows[slug][int(y)] = len(d.get('events', []))
    except Exception as e:
        rows[slug][int(y)] = -1
ys = list(range(2006, 2027))
print('slug'.ljust(28) + ' '.join(f'{y%100:>3d}' for y in ys) + '  total')
for s, r in rows.items():
    print(s.ljust(28) + ' '.join(f'{r.get(y, 0):>3d}' for y in ys) + f'  {sum(v for v in r.values() if v>0)}')

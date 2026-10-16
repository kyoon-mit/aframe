import sys, json
for path in sys.argv[1:]:
    d = json.load(open(path))
    m = d.get('message', {})
    items = m.get('items', [m]) if 'items' in m else [m]
    for it in items:
        t = ' '.join(it.get('title', [''])[:1])
        au = '; '.join((a.get('given', '') + ' ' + a.get('family', '')).strip() for a in it.get('author', []))
        ct = ' '.join(it.get('container-title', [''])[:1])
        yr = (it.get('issued', {}).get('date-parts', [[None]])[0] or [None])[0]
        print('DOI:', it.get('DOI'), '|', yr, '|', ct, '| vol', it.get('volume'), 'iss', it.get('issue'), 'pp', it.get('page'))
        print('   TITLE:', t)
        print('   AUTH:', au)

import sys, json
full = '--full' in sys.argv
for path in [p for p in sys.argv[1:] if not p.startswith('--')]:
    d = json.load(open(path))
    print('# count', d.get('meta', {}).get('count'))
    for w in d.get('results', []):
        src = ((w.get('primary_location') or {}).get('source') or {}).get('display_name')
        au = '; '.join(a['author']['display_name'] for a in w.get('authorships', [])[:6])
        print(f"{w.get('publication_year')} | {w.get('title')} | {src} | {w.get('doi')}")
        if full:
            print('    AUTH:', au)
            inv = w.get('abstract_inverted_index')
            if inv:
                pos = sorted((p, word) for word, ps in inv.items() for p in ps)
                print('    ABS:', ' '.join(wd for _, wd in pos)[:1500])

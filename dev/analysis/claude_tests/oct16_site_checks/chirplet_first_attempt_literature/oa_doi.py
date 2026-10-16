import sys, json
for path in sys.argv[1:]:
    try:
        w = json.load(open(path))
    except Exception as e:
        print(path, 'PARSE FAIL', e); continue
    src = ((w.get('primary_location') or {}).get('source') or {}).get('display_name')
    print(f"{w.get('publication_year')} | {w.get('title')} | {src} | {w.get('doi')}")
    inv = w.get('abstract_inverted_index')
    if inv:
        pos = sorted((p, word) for word, ps in inv.items() for p in ps)
        print('    ABS:', ' '.join(wd for _, wd in pos))
    else:
        print('    ABS: none')

import json, sys
S = sys.argv[1]
G = '/home/kyoon/SSM-BNS/WEB/kyoon-mit.github.io/docs/_data/glossary.yml'
m = json.load(open(f'{S}/merger_position/edits.json'))
g = json.load(open(f'{S}/glossary/edits.json'))
for e in g: e['file'] = G
allE = [('M%d' % i, e) for i, e in enumerate(m)] + [('G%d' % i, e) for i, e in enumerate(g)]
for tag, e in allE:
    txt = open(e['file'], encoding='utf-8').read()
    c = txt.count(e['old_str'])
    print(tag, e['file'].split('/')[-1], 'count', c, '' if c == 1 else '<<<<<<')

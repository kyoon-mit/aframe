import sys, re, glob, os
root = '/home/kyoon/SSM-BNS/WEB/kyoon-mit.github.io/docs'
files = sorted(glob.glob(root + '/projects/bns/*.md') + glob.glob(root + '/_updates/*.md'))
pats = sys.argv[1:]
for pat in pats:
    print('==', pat)
    rx = re.compile(pat, re.I)
    for f in files:
        txt = open(f, encoding='utf-8').read()
        for m in rx.finditer(txt):
            line = txt.count('\n', 0, m.start()) + 1
            s = txt[max(0, m.start()-80):m.end()+60].replace('\n', ' ')
            print(f'  {os.path.relpath(f, root)}:{line}: ...{s}...')

import re,sys,glob
files=sorted(glob.glob('/home/kyoon/SSM-BNS/WEB/kyoon-mit.github.io/docs/projects/bns/*.md')+glob.glob('/home/kyoon/SSM-BNS/WEB/kyoon-mit.github.io/docs/_updates/*.md'))
txt={f:re.sub(r'\s+',' ',open(f).read()) for f in files}
for w in sys.argv[1:]:
    rx=re.compile(r'(^|[^\w-])('+w+r')(?![\w-])', re.I)
    hits=[(f,m) for f in files for m in rx.finditer(txt[f])]
    print('=== %s : %d hits in %d files'%(w,len(hits),len(set(f for f,_ in hits))))
    for f,m in hits[:int(sys.argv[0:1] and 10)]:
        s=txt[f][max(0,m.start()-70):m.end()+50]
        print('  ['+f.split('/')[-1][:30]+']',s)

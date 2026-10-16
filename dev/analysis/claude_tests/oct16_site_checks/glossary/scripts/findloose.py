import sys,re
t=open(sys.argv[1],errors='replace').read()
w=int(sys.argv[2])
for k in sys.argv[3:]:
    rx=''.join((re.escape(c)+r'\s?') if c!=' ' else r'\s+' for c in k)
    hits=list(re.finditer(rx,t,re.I))
    print('## [%s] %d hits'%(k,len(hits)))
    for m in hits[:3]:
        print('   ...', t[max(0,m.start()-w):m.end()+w]); print()

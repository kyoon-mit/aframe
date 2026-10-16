import sys,re
t=re.sub(r'\s+',' ',open(sys.argv[1],errors='replace').read())
w=int(sys.argv[2])
for k in sys.argv[3:]:
    hits=list(re.finditer(k,t,re.I))
    print('## [%s] %d hits'%(k,len(hits)))
    for m in hits[:4]:
        print('   ...', t[max(0,m.start()-w):m.end()+w]); print()

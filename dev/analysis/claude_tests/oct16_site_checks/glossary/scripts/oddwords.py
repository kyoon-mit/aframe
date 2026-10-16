import re,sys,yaml,collections
words=set(w.strip().lower() for w in open('/usr/share/dict/words',errors='ignore'))
g=yaml.safe_load(open(sys.argv[1]))
forms=[]
for e in g:
    fl=([] if e.get('only_also') else [e['term']])+list(e.get('also') or [])
    for f in fl:
        forms.append((e['id'],re.compile(r'(^|[^\w-])('+re.escape(str(f)).replace(r'\ ',r'\s+')+r')(?![\w-])', 0 if e.get('case') else re.I)))
def body(t):
    t=re.sub(r'(?s)^---.*?---','',t,count=1)
    t=re.sub(r'(?s)```.*?```','',t)
    t=re.sub(r'`[^`]*`','',t)
    t=re.sub(r'(?s)\$\$.*?\$\$','',t)
    t=re.sub(r'\$[^$\n]*\$','',t)
    t=re.sub(r'\{%.*?%\}|\{\{.*?\}\}','',t)
    t=re.sub(r'\]\([^)]*\)',']',t)
    t=re.sub(r'<[^>]+>',' ',t)
    t=re.sub(r'(?m)^#+ .*$','',t)
    return t
cnt=collections.Counter()
for f in sys.argv[2:]:
    t=body(open(f).read())
    for m in re.finditer(r"[A-Za-z][A-Za-z'-]*[A-Za-z]",t):
        w=m.group(0)
        parts=w.lower().split('-')
        if all(p in words or p.rstrip('s') in words or p[:-2] in words or p[:-1] in words for p in parts): continue
        cnt[w]+=1
for w,n in sorted(cnt.items(), key=lambda x:-x[1]):
    cov=[i for i,rx in forms if rx.search(' '+w+' ')]
    print(f"{n:4d} {w:26s} {'COVERED '+','.join(sorted(set(cov))) if cov else '--'}")

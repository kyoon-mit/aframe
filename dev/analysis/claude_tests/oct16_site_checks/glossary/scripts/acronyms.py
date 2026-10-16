import re,sys,yaml,collections
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
    for m in re.finditer(r'(?<![\w-])([A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9-]*|[A-Z]{2,})(?![\w])',t):
        cnt[m.group(1)]+=1
for w,n in sorted(cnt.items(), key=lambda x:-x[1]):
    cov=[i for i,rx in forms if rx.search(' '+w+' ')]
    print(f"{n:4d} {w:22s} {'COVERED by '+','.join(sorted(set(cov))) if cov else '-- not covered'}")

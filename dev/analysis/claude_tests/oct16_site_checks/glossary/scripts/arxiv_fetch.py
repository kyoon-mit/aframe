import sys, time, urllib.request, urllib.error, os
out=sys.argv[1]; ids=sys.argv[2:]
t0=time.time()
for i,a in enumerate(ids):
    fn=os.path.join(out, a.replace('/','_')+'.xml')
    if os.path.exists(fn) and os.path.getsize(fn)>500: print(a,'have'); continue
    url="https://export.arxiv.org/api/query?id_list="+a
    data=None
    for wait in (4,8,16,24):
        time.sleep(wait)
        try:
            data=urllib.request.urlopen(url, timeout=30).read()
            if b'Rate exceeded' in data[:100]: data=None; continue
            break
        except urllib.error.HTTPError as e:
            print(a,'HTTP',e.code)
        if time.time()-t0>100: break
    if data: open(fn,'wb').write(data); print(a, len(data))
    if time.time()-t0>100: print('time budget hit'); break

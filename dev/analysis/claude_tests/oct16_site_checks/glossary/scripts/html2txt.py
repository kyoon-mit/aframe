import sys,re,html
t=open(sys.argv[1],errors='replace').read()
t=re.sub(r'(?s)<script.*?</script>|<style.*?</style>','',t)
t=re.sub(r'<[^>]+>',' ',t)
t=html.unescape(t)
open(sys.argv[2],'w').write(re.sub(r'\s+',' ',t))

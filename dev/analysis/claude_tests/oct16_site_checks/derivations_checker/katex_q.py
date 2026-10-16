import json, sys
from py_mini_racer import MiniRacer
ctx = MiniRacer(); ctx.eval("var window=this;var self=this;"); ctx.eval(open(sys.argv[1]).read())
for x in ["h = E[\\phi’‘(n)]", "h = E[\\phi''(n)]"]:
    for strict in ["'warn'", "'ignore'"]:
        r = ctx.eval("(function(){try{var h=katex.renderToString(%s,{throwOnError:true,strict:%s});return h.indexOf('‘')>=0||h.indexOf('’')>=0?'renders curly quotes':'renders primes'}catch(e){return String(e)}})()" % (json.dumps(x), strict))
        print(repr(x), strict, '->', r)

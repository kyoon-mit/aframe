"""Render every math expression of the new text with the site's KaTeX (0.16.11), strict mode on."""
import json, re, sys
from py_mini_racer import MiniRacer
ADV = sys.argv[1]
ctx = MiniRacer()
ctx.eval("var window = this; var self = this;")
ctx.eval(open(ADV + '/katex_dl/katex.min.js').read())
E = json.load(open(ADV + '/final_edits.json'))
exprs = []
for e in E:
    s = e['new_str']
    for m in re.finditer(r'\$\$(.+?)\$\$', s, flags=re.S):
        exprs.append((e['tag'], True, m.group(1)))
    s2 = re.sub(r'\$\$.+?\$\$', '', s, flags=re.S)
    for m in re.finditer(r'(?<![\$\\])\$(?!\$)([^$]+?)(?<!\\)\$(?!\$)', s2):
        exprs.append((e['tag'], False, m.group(1)))
bad = 0
for tag, disp, x in exprs:
    js = "(function(){try{katex.renderToString(%s,{displayMode:%s,throwOnError:true,strict:'error'});return 'ok'}catch(err){return String(err)}})()" % (json.dumps(x), 'true' if disp else 'false')
    r = ctx.eval(js)
    if r != 'ok':
        bad += 1
        print(tag, 'display' if disp else 'inline', repr(x[:80]), '->', r[:200])
print('expressions', len(exprs), 'errors', bad)

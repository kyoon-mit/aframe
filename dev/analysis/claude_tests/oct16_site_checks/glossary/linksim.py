"""Approximate the BNS glossary linker (docs/assets/js/glossary.js) on markdown pages.

Usage: linksim.py GLOSSARY_A GLOSSARY_B  -> prints first-link differences per page.
The page-to-text conversion is approximate: front matter, Liquid tags, code,
math, headings, link texts and skipped HTML elements are removed; figure
captions and labels are kept; straight apostrophes between letters become ’
(kramdown smart quotes) outside include captions.
"""
import glob, os, re, sys
import yaml

ROOT = '/home/kyoon/SSM-BNS/WEB/kyoon-mit.github.io/docs'
W = 'A-Za-z0-9_'

def forms_of(gl):
    forms = []
    for t in gl:
        names = ([] if t.get('only_also') else [t['term']]) + list(t.get('also') or [])
        for form in names:
            form = str(form)
            pat = r'\s+'.join(re.escape(p) for p in form.split(' '))
            flags = 0 if t.get('case') else re.IGNORECASE
            rx = re.compile('(^|[^' + W + '-])(' + pat + ')(?![' + W + '-])', flags)
            forms.append((t['id'], rx))
    return forms

def smart(s):
    return re.sub(r"(?<=[A-Za-z])'(?=[A-Za-z])", '’', s)

def page_nodes(path):
    txt = open(path, encoding='utf-8').read()
    if txt.startswith('---'):
        txt = txt.split('---', 2)[2]
    nodes = []
    # captions and labels of figure includes become their own nodes (raw, no smart quotes)
    def inc(m):
        body = m.group(0)
        for k in ('label', 'caption'):
            mm = re.search(k + r'="([^"]*)"', body)
            if mm:
                nodes_inc.append(mm.group(1))
        return '\n\x00INC\x00\n'
    nodes_inc = []
    txt = re.sub(r'\{%\s*include\s+figure\.html.*?%\}', inc, txt, flags=re.S)
    txt = re.sub(r'\{%.*?%\}', ' ', txt, flags=re.S)
    txt = re.sub(r'\{\{.*?\}\}', ' ', txt, flags=re.S)
    txt = re.sub(r'```.*?```', '\n\x00SEP\x00\n', txt, flags=re.S)
    txt = re.sub(r'<(script|style|button|label|sup|sub|dt|pre|code|a)\b.*?</\1>', '\x00SEP\x00', txt, flags=re.S | re.I)
    txt = re.sub(r'\$\$.*?\$\$', '\x00SEP\x00', txt, flags=re.S)
    txt = re.sub(r'(?<!\\)\$[^$\n]+?(?<!\\)\$', '\x00SEP\x00', txt)
    txt = re.sub(r'`[^`]*`', '\x00SEP\x00', txt)
    txt = re.sub(r'!\[[^\]]*\]\([^)]*\)', '\x00SEP\x00', txt)
    txt = re.sub(r'\[([^\]]*)\]\([^)]*\)', '\x00SEP\x00', txt)
    txt = re.sub(r'(?m)^#{1,6} .*$', '\x00SEP\x00', txt)
    txt = re.sub(r'<[^>]+>', '\x00SEP\x00', txt)
    txt = re.sub(r'\{:[^}]*\}', ' ', txt)
    out = []
    inc_iter = iter(nodes_inc)
    for block in re.split(r'\n\s*\n', txt):
        for piece in re.split(r'\x00(?:SEP|INC)\x00|\*\*|\*|\|', block):
            pass
        # keep order of include captions
        parts = re.split(r'(\x00INC\x00)', block)
        for part in parts:
            if part == '\x00INC\x00':
                for _ in range(2):
                    pass
                continue
            for piece in re.split(r'\x00SEP\x00|\*\*|\*|\|', part):
                if piece.strip():
                    out.append(('md', smart(piece)))
        if '\x00INC\x00' in block:
            # captions belong where the include was; approximate: after the block's text
            n = block.count('\x00INC\x00')
            for _ in range(n):
                pass
    # simpler: interleave captions at the position of their placeholders
    final = []
    cap = list(nodes_inc)
    seq = re.split(r'(\x00INC\x00)', txt)
    ci = 0
    for part in seq:
        if part == '\x00INC\x00':
            # an include holds label then caption, both were appended in order
            # find how many entries this include contributed: label and/or caption
            continue
    return out, nodes_inc

def nodes_in_order(path):
    txt = open(path, encoding='utf-8').read()
    if txt.startswith('---'):
        txt = txt.split('---', 2)[2]
    pieces = []
    pos = 0
    for m in re.finditer(r'\{%\s*include\s+figure\.html.*?%\}', txt, flags=re.S):
        pieces.append(('md', txt[pos:m.start()]))
        body = m.group(0)
        for k in ('label', 'caption'):
            mm = re.search(k + r'="([^"]*)"', body)
            if mm:
                pieces.append(('raw', mm.group(1)))
        pos = m.end()
    pieces.append(('md', txt[pos:]))
    nodes = []
    for kind, s in pieces:
        if kind == 'raw':
            s = re.sub(r'<(a|sup|sub|code)\b.*?</\1>', '\x00', s, flags=re.S)
            s = re.sub(r'<[^>]+>', '\x00', s)
            nodes += [p for p in s.split('\x00') if p.strip()]
            continue
        s = re.sub(r'\{%.*?%\}', ' ', s, flags=re.S)
        s = re.sub(r'\{\{.*?\}\}', ' ', s, flags=re.S)
        s = re.sub(r'```.*?```', '\x00', s, flags=re.S)
        s = re.sub(r'<(script|style|button|label|sup|sub|dt|pre|code|a)\b.*?</\1>', '\x00', s, flags=re.S | re.I)
        s = re.sub(r'\$\$.*?\$\$', '\x00', s, flags=re.S)
        s = re.sub(r'(?<!\\)\$[^$\n]+?(?<!\\)\$', '\x00', s)
        s = re.sub(r'`[^`]*`', '\x00', s)
        s = re.sub(r'!\[[^\]]*\]\([^)]*\)', '\x00', s)
        s = re.sub(r'\[([^\]]*)\]\([^)]*\)', '\x00', s)
        s = re.sub(r'(?m)^#{1,6} .*$', '\x00', s)
        s = re.sub(r'<[^>]+>', '\x00', s)
        s = re.sub(r'\{:[^}]*\}', ' ', s)
        for block in re.split(r'\n\s*\n', s):
            for piece in re.split(r'\x00|\*\*|\*|\|', block):
                if piece.strip():
                    nodes.append(smart(piece))
    return nodes

def link(nodes, forms):
    done = {}
    links = []
    for text in nodes:
        pos = 0
        while pos < len(text):
            best = None
            rest = text[pos:]
            for fid, rx in forms:
                m = rx.search(rest)
                if not m:
                    continue
                start = pos + m.start() + len(m.group(1))
                ln = len(m.group(2))
                if best is None or start < best[1] or (start == best[1] and ln > best[2]):
                    best = (fid, start, ln)
            if best is None:
                break
            fid, start, ln = best
            if fid in done:
                pos = start + ln
                continue
            done[fid] = True
            ctx = text[max(0, start - 50):start + ln + 40].replace('\n', ' ')
            links.append((fid, text[start:start + ln], ctx))
            text = text[start + ln:]
            pos = 0
    return links

def pages():
    return sorted(glob.glob(ROOT + '/projects/bns/*.md') + glob.glob(ROOT + '/_updates/*.md'))

if __name__ == '__main__':
    ga = yaml.safe_load(open(sys.argv[1], encoding='utf-8'))
    gb = yaml.safe_load(open(sys.argv[2], encoding='utf-8'))
    fa, fb = forms_of(ga), forms_of(gb)
    only = sys.argv[3:] if len(sys.argv) > 3 else None
    for p in pages():
        rel = os.path.relpath(p, ROOT)
        if rel.startswith('projects/bns/definitions'):
            continue
        if only and not any(o in rel for o in only):
            continue
        nodes = nodes_in_order(p)
        la = {x[0]: x for x in link(nodes, fa)}
        lb = {x[0]: x for x in link(nodes, fb)}
        diffs = []
        for k in sorted(set(la) | set(lb)):
            a, b = la.get(k), lb.get(k)
            if a is None or b is None or a[2] != b[2]:
                diffs.append((k, a, b))
        if diffs:
            print('##', rel)
            for k, a, b in diffs:
                print('  ', k)
                print('     before:', None if a is None else f'[{a[1]}] ...{a[2]}...')
                print('     after: ', None if b is None else f'[{b[1]}] ...{b[2]}...')

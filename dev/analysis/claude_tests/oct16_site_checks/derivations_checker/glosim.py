"""Simulate docs/assets/js/glossary.js on a kramdown-rendered page body."""
import re
import sys
import html
from html.parser import HTMLParser
import yaml

SKIP = {'a', 'code', 'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'dt', 'script', 'style', 'sup', 'sub', 'button', 'label'}


class P(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.nodes = []

    def handle_starttag(self, tag, attrs):
        if tag in ('br', 'img', 'hr', 'input', 'meta', 'link'):
            return
        self.stack.append(tag)

    def handle_endtag(self, tag):
        while self.stack:
            t = self.stack.pop()
            if t == tag:
                break

    def handle_data(self, data):
        self.nodes.append((data, any(t in SKIP for t in self.stack)))


def strip_math(t):
    t = re.sub(r'\\\[.*?\\\]', ' \x00 ', t, flags=re.S)
    t = re.sub(r'\$\$.*?\$\$', ' \x00 ', t, flags=re.S)
    t = re.sub(r'\$[^$]+?\$', ' \x00 ', t, flags=re.S)
    return t


def run(html_path, glossary_path, extra=None):
    G = yaml.safe_load(open(glossary_path))
    if extra:
        G += yaml.safe_load(open(extra))
    forms = []
    for t in G:
        fs = ([] if t.get('only_also') else [t['term']]) + list(t.get('also') or [])
        for f in fs:
            rx = re.compile(r'(^|[^\w-])(' + re.escape(str(f)).replace(r'\ ', r'\s+') + r')(?![\w-])',
                            (0 if t.get('case') else re.I) | re.A)
            forms.append((t['id'], rx))
    p = P()
    p.feed(open(html_path).read())
    done = {}
    for text, skipped in p.nodes:
        if skipped:
            continue
        text = strip_math(text)
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
            ctx = text[max(0, start - 60):start + ln + 60].replace('\n', ' ')
            done[fid] = ctx
            pos = start + ln
    return done


if __name__ == '__main__':
    d = run(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    for k in sys.argv[4:]:
        print(k, '=>', d.get(k))

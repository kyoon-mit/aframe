"""Rough simulation of docs/assets/js/glossary.js on kramdown HTML: first appearance of each term.
Candidates from one finditer pass per form; then a sweep: earliest start, longest at the same start,
spans of already linked terms passed over."""
import html, re, sys, yaml
glo_path, html_path = sys.argv[1], sys.argv[2]
marks = sys.argv[3:]
G = yaml.safe_load(open(glo_path))
h = open(html_path).read()
h = re.sub(r"\$\$.*?\$\$", " ", h, flags=re.S)
h = re.sub(r"\$[^$<]*?\$", " ", h)
h = re.sub(r"<(a|code|pre|h[1-6]|sup|sub|script|style|button|label|dt)\b[^>]*>.*?</\1>", " ", h, flags=re.S)
text = html.unescape(re.sub(r"<[^>]+>", " ", h))
cands = []
for t in G:
    names = ([] if t.get("only_also") else [t["term"]]) + list(t.get("also") or [])
    for f in names:
        f = str(f)
        rx = re.compile(r"(?<![\w-])(" + re.escape(f).replace(r"\ ", r"\s+") + r")(?![\w-])", 0 if t.get("case") else re.I)
        for m in rx.finditer(text):
            cands.append((m.start(1), -(m.end(1) - m.start(1)), t["id"], m.group(1)))
cands.sort()
done, links, pos = set(), [], 0
for st, negln, tid, s in cands:
    if st < pos:
        continue
    if tid not in done:
        done.add(tid); links.append((tid, st, s))
    pos = st - negln
for mk in marks:
    k = text.find(mk)
    print(f"--- linked at or after {mk!r} (pos {k}):")
    nxt = [text.find(m2) for m2 in marks if text.find(m2) > k]
    end = min(nxt) if nxt else len(text)
    for tid, st, s in links:
        if k >= 0 and k <= st < end:
            print(f"   {tid:26s} {s!r:28s} ctx: {text[max(0, st-50):st+40]!r}".replace("\\n", " "))
print("total linked ids:", len(links))

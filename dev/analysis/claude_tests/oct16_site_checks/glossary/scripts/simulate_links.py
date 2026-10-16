import re, sys, yaml

gloss = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
watch = set(sys.argv[2].split(","))
pages = sys.argv[3:]

forms = []
for t in gloss:
    for f in ([] if t.get("only_also") else [t["term"]]) + list(t.get("also") or []):
        rx = re.compile(r"(^|[^\w-])(" + re.escape(f).replace(r"\ ", r"\s+") + r")(?![\w-])",
                        0 if t.get("case") else re.I)
        forms.append((t["id"], rx))


def body(s):
    s = re.sub(r"(?s)^---.*?---", "", s, count=1)
    s = re.sub(r"(?s)```.*?```", " ", s)
    s = re.sub(r"`[^`]*`", " ", s)
    s = re.sub(r"(?s)\$\$.*?\$\$", " ", s)
    s = re.sub(r"\$[^$\n]*\$", " ", s)
    s = re.sub(r"(?s)\{%.*?%\}|\{\{.*?\}\}", " ", s)
    s = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", s)          # link text is skipped (A)
    s = re.sub(r"(?s)<a\b.*?</a>", " ", s)
    s = re.sub(r"(?s)<(script|style|button|label)\b.*?</\1>", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"(?m)^#+ .*$", " ", s)                   # headings are skipped
    return re.sub(r"\s+", " ", s)


for p in pages:
    text = body(open(p, encoding="utf-8").read())
    cands = []
    for i, rx in forms:
        for m in rx.finditer(text):
            cands.append((m.start(2), -len(m.group(2)), i))
    cands.sort()
    done, pos, links = set(), 0, []
    for start, negln, i in cands:
        ln = -negln
        if start < pos:
            continue
        pos = start + ln
        if i not in done:
            done.add(i)
            links.append((i, text[start:start + ln], text[max(0, start - 50):start + ln + 30]))
    print("==", p.split("/")[-1])
    for i, s, c in links:
        if i in watch:
            print(f"   {i:20s} '{s}'  ... {c}")

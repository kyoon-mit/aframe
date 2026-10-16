import json, re, sys
E = json.load(open(sys.argv[1]))
s = E[1]["new_str"]
t = re.sub(r"\$\$(.*?)\$\$", "", s, flags=re.S)
bad = []
for m in re.findall(r"\$(.+?)\$", t, re.S):
    for k in re.finditer(r"(?<![A-Za-z0-9\\])_", m):
        bad.append(m)
print("inline math with '_' after a non-word char:", bad)
print("asterisks inside inline math:", [m for m in re.findall(r"\$(.+?)\$", t, re.S) if "*" in m])
# any line in a paragraph starting with '|' or containing ' | ' outside tables
for i, l in enumerate(s.split("\n")):
    if "|" in l and not l.startswith("|") and "relative_url" not in l:
        print("pipe in text line", i + 1, l[:100])

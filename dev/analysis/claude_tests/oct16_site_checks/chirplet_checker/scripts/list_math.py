import json, re, sys
E = json.load(open(sys.argv[1]))
s = E[1]["new_str"]
for m in re.findall(r"\$\$(.*?)\$\$", s, re.S):
    print("D:", m.strip())
t = re.sub(r"\$\$(.*?)\$\$", "", s, flags=re.S)
seen = []
for m in re.findall(r"\$(.+?)\$", t, re.S):
    if m not in seen:
        seen.append(m)
        print("I:", m)

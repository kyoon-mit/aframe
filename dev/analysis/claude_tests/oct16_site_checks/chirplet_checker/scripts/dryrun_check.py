"""Apply the corrected edits to fresh copies of the site files (never the repo) and check them."""
import json, os, re, shutil, sys
import yaml

SP = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/oct16fix/chirplet_adv"
DOCS = "/home/kyoon/SSM-BNS/WEB/kyoon-mit.github.io/docs"
E = json.load(open(sys.argv[1]))
os.makedirs(SP + "/dryrun", exist_ok=True)
txt = {}
for e in E:
    src = open(e["file"], encoding="utf-8").read()
    n = src.count(e["old_str"])
    print("old_str count in", e["file"].split("docs/")[1], "=", n)
    assert n == 1
    assert e["new_str"].count(e["old_str"]) == 1, "insertion must keep old_str"
    new = src.replace(e["old_str"], e["new_str"])
    txt[e["file"]] = new
    open(SP + "/dryrun/" + os.path.basename(e["file"]), "w", encoding="utf-8").write(new)

der = txt[DOCS + "/projects/bns/derivations.md"]
bra = txt[DOCS + "/projects/bns/brainstorm.md"]
glo = txt[DOCS + "/_data/glossary.yml"]

# anchors
ids = re.findall(r"\{#([A-Za-z0-9_-]+)\}", der)
dup = {i for i in ids if ids.count(i) > 1}
print("derivations anchors:", len(ids), "duplicates:", dup)
idset = set(ids)
new_sec = E[1]["new_str"]
links = re.findall(r"\]\(#([A-Za-z0-9_-]+)\)", new_sec)
print("internal links in new section:", sorted(set(links)), "missing:", [l for l in set(links) if l not in idset])
blinks = re.findall(r"derivations/' \| relative_url \}\}#([A-Za-z0-9_-]+)", E[0]["new_str"])
print("brainstorm derivation anchors:", blinks, "missing:", [l for l in blinks if l not in idset])
# update links
ups = re.findall(r"append: '([0-9a-z-]+)/'", E[0]["new_str"] + E[1]["new_str"])
print("update links:", ups, "missing files:", [u for u in ups if not os.path.exists(f"{DOCS}/_updates/{u}.md")])
# glossary
G = yaml.safe_load(glo)
gid = [t["id"] for t in G]
print("glossary entries", len(G), "dup ids", {i for i in gid if gid.count(i) > 1})
forms = {}
for t in G:
    for f in ([t["term"]] if not t.get("only_also") else []) + (t.get("also") or []):
        forms.setdefault(f.lower(), []).append(t["id"])
print("dup forms:", {k: v for k, v in forms.items() if len(v) > 1})
for t in G:
    if t["id"] in ("chirplet", "linear-chirp"):
        print(t["id"], "|", t["term"], "|", t.get("also"), "|", t["group"])
        print("  short:", t["short"])
        print("  source:", t["source"])
glinks = re.findall(r"derivations/#([A-Za-z0-9_-]+)", E[2]["new_str"])
print("glossary derivation anchors:", glinks, "missing:", [l for l in glinks if l not in idset])
# style
for e in E:
    s = e["new_str"]
    bad = [c for c in ("—", "–") if c in s]
    words = re.findall(r"\b(about|roughly|approximately|around)\b", s, re.I)
    print(os.path.basename(e["file"]), "dashes:", bad, "hedges:", words)
# headings blank line
lines = der.split("\n")
for i, l in enumerate(lines):
    if l.startswith("#") and i > 0 and lines[i - 1].strip() != "":
        print("heading without blank line before, line", i + 1, l[:70])
# math balance in the new section
body = new_sec
disp = re.findall(r"\$\$(.*?)\$\$", body, re.S)
inl_src = re.sub(r"\$\$(.*?)\$\$", "", body, flags=re.S)
for para in inl_src.split("\n\n"):
    if para.count("$") % 2:
        print("ODD $ in paragraph:", para[:120])
inl = re.findall(r"\$(.+?)\$", inl_src, re.S)
def bal(m):
    depth = 0
    for ch in m.replace("\\{", "").replace("\\}", ""):
        depth += ch == "{"
        depth -= ch == "}"
        if depth < 0:
            return False
    return depth == 0
print("math segments:", len(disp), "display,", len(inl), "inline; unbalanced braces:", [m for m in disp + inl if not bal(m)])
lr = [m for m in disp + inl if m.count("\\left") != m.count("\\right")]
print("left/right mismatch:", lr)
big = [m for m in disp + inl if (m.count("\\big(") + m.count("\\Big(")) != (m.count("\\big)") + m.count("\\Big)"))]
print("big paren mismatch:", big)
# html balance in brainstorm item
li = E[0]["new_str"].split("\n")[0]
for tag in ("li", "span", "b", "a", "i"):
    o = len(re.findall(rf"<{tag}[ >]", li)); c = li.count(f"</{tag}>")
    if o != c:
        print("unbalanced", tag, o, c)
print("brainstorm li ok; text length", len(li))

"""Replace Liquid tags so kramdown can render the page body as Jekyll would after Liquid."""
import re, sys
src, dst = sys.argv[1], sys.argv[2]
t = open(src).read()
if t.startswith("---"):
    t = t.split("---", 2)[2]
t = t.replace("{{ d | relative_url }}", "/projects/bns/derivations/")
t = re.sub(r"\{\{ u \| append: '([^']*)' \| relative_url \}\}", r"/projects/bns/updates/\1", t)
t = re.sub(r"\{\{ '([^']*)' \| relative_url \}\}", r"\1", t)
t = re.sub(r"\{%-?.*?-?%\}", "", t, flags=re.S)
left = re.findall(r"\{\{.*?\}\}|\{%.*?%\}", t)
if left:
    print("Liquid left:", left[:5])
open(dst, "w").write(t)

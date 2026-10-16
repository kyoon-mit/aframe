import sys, xml.etree.ElementTree as ET
ns = {'a': 'http://www.w3.org/2005/Atom', 'x': 'http://arxiv.org/schemas/atom', 'o': 'http://a9.com/-/spec/opensearch/1.1/'}
full = '--full' in sys.argv
for path in [p for p in sys.argv[1:] if not p.startswith('--')]:
    root = ET.parse(path).getroot()
    tot = root.find('o:totalResults', ns)
    if tot is not None:
        print('# totalResults', tot.text)
    for e in root.findall('a:entry', ns):
        def g(tag, n=ns):
            el = e.find(tag, n)
            return el.text.strip().replace('\n', ' ') if el is not None and el.text else ''
        idv = g('a:id'); title = ' '.join(g('a:title').split())
        auth = [a.find('a:name', ns).text for a in e.findall('a:author', ns)]
        print('ID:', idv)
        print('  TITLE:', title)
        print('  AUTH:', '; '.join(auth))
        print('  PUB:', g('a:published')[:10], ' JREF:', g('x:journal_ref'), ' DOI:', g('x:doi'))
        if full:
            print('  ABS:', ' '.join(g('a:summary').split()))

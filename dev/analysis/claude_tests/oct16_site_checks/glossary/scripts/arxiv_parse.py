import sys, glob, xml.etree.ElementTree as ET
ns={'a':'http://www.w3.org/2005/Atom','x':'http://arxiv.org/schemas/atom'}
for f in sys.argv[1:]:
    r=ET.parse(f).getroot()
    for e in r.findall('a:entry',ns):
        print('ID:', e.find('a:id',ns).text)
        print('TITLE:', ' '.join(e.find('a:title',ns).text.split()))
        print('AUTHORS:', ', '.join(a.find('a:name',ns).text for a in e.findall('a:author',ns)))
        print('PUBLISHED:', e.find('a:published',ns).text)
        jr=e.find('x:journal_ref',ns); print('JOURNAL_REF:', jr.text if jr is not None else None)
        dd=e.find('x:doi',ns); print('DOI:', dd.text if dd is not None else None)
        cm=e.find('x:comment',ns); print('COMMENT:', ' '.join(cm.text.split()) if cm is not None else None)
        print('ABSTRACT:', ' '.join(e.find('a:summary',ns).text.split())[:1400])
        print()

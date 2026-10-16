import sys, pickle, types, torch
class Stub:
    def __init__(self,*a,**k): pass
    def __setstate__(self, s): pass
class U(pickle.Unpickler):
    def find_class(self, mod, name):
        try: return super().find_class(mod, name)
        except Exception: return Stub
pm = types.ModuleType('pm'); pm.Unpickler = U; pm.load = pickle.load
for c in sys.argv[1:]:
    ck = torch.load(c, map_location='cpu', weights_only=False, pickle_module=pm)
    sd = ck['state_dict']
    n = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
    print(c.split('/')[0], 'numel', n, 'epoch', ck.get('epoch'))

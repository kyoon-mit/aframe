import h5py
for p in ['/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5',
          '/home/kyoon/SSM-BNS/DATA/aframe/val/end_o3_ratesandpops_bns_snr4.hdf5']:
    with h5py.File(p, 'r') as f:
        print(p)
        print('  attrs:', {k: (f.attrs[k] if f.attrs[k].size < 5 else '...') for k in f.attrs})
        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print('   ', name, obj.shape, obj.dtype)
        f.visititems(show)

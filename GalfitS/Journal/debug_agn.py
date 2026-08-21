import sys,os,numpy as np
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
sys.path.insert(0,'GalfitS/src')
import jax.numpy as jnp
import galfits.gsutils as gsutils
from astropy.table import Table

_o=jnp.array
def _s(o,*a,**k):
    if isinstance(o,np.ndarray) and o.dtype.byteorder in ('>','B'):
        o=o.byteswap().view(o.dtype.newbyteorder('='))
    return _o(o,*a,**k)
jnp.array=_s

wd='MIRI_NIRCam_result/fixed_center_C1host_C2AGN/noSED'
c=[f for f in os.listdir(wd) if f.endswith('.lyric')][0]
g=[f for f in os.listdir(wd) if f.endswith('.gssummary')][0]
fitter,_,_=gsutils.read_config_file(os.path.join(wd,c),wd)
sm=Table.read(os.path.join(wd,g),format='ascii')
for r in sm:
    if r['pname'] in fitter.lmParameters:
        fitter.lmParameters[r['pname']].set(value=r['best_value'],min=r['best_value']-0.1,max=r['best_value']+0.1)
fitter.loose_fix_pars()

# Check AGN per-band logL
for k in sorted(fitter.pardict.keys()):
    if 'AGN' in k and 'logL' in k: print(k, fitter.pardict[k])
    if k.startswith('Ni_AGN'): print(k, fitter.pardict[k])

# Check model images
fitter.cal_model_image()
for i in range(3):
    im=fitter.GSdata.get_image(i)
    mm=np.array(im.model_image)
    print(f'{im.band}: max={np.max(mm):.6f} sum={np.sum(mm):.6f}')

# Set Ni to 0 and re-check
for k in list(fitter.pardict.keys()):
    if k.startswith('Ni_AGN'): fitter.pardict[k]=0.0
fitter.cal_model_image()
for i in range(3):
    im=fitter.GSdata.get_image(i)
    mm=np.array(im.model_image)
    print(f'{im.band} (Ni=0): max={np.max(mm):.6f} sum={np.sum(mm):.6f}')

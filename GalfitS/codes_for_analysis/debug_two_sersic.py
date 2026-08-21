import sys, os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
sys.path.insert(0,'GalfitS/src')
import galfits.gsutils as gsutils
from astropy.table import Table

fitter, targ, fs = gsutils.read_config_file(
    'MIRI_NIRCam_result/two_sersic_C1C2/noSED/C1C2_noSED.lyric',
    'MIRI_NIRCam_result/two_sersic_C1C2/noSED')

ptab = Table.read('MIRI_NIRCam_result/two_sersic_C1C2/noSED/C1C2_noSED.params', format='ascii')
for r in ptab: fitter.pardict[r['name']] = r['value']

gal = [m for i,m in enumerate(fitter.model_list) if fitter.mtype_list[i]=='galaxy'][0]
print('subnames:', gal.subnames)
print('subCs keys:', list(gal.subCs.keys()))

for key in sorted(fitter.pardict.keys()):
    if 'logNorm' in key and 'miri_f770w' in key:
        print(key, fitter.pardict[key])
    if 'logNorm' in key and 'miri_f2100w' in key:
        print(key, fitter.pardict[key])

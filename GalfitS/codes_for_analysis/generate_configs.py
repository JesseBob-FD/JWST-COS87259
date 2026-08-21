"""Generate all 8 .lyric configs for MIRI+NIRCam joint fitting.

4 setups: single_sersic, host_agn, bulge_disk, sersic_freesky
2 modes each: noSED (Ia15=0), SED (Ia15=1)
Output: MIRI_NIRCam_result/run00{1-8}/
"""
import os

OUT = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_NIRCam_result'
CUT_M = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_cutout_v3'
CUT_N = '/mnt/d/Fudan_University/Research/JWST/GalfitS/NIRCam_cutout'

RA, DEC, Z = 149.74276239, 1.6555373, 6.83
EBV = 0.0217

# (band, letter, is_miri, ia9)
BANDS = [
    ('F560W',  'a', True,  '1.251856e+21'),
    ('F770W',  'b', True,  '2.313332e+21'),
    ('F1000W', 'c', True,  '3.912241e+21'),
    ('F1130W', 'd', True,  '5.041238e+21'),
    ('F1280W', 'e', True,  '6.490452e+21'),
    ('F1500W', 'f', True,  '8.959912e+21'),
    ('F1800W', 'g', True,  '1.276140e+22'),
    ('F2100W', 'h', True,  '1.712489e+22'),
    ('F2550W', 'i', True,  '2.536078e+22'),
    ('F115W',  'j', False, '2.100850e+20'),
    ('F200W',  'k', False, '6.235954e+20'),
    ('F410M',  'l', False, '2.627748e+21'),
]

MIRI_LETTERS = [b[1] for b in BANDS if b[2]]
NIRCAM_LETTERS = [b[1] for b in BANDS if not b[2]]


def img_block(band, letter, is_miri, ia9, ia15, free_sky=False):
    """15-line image definition block."""
    cdir = CUT_M if is_miri else CUT_N
    prefix = 'miri' if is_miri else 'nircam'
    img = f'{cdir}/images/cos87259_{band}_cut.fits'
    psf = f'{cdir}/psf/cos87259_{band}_psf.fits'
    sky_vary = 1 if free_sky else 0
    return [
        f"I{letter}1) [{img},0]",
        f"I{letter}2) {prefix}_{band.lower()}",
        f"I{letter}3) [{img},2]",
        f"I{letter}4) [{psf},0]",
        f"I{letter}5) 1",
        f"I{letter}6) [{img},1]",
        f"I{letter}7) MJy/sr",
        f"I{letter}8) 2.0",
        f"I{letter}9) {ia9}",
        f"I{letter}10) 0",
        f"I{letter}11) uniform",
        f"I{letter}12) [[0,-0.5,0.5,0.1,{sky_vary}]]",
        f"I{letter}13) 0",
        f"I{letter}14) [[0,-2,2,0.1,0],[0,-2,2,0.1,0]]",
        f"I{letter}15) {ia15}",
    ]


def profile_sersic(letter, name, sed_free):
    """Sersic profile block. sed_free: SED params free (1) or fixed (0)."""
    v = 1 if sed_free else 0
    # P_14 (logM) is always free in SED mode per user instruction
    v14 = 1 if sed_free else 0
    return [
        f"P{letter}1) {name}",
        f"P{letter}2) sersic",
        f"P{letter}3) [0, -1.5, 1.5, 0.01, 1]",
        f"P{letter}4) [0, -1.5, 1.5, 0.01, 1]",
        f"P{letter}5) [0.15, 0.02, 1.5, 0.01, 1]",
        f"P{letter}6) [2.0, 0.3, 8.0, 0.1, 1]",
        f"P{letter}7) [0, -90, 90, 1, 1]",
        f"P{letter}8) [0.6, 0.1, 1.0, 0.01, 1]",
        f"P{letter}9)  [[-2,-8,0,0.1,{v}]]",
        f"P{letter}10) [[0.5,0.01,10,0.01,{v}]]",
        f"P{letter}11) [[0.02,0.0001,0.04,0.001,{v}]]",
        f"P{letter}12) [[0.3,0,5.1,0.1,{v}]]",
        f"P{letter}13) [100,40,200,1,0]",
        f"P{letter}14) [10.0,6.5,12,0.1,{v14}]",
        f"P{letter}15) burst",
        f"P{letter}16) [-2,-4,-2,0.1,0]",
        f"P{letter}26) [3,0,5,0.1,0]",
        f"P{letter}27) 0",
    ]


def agn_block(sed_free):
    """AGN block. sed_free: AGN params free."""
    v = 1 if sed_free else 0
    return [
        "Na1) AGN",
        "Na2) [6.83, 5.0, 8.0, 0.01, 0]",
        f"Na3) {EBV}",
        f"Na4) [0, -1.5, 1.5, 0.01, {v}]",
        f"Na5) [0, -1.5, 1.5, 0.01, {v}]",
        "Na6) [7,5,10,0.1,0]",
        "Na7) [-1,-4,2,0.1,0]",
        "Na8) [0,0,0.99,0.01,0]",
        f"Na9) [0,0,3.1,0.1,{v}]",
        f"Na10) [43,41,47,0.1,{v}]",
        f"Na11) [[1,0,4,0.1,{v}], [0.6, 0, 5, 0.1,{v}]]",
        "Na12) []",
        "Na13) []",
        "Na14) 1",
        "Na15) 1",
        "Na16) 0",
        "Na17) 0",
        "Na18) 0",
        "Na19) [1.,0.5,2,0.05,0]",
        "Na20) 0",
        "Na21) [41,39,44,0.1,0]",
        "Na22) [-0.5,-2.5,-0.25,0.05,0]",
        "Na23) [0.5,0.25,1.5,0.05,0]",
        "Na24) [7,5,10,0.5,0]",
        "Na25) [15,0,90,5,0]",
        f"Na26) [1.,0.2,5,0.1,{v}]",
                f"Na27) [[40,38,42,0.1,{v}], [40,38,42,0.1,0]]",
    ]


def galaxy_block(letter, profiles, has_agn):
    """Galaxy block."""
    plist = str([f"'{p}'" for p in profiles]).replace('"', '')
    return [
        f"G{letter}1) host_galaxy",
        f"G{letter}2) {plist}",
        f"G{letter}3) [6.83, 5.0, 8.0, 0.01, 0]",
        f"G{letter}4) {EBV}",
        f"G{letter}5) [1.,0.5,2,0.05,0]",
        f"G{letter}6) []",
        f"G{letter}7) 1",
    ]


def atlas_block(letter, name, image_letters):
    """Atlas definition block."""
    ilist = str([f"'{l}'" for l in image_letters]).replace('"', '')
    return [
        f"A{letter}1) {name}",
        f"A{letter}2) {ilist}",
        f"A{letter}3) 1",
        f"A{letter}4) 0",
        f"A{letter}5) []",
        f"A{letter}6) []",
        f"A{letter}7) []",
    ]


def build_config(run_id, tag, sed_mode, free_sky, has_agn, two_comp):
    """Build and write a complete .lyric config."""
    ia15 = 1 if sed_mode else 0
    lines = [
        f"# {tag}",
        f"R1) {tag}",
        f"R2) [{RA},{DEC}]",
        f"R3) {Z}",
        "",
    ]
    # Image blocks
    for band, letter, is_miri, ia9 in BANDS:
        lines += img_block(band, letter, is_miri, ia9, ia15, free_sky)
        lines += [""]
    # Atlas blocks
    lines += atlas_block('a', 'jwst_miri', MIRI_LETTERS)
    lines += [""]
    lines += atlas_block('b', 'jwst_nircam', NIRCAM_LETTERS)
    lines += [""]
    # Profile(s)
    if two_comp:
        lines += profile_sersic('a', 'bulge', sed_mode)
        lines += [""]
        lines += profile_sersic('b', 'disk', sed_mode)
        lines += [""]
    else:
        lines += profile_sersic('a', 'host', sed_mode)
        lines += [""]
    # AGN
    if has_agn:
        lines += agn_block(sed_mode)
        lines += [""]
    # Galaxy
    if two_comp:
        lines += galaxy_block('a', ['a', 'b'], has_agn)
    else:
        lines += galaxy_block('a', ['a'], has_agn)

    # Write
    rundir = os.path.join(OUT, run_id)
    os.makedirs(rundir, exist_ok=True)
    fpath = os.path.join(rundir, f'{tag}.lyric')
    with open(fpath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return fpath


# ====== Generate all 8 configs ======
os.makedirs(OUT, exist_ok=True)
configs = [
    # (run_id, tag, sed_mode, free_sky, has_agn, two_comp)
    ('run001', 'COS87259_R001_noSED', False, False, False, False),
    ('run002', 'COS87259_R001_SED',  True,  False, False, False),
    ('run003', 'COS87259_R002_noSED', False, False, True,  False),
    ('run004', 'COS87259_R002_SED',  True,  False, True,  False),
    ('run005', 'COS87259_R003_noSED', False, False, False, True),
    ('run006', 'COS87259_R003_SED',  True,  False, False, True),
    ('run007', 'COS87259_R004_noSED', False, True,  False, False),
    ('run008', 'COS87259_R004_SED',  True,  True,  False, False),
]

for run_id, tag, sed, fsky, agn, two in configs:
    fpath = build_config(run_id, tag, sed, fsky, agn, two)
    print(f"  {fpath}")

print(f"\nDone. {len(configs)} configs written to {OUT}/")

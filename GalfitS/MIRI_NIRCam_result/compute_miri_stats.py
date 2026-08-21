"""
Extract MIRI-only chi², reduced chi², BIC from all noSED gssummary files.
Solves for k_fit from full BIC formula, then computes MIRI-only BIC.
"""
import os, re, numpy as np

BASE = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_NIRCam_result'

# All noSED models with their gssummary paths
# (free-center variants live in folders renamed on 2026-08-17 without the C1C2 suffix)
MODELS = [
    ('R001 single Sersic', 'single_sersic/noSED'),
    ('R003 Host+AGN PL', 'host_agn_powerlaw/noSED'),
    ('R005 Bulge+Disk', 'bulge_disk/noSED'),
    ('R007 Free sky', 'free_sky_single_sersic/noSED'),
    ('R010 Host+AGN NP', 'host_freeAGN/noSED'),
    ('R011 Dual AGN no host (free)', 'dual_AGN/noSED'),
    ('R012 Host+dual AGN (free)', 'host_dualAGN/noSED'),
    ('C1host_C2AGN NP', 'fixed_center_C1host_C2AGN/noSED'),
    ('C2host_C1AGN NP', 'fixed_center_C2host_C1AGN/noSED'),
    ('C1C2 two-Sersic', 'two_sersic_C1C2/noSED'),
    ('R013 2Sersic+2AGN (free AGN)', 'two_sersic_two_AGN/noSED'),
    ('Dual AGN fixed centers', 'dual_AGN_C1C2/noSED'),
    ('Host+dual AGN fixed centers', 'host_dualAGN_C1C2/noSED'),
    ('2Sersic+2AGN fixed centers', 'two_sersic_two_AGN_C1C2/noSED'),
]

results = []

for name, subdir in MODELS:
    gspath = os.path.join(BASE, subdir)
    gsfiles = [f for f in os.listdir(gspath) if f.endswith('.gssummary')]
    if not gsfiles:
        print(f"SKIP {name}: no gssummary in {subdir}")
        continue
    gsfile = os.path.join(gspath, gsfiles[0])

    with open(gsfile) as f:
        lines = f.readlines()

    # Parse header
    total_chisq = None
    total_BIC = None
    for line in lines:
        line = line.strip()
        if line.startswith('# chisq:'):
            total_chisq = float(line.split(':')[1].strip())
        if line.startswith('# BIC:'):
            total_BIC = float(line.split(':')[1].strip())

    # Parse per-band data
    miri_chisq = 0.0
    miri_dof = 0.0
    nircam_chisq = 0.0
    nircam_dof = 0.0
    in_miri = False
    in_nircam = False
    for line in lines:
        if 'image atals: jwst_miri' in line:
            in_miri = True; in_nircam = False; continue
        if 'image atals: jwst_nircam' in line:
            in_miri = False; in_nircam = True; continue
        if 'image atals:' in line and 'jwst_nircam' not in line and 'jwst_miri' not in line:
            in_miri = False; in_nircam = False; continue
        if not (in_miri or in_nircam):
            continue
        m = re.search(r'chisq:\s*\[([0-9.e+\-]+)\]', line)
        if m:
            chisq_val = float(m.group(1))
            m2 = re.search(r'dof:\s*\[([0-9.e+\-]+)\]', line)
            dof_val = float(m2.group(1)) if m2 else 0
            if in_miri:
                miri_chisq += chisq_val
                miri_dof += dof_val
            else:
                nircam_chisq += chisq_val
                nircam_dof += dof_val

    full_chisq = miri_chisq + nircam_chisq
    full_dof = miri_dof + nircam_dof

    # Solve for k_fit: BIC = chi² + k * ln(dof + k)
    # Use Newton iteration or simple approximation
    # k * ln(dof_full + k) = BIC - chi²
    target = total_BIC - total_chisq
    if target <= 0:
        k_fit = 0
    else:
        # Newton: f(k) = k*ln(dof+k) - target, f'(k) = ln(dof+k) + k/(dof+k)
        k = target / np.log(full_dof)  # initial guess
        for _ in range(20):
            N = full_dof + k
            fk = k * np.log(N) - target
            fpk = np.log(N) + k / N
            dk = fk / fpk if abs(fpk) > 1e-15 else 0
            k -= dk
            if abs(dk) < 1e-6:
                break
        k_fit = k

    # MIRI-only stats
    N_miri = miri_dof + k_fit
    BIC_miri = miri_chisq + k_fit * np.log(N_miri) if N_miri > 0 else miri_chisq
    red_chisq_miri = miri_chisq / miri_dof if miri_dof > 0 else 0
    red_chisq_full = full_chisq / full_dof if full_dof > 0 else 0

    results.append((name, miri_chisq, miri_dof, red_chisq_miri, BIC_miri,
                    full_chisq, full_dof, red_chisq_full, total_BIC, k_fit))

# Sort by MIRI BIC
results.sort(key=lambda x: x[4])

print("=" * 120)
print("MIRI-ONLY RANKING (noSED models)")
print("=" * 120)
print(f"{'Rank':<5} {'Model':<25} {'χ²_MIRI':<14} {'dof_MIRI':<10} {'red_χ²_MIRI':<12} {'BIC_MIRI':<14} {'χ²_full':<14} {'BIC_full':<14} {'k_free':<8}")
print("-" * 120)
for rank, (name, c_m, d_m, r_m, b_m, c_f, d_f, r_f, b_f, k) in enumerate(results, 1):
    print(f"{rank:<5} {name:<25} {c_m:<14.1f} {d_m:<10.0f} {r_m:<12.4f} {b_m:<14.1f} {c_f:<14.1f} {b_f:<14.1f} {k:<8.1f}")

# Also show full-fit ranking for comparison
print("\n" + "=" * 120)
print("FULL-FIT RANKING (noSED, for comparison)")
print("=" * 120)
results_full = sorted(results, key=lambda x: x[8])
print(f"{'Rank':<5} {'Model':<25} {'χ²_full':<14} {'red_χ²_full':<12} {'BIC_full':<14} {'χ²_MIRI':<14} {'BIC_MIRI':<14}")
print("-" * 120)
for rank, (name, c_m, d_m, r_m, b_m, c_f, d_f, r_f, b_f, k) in enumerate(results_full, 1):
    print(f"{rank:<5} {name:<25} {c_f:<14.1f} {r_f:<12.4f} {b_f:<14.1f} {c_m:<14.1f} {b_m:<14.1f}")

print("\nDone.")

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')

spectra_dir = 'd:/Fudan_University/Research/JWST/NewApertureTrials/Watershed_Output'
output_dir = 'd:/Fudan_University/Research/JWST/NewApertureTrials/fitted_lines_watershed'
os.makedirs(output_dir, exist_ok=True)

clumps = ['Center', 'West', 'North', 'South']
grating = 'G395H'
c_kms = 299792.458

# Rest wavelengths in Angstroms
wl_rest = {
    'Hb': 4861.33,
    'O4959': 4958.91,
    'O5007': 5006.84,
    'N6548': 6548.05,
    'Ha': 6562.81,
    'N6584': 6583.45
}

def gaussian(x, amp, cen, sig):
    return amp * np.exp(-(x-cen)**2 / (2*sig**2))

def model_HBOIII(x, m, c, z, amp_Hb_n, amp_Hb_b, amp_O4959, sig_v_n, sig_v_b):
    # Centers
    cen_Hb = wl_rest['Hb'] * (1+z)
    cen_O49 = wl_rest['O4959'] * (1+z)
    cen_O50 = wl_rest['O5007'] * (1+z)
    
    # Widths (sigma)
    sig_Hb_n = cen_Hb * (sig_v_n / c_kms)
    sig_O49_n = cen_O49 * (sig_v_n / c_kms)
    sig_O50_n = cen_O50 * (sig_v_n / c_kms)
    sig_Hb_b = cen_Hb * (sig_v_b / c_kms)
    
    # Components
    cont = m * x + c
    Hb_n = gaussian(x, amp_Hb_n, cen_Hb, sig_Hb_n)
    Hb_b = gaussian(x, amp_Hb_b, cen_Hb, sig_Hb_b)
    O49 = gaussian(x, amp_O4959, cen_O49, sig_O49_n)
    O50 = gaussian(x, amp_O4959 * 2.98, cen_O50, sig_O50_n) # Fixed 1:3 ratio
    
    return cont + Hb_n + Hb_b + O49 + O50

def model_HaNII(x, m, c, z, amp_Ha_n, amp_Ha_b, amp_N6548, sig_v_n, sig_v_b):
    # Centers
    cen_Ha = wl_rest['Ha'] * (1+z)
    cen_N64 = wl_rest['N6548'] * (1+z)
    cen_N68 = wl_rest['N6584'] * (1+z)
    
    # Widths (sigma) 
    sig_Ha_n = cen_Ha * (sig_v_n / c_kms)
    sig_N64_n = cen_N64 * (sig_v_n / c_kms)
    sig_N68_n = cen_N68 * (sig_v_n / c_kms)
    sig_Ha_b = cen_Ha * (sig_v_b / c_kms)    # 假设所有线来自同一片云团源，多普勒展宽与中心波长成正比
    
    # Components
    cont = m * x + c
    Ha_n = gaussian(x, amp_Ha_n, cen_Ha, sig_Ha_n)
    Ha_b = gaussian(x, amp_Ha_b, cen_Ha, sig_Ha_b)
    N64 = gaussian(x, amp_N6548, cen_N64, sig_N64_n)
    N68 = gaussian(x, amp_N6548 * 2.95, cen_N68, sig_N68_n) # Fixed theoretical ratio
    
    return cont + Ha_n + Ha_b + N64 + N68

results_hboiii = []
results_ha = []

for clump in clumps:
    file_path = os.path.join(spectra_dir, f'{clump}_spectrum.txt')
    if not os.path.exists(file_path): continue
    
    data = np.loadtxt(file_path, skiprows=1)
    # Filter 0 or negative fluxes
    valid = data[:, 1] > -1e-18
    wl_obs = data[valid, 0] * 10000 
    flux = data[valid, 1] * 1e19 # Scale up to avoid floating point precision issues in optimizers
    
    print(f"\\nFitting Clump: {clump}")
    
    # -----------------------------
    # 1. H-beta + [OIII] Window
    # -----------------------------
    w1_mask = (wl_obs > 36000) & (wl_obs < 40500)
    w1, f1 = wl_obs[w1_mask], flux[w1_mask]
    
    if len(w1) > 20:
        # p0: m, c, z, a_hb_n, a_hb_b, a_o4959, sig_n, sig_b
        p0 = [0, np.median(f1), 6.85, 1.0, 1.0 if clump=='West' else 0, 1.0, 200, 3000]
        # bounds
        bounds_min = [-np.inf, -100, 6.80, 0, 0, 0, 10, 800]
        bounds_max = [np.inf, 100, 6.90, np.inf, np.inf, np.inf, 800, 10000]
        
        try:
            popt1, _ = curve_fit(model_HBOIII, w1, f1, p0=p0, bounds=(bounds_min, bounds_max), maxfev=10000)
            
            # Reconstruct components for plot
            m, c, z, a_hb_n, a_hb_b, a_o49, sn, sb = popt1
            cont = m * w1 + c
            hb_n = gaussian(w1, a_hb_n, wl_rest['Hb']*(1+z), (wl_rest['Hb']*(1+z))*(sn/c_kms))
            hb_b = gaussian(w1, a_hb_b, wl_rest['Hb']*(1+z), (wl_rest['Hb']*(1+z))*(sb/c_kms))
            o49 = gaussian(w1, a_o49, wl_rest['O4959']*(1+z), (wl_rest['O4959']*(1+z))*(sn/c_kms))
            o50 = gaussian(w1, a_o49*2.98, wl_rest['O5007']*(1+z), (wl_rest['O5007']*(1+z))*(sn/c_kms))
            
            plt.figure(figsize=(10,6))
            plt.plot(w1/10000, f1/1e19, 'k-', label='Data', ds='steps-mid', alpha=0.7)
            plt.plot(w1/10000, model_HBOIII(w1, *popt1)/1e19, 'r-', label='Total Fit', lw=2)
            plt.plot(w1/10000, cont/1e19, 'gray', linestyle='--', label='Continuum')
            plt.plot(w1/10000, (cont+hb_n)/1e19, 'b-', label='Hb Narrow', alpha=0.7)
            plt.plot(w1/10000, (cont+o49+o50)/1e19, 'g-', label='[OIII] doublets', alpha=0.7)
            if a_hb_b > 0.1:
                plt.plot(w1/10000, (cont+hb_b)/1e19, 'm--', label='Hb Broad', lw=2)
                
            plt.title(f"{clump} Clump | Hb + [OIII]")
            plt.xlabel("Observed $\mu$m")
            plt.ylabel("Flux (erg/s/cm$^2$/A)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{clump}_Hb_OIII_fit.png"), dpi=200)
            plt.close()
            
            results_hboiii.append({
                'Clump': clump, 'z': z, 'Narrow FWHM (km/s)': sn*2.355, 'Broad FWHM (km/s)': sb*2.355, #FWHM = 2sqrt{2ln2}\sigma
                'Flux_Hb_n': a_hb_n, 'Flux_Hb_b': a_hb_b, 'Flux_O5007': a_o49*2.98
            })
            print(f"  [Hb+OIII] Success: z={z:.4f}, narrow_width={sn*2.355:.0f}km/s")
        except Exception as e:
            print(f"  [Hb+OIII] Fit failed: {e}")

    # -----------------------------
    # 2. H-alpha + [NII] Window
    # -----------------------------
    w2_mask = (wl_obs > 50500) & (wl_obs < 52500)
    w2, f2 = wl_obs[w2_mask], flux[w2_mask]
    
    if len(w2) > 20:
        p0 = [0, np.median(f2), 6.85, 2.0, 2.0 if clump=='West' else 0, 0.5, 200, 3000]
        bounds_min = [-np.inf, -100, 6.80, 0, 0, 0, 10, 800]
        bounds_max = [np.inf, 100, 6.90, np.inf, np.inf, np.inf, 800, 10000]
        
        try:
            popt2, _ = curve_fit(model_HaNII, w2, f2, p0=p0, bounds=(bounds_min, bounds_max), maxfev=10000)
            
            m, c, z, a_ha_n, a_ha_b, a_n64, sn, sb = popt2
            cont = m * w2 + c
            ha_n = gaussian(w2, a_ha_n, wl_rest['Ha']*(1+z), (wl_rest['Ha']*(1+z))*(sn/c_kms))
            ha_b = gaussian(w2, a_ha_b, wl_rest['Ha']*(1+z), (wl_rest['Ha']*(1+z))*(sb/c_kms))
            n64 = gaussian(w2, a_n64, wl_rest['N6548']*(1+z), (wl_rest['N6548']*(1+z))*(sn/c_kms))
            n68 = gaussian(w2, a_n64*2.95, wl_rest['N6584']*(1+z), (wl_rest['N6584']*(1+z))*(sn/c_kms))
            
            plt.figure(figsize=(10,6))
            plt.plot(w2/10000, f2/1e19, 'k-', label='Data', ds='steps-mid', alpha=0.7)
            plt.plot(w2/10000, model_HaNII(w2, *popt2)/1e19, 'r-', label='Total Fit', lw=2)
            plt.plot(w2/10000, cont/1e19, 'gray', linestyle='--', label='Continuum')
            plt.plot(w2/10000, (cont+ha_n)/1e19, 'b-', label='Ha Narrow', alpha=0.7)
            plt.plot(w2/10000, (cont+n64+n68)/1e19, 'g-', label='[NII] doublets', alpha=0.7)
            if a_ha_b > 0.1:
                plt.plot(w2/10000, (cont+ha_b)/1e19, 'm--', label='Ha Broad', lw=2)
                
            plt.title(f"{clump} Clump | Ha + [NII]")
            plt.xlabel("Observed $\mu$m")
            plt.ylabel("Flux (erg/s/cm$^2$/A)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{clump}_Ha_NII_fit.png"), dpi=200)
            plt.close()
            
            results_ha.append({
                'Clump': clump, 'z': z, 'Narrow FWHM (km/s)': sn*2.355, 'Broad FWHM (km/s)': sb*2.355,
                'Flux_Ha_n': a_ha_n, 'Flux_Ha_b': a_ha_b, 'Flux_N6584': a_n64*2.95
            })
            print(f"  [Ha+NII] Success: z={z:.4f}, narrow_width={sn*2.355:.0f}km/s")
        except Exception as e:
            print(f"  [Ha+NII] Fit failed: {e}")

df_hboiii = pd.DataFrame(results_hboiii)
df_hboiii.to_csv(os.path.join(output_dir, 'hboiii_fit_results.csv'), index=False)
df_ha = pd.DataFrame(results_ha)
df_ha.to_csv(os.path.join(output_dir, 'ha_fit_results.csv'), index=False)

print("\\nProcessing complete! Plots and CSVs saved.")

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

miri_file = 'd:/Fudan_University/Research/JWST/extracted_photometry/miri_photometry.csv'
output_dir = 'd:/Fudan_University/Research/JWST/extracted_photometry'

if not os.path.exists(miri_file):
    print("MIRI photometry file not found.")
    exit()

df = pd.read_csv(miri_file)

clumps = df['Clump ID'].unique()

plt.figure(figsize=(10, 6))

# Plot SED for each clump
for clump in clumps:
    clump_data = df[df['Clump ID'] == clump].sort_values(by='Wavelength_um')
    
    # Filter out pure 0 or negative fluxes for log scale plot
    valid_data = clump_data[clump_data['Flux_uJy'] > 0]
    
    plt.errorbar(
        valid_data['Wavelength_um'], 
        valid_data['Flux_uJy'], 
        yerr=valid_data['Flux_Err_uJy'], 
        label=clump, 
        marker='o', 
        linestyle='-',
        capsize=3
    )

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Observed Wavelength ($\mu$m)')
plt.ylabel('Flux Density ($\mu$Jy)')
plt.title('MIRI Photometry SED by Clump')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)

output_plot = os.path.join(output_dir, 'miri_seds.png')
plt.savefig(output_plot, dpi=300)
print(f"MIRI SED plot saved to {output_plot}")

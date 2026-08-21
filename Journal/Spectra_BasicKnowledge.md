# Basic Knowledge of Common AGN Spectral Lines

Active Galactic Nuclei (AGN) are incredibly luminous structures powered by supermassive black holes accreting matter. Their spectra are characterized by a strong power-law continuum and prominent emission lines. These lines are crucial diagnostics for understanding the physical conditions (temperature, density, ionization state), kinematics, and black hole masses of AGN.

## 1. Formation Mechanisms

Emission lines in AGN generally fall into two physical categories based on how they are produced:

### A. Recombination Lines (Permitted Lines)

* **Examples**: H$\alpha$, H$\beta$, Ly$\alpha$, He II.

* **Mechanism**: These form when a free electron recombines with an ionized atom (like $H^+$ or $He^{++}$) and cascades down through discrete atomic energy levels. Each drop to a lower energy state emits a photon of a specific wavelength.

* **Location**: These can be formed in both high-density environments (the **Broad Line Region** or BLR, very close to the black hole) and low-density environments (the **Narrow Line Region** or NLR, further out). Therefore, H$\alpha$ and H$\beta$ often show a dual profile: a broad base with a narrow peak.

### B. Collisionally Excited Lines (Forbidden Lines)

* **Examples**: [O III], [N II], [S II], [O II].

* **Notation**: Always denoted by square brackets (e.g., `[O III]`).

* **Mechanism**: Free electrons collide with heavy elements (metals like Oxygen, Nitrogen), bumping their remaining electrons into excited states. The electrons then slowly decay back to the ground state, emitting a photon. However, quantum mechanics dictates these specific transitions are highly improbable ("forbidden"), meaning the electron stays excited for a very long time before decaying.

* **Location**: Because the excited state lasts so long, if the surrounding gas is dense (like the BLR), another electron will collide and de-excite the atom *before* it can emit a photon (collisional de-excitation). Therefore, **forbidden lines ONLY form in low-density environments**, specifically the Narrow Line Region (NLR) extended far beyond the AGN core. Thus, they are practically always *narrow*.

---

## 2. Common Rest-Frame Ultraviolet Lines

These lines are often the primary features observed in high-redshift quasars when shifted into the optical/near-IR bands.

| Line ID | Rest Wavelength ($\AA$) | Type | Description |
| :--- | :--- | :--- | :--- |
| **Ly $\alpha$** | 1215.67 | Permitted | Lyman Alpha. The strongest emission line in the UV. Transition of Hydrogen $n=2 \rightarrow 1$. |
| **N V** | 1240.14 | Permitted | Nitrogen V. High ionization state line often blended closely with Ly$\alpha$. |
| **C IV** | 1549.06 | Permitted | Carbon IV doublet. Very prominent broad line. Extensively used to estimate black hole masses at high redshift. |
| **C III]** | 1908.73 | Semi-forbidden | Used alongside C IV to estimate electron densities in the Broad Line Region. |
| **Mg II** | 2797.92, 2802.71 | Permitted | Magnesium II doublet. A strong, broad line crucial for black hole mass estimation at intermediate redshifts ($1 \lesssim z \lesssim 2.5$). |

## 3. Common Rest-Frame Optical Lines

These are classical diagnostics heavily utilized for nearby galaxies and intermediate-to-high redshift galaxies observed with JWST instruments like NIRSpec IFU.

| Line ID | Rest Wavelength ($\AA$) | Type | Description |
| :--- | :--- | :--- | :--- |
| **[O II]** | 3726.03, 3728.82 | Forbidden | Oxygen II doublet. Excellent tracer of star formation and NLR gas. |
| **[Ne III]** | 3868.76 | Forbidden | Neon III. Requires high energy to ionize. |
| **H $\gamma$** | 4340.47 | Permitted | Hydrogen Gamma recombination. |
| **H $\beta$** | 4861.33 | Permitted | Hydrogen Beta. An anchor point for analyzing AGN. Seeing its broad component confirms a direct view of the BLR (Type 1 AGN). |
| **[O III]** | 4958.91, 5006.84 | Forbidden | The classic AGN hallmark. Very strong in AGN because it requires hard UV/X-ray photons to doubly ionize Oxygen. The $\lambda5007$ line is intrinsically ~2.98x stronger than $\lambda4959$. |
| **[O I]** | 6300.30 | Forbidden | Neutral Oxygen. Formed in partially ionized zones often excited by shocks or hard AGN radiation. |
| **H $\alpha$** | 6562.81 | Permitted | The strongest prominent optical hydrogen line. |
| **[N II]** | 6548.05, 6583.45 | Forbidden | Nitrogen II. Flanks H$\alpha$. Used critically in **BPT diagrams** (comparing [N II]/H$\alpha$ vs [O III]/H$\beta$) to distinguish AGN from pure star-forming galaxies. |
| **[S II]** | 6716.44, 6730.82 | Forbidden | Sulfur II doublet. The flux ratio between these two identical lines ($\lambda6716/\lambda6731$) is the most popular way to measure exact electron density ($n_e$) in the NLR. |

---

[!TIP]

### Why Do We Look at Ratios?

In astronomy, observing absolute flux is difficult because objects are far away and light is often absorbed by intervening cosmic dust. Looking at **ratios of lines that are close together in wavelength** (like [O III] $\lambda5007$ / H$\beta$, or [N II] $\lambda6584$ / H$\alpha$) makes the measurement practically immune to dust reddening, yielding highly reliable physical diagnostics!

## Break

### Lyman Break

The Lyman-break galaxy selection technique relies upon the fact that radiation at higher energies than the Lyman limit (photon let H -> H+) at 912 Å (galaxy rest-frame) is almost completely absorbed by neutral gas around star-forming regions of galaxies.

### Lyman alpha Break

1216 Å (rest-frame). Photon with energy at this wavelength can make electro from n=1 ro n=2. This break is formed when the Lyman Forest is deep enough.

### Balmer Break

364.5nm

## Narrow and Broad Lines

In the spectra of AGN, emission lines are fundamentally categorized by their kinematic width (their "fatness" on a plot), which directly reveals where the emitting gas is located.

### 1. Broad Lines

* **Definition**: Emission lines with a very large velocity spread, typically showing a Full Width at Half Maximum (FWHM) of **$1,000$ to $>10,000$ km/s**.
* **Formation Principle**:
  * These lines originate from the **Broad Line Region (BLR)**, which is located extremely close to the central supermassive black hole (typically less than a parsec away).
  * **Why are they broad?** Because the gas clouds are so deep inside the black hole's gravitational well, they orbit at incredibly fast, relativistic speeds. This massive variation in velocity towards and away from our line-of-sight causes extreme **Doppler broadening** of the light.
  * Estimate black hole mass from FWHM of broad lines:
  * **Note**: Only **Permitted lines** (like H$\alpha$, H$\beta$, Ly$\alpha$, C IV) appear broad. The gas density in the BLR is exceptionally high ($>10^9$ particles/cm$^3$). In this crowded environment, atoms are constantly bumping into each other, meaning "Forbidden" transitions are completely suppressed by collisions before they can ever emit a photon.

### 2. Narrow Lines

* **Definition**: Emission lines with a much smaller velocity spread, typically showing a FWHM of a **few hundred km/s** (usually $< 500 - 1000$ km/s).
* **Formation Principle**:
  * These lines originate from the **Narrow Line Region (NLR)**, which is much further away from the black hole (tens to thousands of parsecs away), extending out into the host galaxy itself.
  * **Why are they narrow?** Because these gas clouds are further out, the gravitational pull from the central black hole is much weaker. The gas moves much more slowly, resulting in very little Doppler broadening.
  * **Note**: Both **Permitted lines** and **Forbidden lines** (like [O III], [N II], [S II]) emit here. The gas density in the NLR is relatively low ($\approx 10^3$ to $10^6$ particles/cm$^3$). This sparse environment is exactly what is required for Forbidden transitions to survive long enough to emit photons without being interrupted by a collision!

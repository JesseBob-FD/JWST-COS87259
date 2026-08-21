# First stage mission

## The general Introduction of this project

We have obtained JWST/NIRSpec IFU spectroscopy and JWST/MIRI imaging of an extremely luminous AGN at z ~ 6.8.

This target was identified and its radio and far-IR emission was studied in[Endsley et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022MNRAS.512.4248E/abstract), characterizing as a dust obscured AGN with supermassive black hole (mass_BH10^9 M_sun) harbored by a very massive galaxy (M_star10^10 M_sun) at high redshift.

The goal of the JWST observations is to characterize the host galaxy properties, dust emission, and the galaxy environment nearby this source using the rest - frame optical IFU spectra (with JWST/NIRSpec) and rest - frame IR photometry (with JWST/MIRI). Ultimately we aim to understand how the early supermassive black hole and the dust content formed and grew in such a short period, how the AGN and the host galaxy interplay with each other.

Some useful information can be found in the abstract of JWST proposals:

- [https://www.stsci.edu/jwst/phase2-public/4877.pdf]
- [https://www.stsci.edu/jwst-program-info/download/jwst/pdf/6576/]
- The JWST data have been reduced. The target appears to have multiple clumps and the next step is to analyze the spectra and imaging of individual clumps, modeling the spectral energy distributions (SEDs) and identifying which clumps may host the AGN, stars, and dust emission.

## The first stage mission

You can find the JWST data of the luminous AGN (COS-87259, z = 6.85, RA: 09:58:58.3, DEC: +01:39:20.2) in the folder of this workplace. The data include the NIRSpec IFU spectra (./NIRSpec_IFU/), MIRI imaging (./MIRI/), and NIRCam imaging (./NIRCam/), all have been reduced with the JWST data reduction pipeline [https://jwst-pipeline.readthedocs.io/en/stable/]. These data are in FITS files, and you can visualize FITS file with the DS9 tool [https://sites.google.com/cfa.harvard.edu/saoimageds9].

The NIRSpec IFU spectrum is 3D, with wavelength as one dimension and there is 2D flux image at each wavelength bin. The 2D flux image has a size of 3” x 3” with pixel size = 0.1". The spectra were taken with two grating/filter sets: PRISM (COS-87259_NOBG_prism-clear_s3d.fits) and G395H/F290LP (COS-87259_NOBG_g395h-f290lp_s3d.fits). PRISM has a wide wavelength coverage (0.6 - 5.3 μm, corresponding to rest-frame 800 - 6700 Angstrom) but low spectral resolution (R ~ 100). G395H has a much narrower wavelength coverage (2.8 - 5.2 μm, corresponding to rest-frame 3600 - 6600 Angstrom) and high resolution (R ~ 2700). The spectra need to remove the background, and you can find background subtraction code here (COS87259.ipynb) and the output background-subtracted spectra (COS-87259_BGSUB_prism-clear_s3d.fits, COS-87259_BGSUB_g395h-f290lp_s3d.fits).

One of the interesting things revealed by the NIRspec IFU PRISM (and NIRCam image) is that this object shows multiple clumps. Below is the F410M (3.87 - 4.30 μm) image of this object, and there are five clumps: center, west, north, south, and southeast.

![clumps](./Missionline/clumps.png)

In COS87259.ipynb, I also wrote code to quickly extract these clumps with elliptical apertures and make 1D low resolution PRISM flux spectra of these clumps (by coadding the 2D flux image of each clump) and plot them.

***You may play around with the code to see if you can improve the clump extraction (e.g., with different apertures, or simply coadd the pixels belong to each clump) etc.***

The spectra of these clumps look very different. **The Center has Balmer break with relatively weak H-beta+[OIII] emission lines.** The North has much weaker emission lines with a Balmer break. The West has strong and broad emission lines. The South has strong, narrow emission lines without Balmer break. And the Southeast does not have any significant features.

The Balmer break likely indicates the presence of evolved stellar populations (e.g., A stars). The broad emission lines (West) may come from a broad-line AGN, and the strong, narrow lines (South) may come from very young stellar populations (e.g., O & early type B stars).

***The goal is to understand what the spectra (as well as multi-wavelength imaging) imply for each clump, e.g., whether a clump is dominated by star formation, AGN, dust or so.***

An interesting task is to compare the multi-wavelength observations of these clumps.

**IR, sub-mm, and radio observations may suggest that the AGN is centered at the west part of the object, though the spatial resolution is much lower than JWST: [Endsley et al. 2022](./Endsley_2022.pdf); [Endsley et al. 2023](./Endsley_2023.pdf)**. (Figure 1b from Endsley et al. 2023 is attached below). This may be consistent with what we see from the NIRSpec spectrum of the western clump (showing broad lines which is common in broad line AGN). There is a AGN-driven radio jet feature which may be associated with the southeastern clump. The southern clump is likely purely dominated by young stars (or shocks) without sub-mm or radio detections.

![Endsley et al. 2023-1b](./Missionline/Endsley_2023-1b.png)

We would like to check the G395H spectra which have a much higher spectral resolution to better quantify the rest-frame optical emission line (H-beta, [OIII]4959,5007, H-alpha) profiles, and also the MIRI images which have much higher spatial resolution.

Can you please:

1. Try to extract the spectra of the five individual clumps from the G395H IFU spectra;

2. Try to extract the MIRI photometry flux (nine broadband filters, here you can find MIRI information: [https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument#gsc.tab=0]) of individual clumps?

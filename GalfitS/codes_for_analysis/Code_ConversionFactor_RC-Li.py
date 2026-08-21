from galfits import gsutils
img = fits.open(filepath+'{0}_{1}_cutout.fits'.format(tagname, Band[loopx]))
header = img[0].header
ZP_GALFIT = 2.5*np.log10(3631/(header['PIXAR_SR']*1e6))
magab = ZP_GALFIT
a = gsutils.ABmag_to_covf(magab,gsutils.effective_wave[jwstbands[loopx]])
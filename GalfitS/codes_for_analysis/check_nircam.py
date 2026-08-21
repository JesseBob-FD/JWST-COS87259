import astropy.io.fits as fits

for fname in ['F115W_cos87259_sci.fits', 'F200W_cos87259_sci.fits', 'F410M_cos87259_sci.fits']:
    hdu = fits.open(f'NIRCam/{fname}')
    print(f'{fname}:')
    print(f'  Shape: {hdu[1].data.shape}')
    print(f'  Extensions: {[h.name for h in hdu]}')
    for key in ['PIXAR_SR', 'BUNIT', 'CRVAL1', 'CRVAL2', 'PHOTMJSR']:
        if key in hdu[1].header:
            print(f'  {key}: {hdu[1].header[key]}')
    hdu.close()
    print()

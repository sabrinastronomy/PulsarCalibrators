import ATNFCatalog as a
import pandas
import numpy as np
import matplotlib.pyplot as plt
def clean_text_file(filename="psr_snr_list_normalised_lower_transit.txt"):
    # removes extra columns in transit .csv file
    df = pandas.read_csv(filename, delim_whitespace=True)
    del df["chime_snr"]
    del df["gbt_snr"]
    df.to_csv("update_lower.txt", index=False, sep=" ")
    return


if __name__ == "__main__":
    ### CHECKING OVERLAP BETWEEN PULSARS IN CHIME AND ATNF
    all_CHIME = a.PulsarLimitedList("CHIME", 0, np.inf, fix_automatic=False)
    all_ATNF = a.PulsarCatalog()

    ra_neg10 = []
    dec_neg10 = []
    flux_neg10 = []
    ra_err_neg10 = []
    for ra, dec, flux, ra_err, dec_err in zip(all_ATNF.rajd, all_ATNF.decjd, all_ATNF.fluxes, all_ATNF.raj_err_mas, all_ATNF.decj_err_mas):
        if dec > -10 and dec_err < 100 and ra_err < 100 and dec_err > 0. and ra_err > 0.:
            ra_neg10.append(int(ra/15))
            dec_neg10.append(dec)
            flux_neg10.append(flux)
            ra_err_neg10.append(ra_err)
    plt.scatter(ra_neg10, dec_neg10, label="All ATNF (DEC > -10 && $loc_{err}$ < 100 mas)")
    plt.scatter(all_CHIME.df["RA"].values, all_CHIME.df["Dec"].values, label="All CHIME")
    plt.legend(loc=4)
    plt.xlabel("RA (hours)")
    plt.ylabel("DEC ($^{\circ}$)")
    plt.title("Missing Pulsars in CHIME Pulsar Detections")
    plt.savefig("missing_pulsars.pdf")
    plt.show()
    plt.close()

    ### GENERATING CHIME/GBT PULSAR RA HISTOGRAMS
    types = ["CHIME", "GBT"]
    snrs = [15, 50]
    locs = [110, 50]
    for type in types:
        plls = [] # PulsarLimitedList objects in an array
        for loc, snr in zip(locs, snrs):
            plls.append(a.PulsarLimitedList(type, snr, loc))
        # plotting two pulsar lists
        plls[0].plotter(plls[1], "name_{}_SNRs_{}_locs_{}".format(type, snrs, locs))


    lone = a.PulsarLimitedList("CHIME", 11, 150)
    lone.plotter_alone("chime_all")
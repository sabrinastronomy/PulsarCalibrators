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

def read_names(filename="daniele_names.txt"):
    return open(filename).read().splitlines()

DANIELE_NAMES = read_names()

if __name__ == "__main__":
    ### CHECKING OVERLAP BETWEEN PULSARS IN CHIME AND ATNF
    all_ATNF = a.RadioPulsarCatalog(download=False, wanted_names=DANIELE_NAMES, th=False)
    all_CHIME = a.PulsarLimitedList("CHIME", 0, np.inf, np.inf, fix_automatic=True)

    ra_neg10 = []
    dec_neg10 = []
    flux_neg10 = []
    ra_err_neg10 = []
    names_missing = []
    names_in_chime = np.asarray(all_CHIME.name_match)

    uniques, counts = np.unique(names_in_chime, return_counts=True)
    for x, cou in zip(uniques, counts):
        if cou > 1:
            print("This pulsar is repeated in CHIME list: {}".format(x))
    count = 0
    for name in np.unique(names_in_chime):
        if not "upper" in name and not "lower" in name:
            count += 1

    count = 0
    names_found = []
    lowest_dec = min(all_CHIME.get_decs())
    for jname, bname, ra, dec, flux, ra_err, dec_err in zip(all_ATNF.jnames, all_ATNF.bnames, all_ATNF.rajd, all_ATNF.decjd, all_ATNF.fluxes, all_ATNF.raj_err_mas, all_ATNF.decj_err_mas):
        if dec > lowest_dec:
            ra_neg10.append(ra/15)
            dec_neg10.append(dec)
            flux_neg10.append(flux)
            ra_err_neg10.append(ra_err)
            count += 1
            if jname not in names_in_chime and bname not in names_in_chime:
                names_missing.append(jname)

    np.savetxt("missing_pulsars.csv", np.unique(names_missing), delimiter = ",", fmt="%s")
    plt.scatter(ra_neg10, dec_neg10, label="All ATNF (DEC > -10)")
    plt.scatter(all_CHIME.new_df["RA"].values, all_CHIME.new_df["Dec"].values, label="All CHIME", alpha = 0.5)
    plt.legend(loc=4)
    plt.xlabel("RA (hours)")
    plt.ylabel("DEC ($^{\circ}$)")
    plt.title("Missing ATNF Pulsars in CHIME Pulsar Observations")
    plt.savefig("missing_pulsars.png", dpi=600)
    plt.close()

    ### GENERATING CHIME/GBT PULSAR RA HISTOGRAMS
    # types = ["CHIME"]
    # snrs = [15, 50]
    # locs = [10, 10]
    # for type in types:
    #     plls = [] # PulsarLimitedList objects in an array
    #     for loc, snr in zip(locs, snrs):
    #         plls.append(a.PulsarLimitedList(type, snr, loc, 10))
    #     # plotting two pulsar lists
    #     plls[0].plotter(plls[1], "name_{}_SNRs_{}_locs_{}".format(type, snrs, locs))

    lone = a.PulsarLimitedList("CHIME", 15, 110, 10)
    lone2 = a.PulsarLimitedList("CHIME", 15, 10, 10)
    lone2.plotter_alone("CHIME_gt_15_lt_10_bins_48", bin=48, loc=True)
    lone2.plotter_alone("CHIME_gt_15_lt_10_bins_24", bin=24, loc=True)
    lone2.snr_ra_color_plotter_alone("CHIME_gt_50_lt_10_SNR_theta", "theta")
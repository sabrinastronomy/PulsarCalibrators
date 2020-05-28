import ATNFCatalog as a
import pandas

def clean_text_file(filename="psr_snr_list_normalised_lower_transit.txt"):
    # removes extra columns in transit .csv file
    df = pandas.read_csv(filename, delim_whitespace=True)
    del df["chime_snr"]
    del df["gbt_snr"]
    df.to_csv("update_lower.txt", index=False, sep=" ")
    return


if __name__ == "__main__":
    types = ["CHIME", "GBT"]
    snrs = [15, 50]
    locs = [110, 50]
    for type in types:
        plls = [] # PulsarLimitedList objects in an array
        for loc, snr in zip(locs, snrs):
            plls.append(a.PulsarLimitedList(type, snr, loc))
        # plotting two pulsar lists
        plls[0].plotter(plls[1], "name_{}_SNRs_{}_locs_{}".format(type, snrs, locs))


    lone = a.PulsarLimitedList("GBT", 10, 110)
    lone.plotter_alone("gbt_all")
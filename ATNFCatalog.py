import datetime
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas
from psrqpy import QueryATNF


class PulsarCatalog:
    """
    Loads all data from the ATNF catalog and contains various methods to manipulate this data with users' desired constraints
    """

    def __init__(self, download=True):
        if download:
            # Runs if you don't have a recently downloaded version of the ATNF catalog.
            # Turn this off if you already have a downloaded catalog as it slows the code down quite a bit.
            query = QueryATNF()
            query.save('all_ATNF.npy')
        else:
            # Loading the ATNF catalog from a .npy file. This won't work if you
            # haven't saved it to a .npy file yet. See commented lines above.

            query = QueryATNF(loadquery='all_ATNF.npy')
            numstring = 'Using ATNF catalogue version {} which contains {} pulsars.'
            print(numstring.format(query.get_version, query.num_pulsars))

        # Desired query parameters from the ATNF catalog saved version
        self.table = table = query.table

        # RAJ: Right ascension(J2000)(hh:mm:ss.s)
        # DecJ: Declination(J2000)(+dd:mm:ss)
        self.raj = np.asarray(table['RAJ'])
        self.decj = np.asarray(table['DECJ'])

        # RAJD: Right ascension(J2000)(degrees)
        # DecJD: Declination(J2000)(degrees)
        self.rajd = np.asarray(table['RAJD'])
        self.decjd = np.asarray(table['DECJD'])

        # RAJ & DECJ error
        raj_err = np.asarray(table['RAJ_ERR'])
        self.raj_err = np.nan_to_num(raj_err)
        decj_err = np.asarray(table['DECJ_ERR'])
        self.decj_err = np.nan_to_num(decj_err)

        # Epoch of position, defaults to PEpoch (MJD)
        self.pos_epoch = np.asarray(table['POSEPOCH'])

        # Epoch of period or frequency (MJD)
        self.period_epoch = np.asarray(table['PEPOCH'])

        # Mean flux density at 400 MHz (mJy)
        self.fluxes = np.asarray(table['S400'])

        #  B and J names of pulsars
        self.bnames = np.asarray(table['BNAME'])
        self.jnames = np.asarray(table['JNAME'])

        # Proper motion of RA
        pm_ra = np.asarray(table['PMRA'])
        self.pm_ra = np.nan_to_num(pm_ra)

        # Proper motion of DEC
        pm_dec = np.asarray(table['PMDEC'])
        self.pm_dec = np.nan_to_num(pm_dec)

    def get_names_pos_unc(self):
        self.convert()
        j_name_dict = {}
        b_name_dict = {}
        for bname, jname, ra, dec in zip(self.bnames, self.jnames, self.raj_err_mas, self.decj_err_mas):
            b_name_dict[bname] = [ra, dec]
            j_name_dict[jname] = [ra, dec]
        return b_name_dict, j_name_dict

    def convert(self):
        """
        Convert RA and DEC error to milliarcseconds
        :return:
        """
        ra_err_fixed = self.fix_err('ra', self.raj_err)
        dec_err_fixed = self.fix_err('dec', self.decj_err)

        self.raj_err_mas = ra_err_fixed * 3.6e6 # converted errors to mas, 1 degree = 3.6e6 mas
        self.decj_err_mas = dec_err_fixed * 3.6e6 # converted errors to mas, 1 degree = 3.6e6 mas

        return

    def fix_err(self, typ, xerrarr):
        """
        Fix RA and DEC error since units on error given depends on precision of RA/DEC.

        Important note on RA (hh:mm:ss.s)
        24 RA hours = 360 degrees
        1 RA hour = 15 degrees
        1 RA min = 15'
        1 RA second = 15"

        1 DEC second = 1"
        1 DEC hour =

        :param typ:
        :return:

        Important note on DEC (+dd:mm:ss):
        DEC in degrees:arcminutes:arcseconds
        """
        xerrarr_fix = []
        for x, y_J, y_D, xerr in zip(self.raj, self.decj, self.decjd, xerrarr):
            if typ is 'ra':
                xerrarr_fix.append(self.get_err_ra(x, y_D, xerr))
            elif typ is 'dec':
                xerrarr_fix.append(self.get_err_dec(y_J, xerr))
            else:
                print("Invalid type input!")

        return np.round(np.asarray(xerrarr_fix, dtype=np.float64), 10)

    def get_err_ra(self, x, y, xerr):
        """
        Convert ra errors in seconds or minutes to degrees
        """
        if x.count(':') == 2:
            return xerr / 240 * np.cos(np.radians(y))  # (RA) seconds to degrees
        elif x.count(':') == 1:
            return xerr / 4 * np.cos(np.radians(y))  # (RA) minutes to degrees
        else:
            print("Invalid input!")

    def get_err_dec(self, x, xerr):
        """
        Convert dec errors in arcseconds or arcminutes to degrees
        """
        if x.count(':') == 2:
            return xerr / 3600  # arcseconds to degrees
        elif x.count(':') == 1:
            return xerr / 60  # arcminutes to degrees
        else:
            print("Invalid input!")


class PulsarLimitedList(PulsarCatalog):
    def __init__(self, type, min_snr, max_local, filename = "psr_snr_list_normalised.txt", fix_automatic=True):
        """
        :param filename: text file output from CHIME pulsar which is used as input
        Columns are: PSR, S/N, sigma(S/N), integration_time, normalised_snr, RA, Dec,
        where PSR is the pulsar name
        :param min_snr: minimum SNR to be considered for list, e.g., 50
        :param max_local: maximum error in MILLIARCSECONDS in RA/DEC to be considered for list, e.g. 100 mas
        :param GBT: True if GBT SNRs wanted
        :param fix_automatic: True if automatically constraining pulsar list with all constraints
        """
        PulsarCatalog.__init__(self)
        self.df = self.get_data_csv(filename)
        self.max_local = max_local
        self.min_snr = min_snr
        self.b_name_dict_atnf, self.j_name_dict_atnf = self.get_names_pos_unc()
        self.type = type
        self.GBT = (self.type == "GBT")

        # Pulsar name replacements where these is a mismatch between the CHIME/ATNF names
        self.pulsar_replacements = {"J1327+3423": "J1326+33", "J2228+3041": "J2227+30",
                                    "J1629+43": "J1628+4406", "J2122+54": "J2123+5434",
                                    "J1954+43": "J1954+4357", "J2017+59": "J2017+5906",
                                    "J1748+59": "J1749+59", "J0742+4110": "J0740+41",
                                    "J0325+67": "J0325+6744", "J1941+02": "J1940+0239",
                                    "J1836+51": "J1836+5925"}
        if fix_automatic:
            self.filter_SNR()
            self.filter_local()
            self.save_csv()


    def get_data_csv(self, filename):
        """
        Reads in data from a text file delimited by whitespace
        :param filename: location of text file, assumining delimiter is whitespace
        :return: Pandas dataframe cont6ain ing file contents
        """
        df = pandas.read_csv(filename, delim_whitespace=True)
        return df

    def filter_SNR(self):
        """
        Filter CHIME pulsar list above min SNR
        :return:
        """
        df = self.df

        # Constraining list based on GBT/CHIME SNR
        if self.GBT:
            df = df[self.get_gbt_snr_from_CHIME(df.normalised_snr, df.Dec) > self.min_snr]
        else:
            df = df[df.normalised_snr > self.min_snr]

        # Getting B/J Pulsar name, cutting off "lower following name" for pulsars from lower transit
        df.insert(0, "Pulsar Name", df.PSR, True) # True allows duplicates
        df.PSR = [n.split("_")[0] for n in df.PSR]

        # manually adjusting pulsar names
        df["PSR"].replace(self.pulsar_replacements, inplace=True)

        # manually omitting pulsars that can not be paired to ATNF or are new
        self.df = df[(df.PSR != "J0534-13") & (df.PSR != "J0414+31") & (df.PSR != "J2022+2534") & (df.PSR != "J2044+28") & (
                    df.PSR != "J2108+45") & (df.PSR != "J0406+3039") & (df.PSR != "J1239+32") & (df.PSR != "J1907+57")]
        return

    def filter_local(self):
        df = self.df
        match_names = df["PSR"]
        ra_err_match = []
        dec_err_match = []
        j_name_dict= self.j_name_dict_atnf
        b_name_dict = self.b_name_dict_atnf

        count = 0
        for name in match_names:
            if name in j_name_dict.keys():
                ra_err_match.append(j_name_dict[name][0])
                dec_err_match.append(j_name_dict[name][1])
            elif name in b_name_dict.keys():
                ra_err_match.append(b_name_dict[name][0])
                dec_err_match.append(b_name_dict[name][1])
            else:
                print("{} pulsars successfully found".format(count))
                print("{} not in either list!".format(name))
                sys.exit()
            count += 1

        df["ra_err_mas"] = ra_err_match
        df["dec_err_mas"] = dec_err_match

        # Filtering out pulsars with errors greater than max local or that are equal to 0
        self.df = df[(df.ra_err_mas < self.max_local) & (df.dec_err_mas < self.max_local) & ((df.ra_err_mas != 0) | (df.dec_err_mas != 0))]
        return

    def get_RAs(self):
        return self.df["RA"]


    def save_csv(self):
        self.df.sort_values(by="RA", inplace=True)
        saving_df = self.df.rename(columns={"integration_time":"integration_time (s)",
                                "normalised_snr":"SNR (normalized to 10 min integration time)",
                                "RA":"RA (hours)", "Dec":"DEC (degrees)", "ra_err_mas":"RA_err (mas)",
                                "dec_err_mas":"DEC_err (mas)"})
        del saving_df["PSR"]
        saving_df.to_csv("pulsars_snr_{}_mas_{}_{}.csv".format(self.min_snr, self.max_local, self.type), index=False)

    def get_gbt_snr_from_CHIME(self, normalized_snr, dec):
        """
        Scale normalized SNR from CHIME to scaled SNR at GBT latitude
        """
        gbt_latitude = 38.4330
        chime_latitude = 49.3208
        gbt_snr = normalized_snr * (np.cos(np.degrees(dec - gbt_latitude)) / np.cos(np.degrees(dec - chime_latitude)))
        return gbt_snr

    def plotter(self, other_cat, name):
        plt.xlim([0, 23])
        ra_1 = self.get_RAs()
        ra_2 = other_cat.get_RAs()
        ra_1.plot.hist(bins=24, color ="green", edgecolor = 'black', label="{} S/N > {}, $RA_{{err}}$ & $DEC_{{err}}$ < {} mas".format(self.type, self.min_snr, self.max_local))
        ra_2.plot.hist(bins=24, color="red", edgecolor='black', label="{} S/N > {}, $RA_{{err}}$ & $DEC_{{err}}$ < {} mas".format(other_cat.type, other_cat.min_snr, other_cat.max_local))
        plt.xlabel("RA (hours)")
        plt.ylabel("# of Pulsars")
        plt.legend()
        plt.savefig("{}.png".format(name))
        plt.savefig("{}.pdf".format(name))
        plt.close()

    def plotter_alone(self, name):
        plt.xlim([0, 23])
        ra_1 = self.get_RAs()
        ra_1.plot.hist(bins=24, color ="green", edgecolor = 'black', label="{} S/N > {}, $RA_{{err}}$ & $DEC_{{err}}$ < {} mas".format(self.type, self.min_snr, self.max_local))
        plt.xlabel("RA (hours)")
        plt.ylabel("# of Pulsars")
        plt.legend()
        plt.savefig("{}.png".format(name))
        plt.savefig("{}.pdf".format(name))
        plt.close()

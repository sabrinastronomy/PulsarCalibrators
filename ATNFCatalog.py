import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pyne2001
from psrqpy import QueryATNF


class RadioPulsarCatalog:
    """
    Loads all data from the ATNF catalog and contains various methods to manipulate this data with users' desired constraints
    """

    def __init__(self, download=False, correct_for_pm=True, th=True):
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

        self.glon = np.nan_to_num(np.asarray(table["GL"]))
        self.glat = np.nan_to_num(np.asarray(table["GB"]))

        # RAJ & DECJ error
        raj_err = np.asarray(table['RAJ_ERR'])
        self.raj_err = np.nan_to_num(raj_err, 100000)
        decj_err = np.asarray(table['DECJ_ERR'])
        self.decj_err = np.nan_to_num(decj_err, 100000) # if pm correction on

        # Epoch of position, defaults to PEpoch (MJD)
        self.pos_epoch = np.asarray(table['POSEPOCH'])

        # Epoch of period or frequency (MJD)
        self.period_epoch = np.asarray(table['PEPOCH'])

        # Mean flux density at 400 MHz (mJy)
        self.fluxes = np.asarray(table['S400'])
        self.type = "Radio"

        #  B and J names of pulsars
        self.bnames = np.asarray(table['BNAME'])
        self.jnames = np.asarray(table['JNAME'])
        self.type = np.asarray(table['TYPE'])

        # Proper motion of RA in degrees
        pm_ra = np.asarray(table['PMRA'])
        self.pm_ra = np.nan_to_num(pm_ra) / 3.6e6 # converting to degrees and removing nans

        # Proper motion of DEC in degrees
        pm_dec = np.asarray(table['PMDEC'])
        self.pm_dec = np.nan_to_num(pm_dec) / 3.6e6 # converting to degrees and removing nans

        # Proper motion of RA in degrees
        pm_ra_err = np.asarray(table['PMRA_ERR'])
        self.pm_ra_err_mas = pm_ra_err
        self.pm_ra_err_deg = np.nan_to_num(pm_ra_err, 100000) / 3.6e6 # converting to degrees and removing nans

        # Proper motion of DEC in degrees
        pm_dec_err = np.asarray(table['PMDEC_ERR'])
        self.pm_dec_err_mas = pm_dec_err
        self.pm_dec_err_deg = np.nan_to_num(pm_dec_err, 100000) / 3.6e6 # converting to degrees and removing nans

        # time of position observation (MJD)
        time_pos = np.asarray(table['POSEPOCH'])
        self.time_pos = np.nan_to_num(time_pos)

        # time of period observation (MJD)
        time_per = np.asarray(table['PEPOCH'])
        self.time_per = np.nan_to_num(time_per)

        fluxes = np.asarray(table['S400'])
        self.fluxes = fluxes
        self.convert()
        self.tau_s = np.asarray(table['TAU_SC'])
        self.tau_400 = self.tau_s*(((400e6)**(-4.4))/((1e9)**(-4.4))) # scattering time scaled to 400 MHz in seconds
        self.distance = self.table['DIST']  # in kpc
        self.distance_DM = self.table['DIST_DM']  # in kpc
        self.theta_atnf = []
        self.tau_400_NE2001 = []
        if th:
              # in milliarcseconds
            self.theta_NE2001 = []
            self.theta_NE2001_DM = []

            for glon, glat, distance, distance_dm, name, tau_400 in zip(self.glon, self.glat, self.distance, self.distance_DM, self.jnames, self.tau_400):
                if not isinstance(distance, float) or not isinstance(distance_dm, float):
                    theta = np.nan
                    theta_DM = np.nan
                    atnf = np.nan
                else:
                    theta = self.convert_taus_thetas(glon, glat, distance)
                    theta_DM = self.convert_taus_thetas(glon, glat, distance_dm)
                    atnf = self.convert_taus_thetas(glon, glat, distance, tau_400, ne2001=False)
                self.theta_atnf.append(atnf)
                self.theta_NE2001.append(theta)
                self.theta_NE2001_DM.append(theta_DM)
        else:
            self.theta_atnf = np.zeros(len(self.distance))
            self.theta_NE2001 = np.zeros(len(self.distance))
            self.theta_NE2001_DM = np.zeros(len(self.distance))

        if correct_for_pm:
            self.correct_pm()


    def convert_tau1GHz_tau400MHz(self, tau):
        """
        :param tau: tau in seconds at 1GHz
        :return: tau in seconds at 400 MHz
        """
        return tau * (((400e6) ** (-4.4)) / ((1e9) ** (-4.4)))

    def convert_taus_thetas(self, glon, glat, distance, tau=-100, ne2001=True):
        """
        :param tau: seconds
        :param glon: degrees
        :param glat: degrees
        :param distance: kiloparsecs (kpc)
        :param ne2001: whether or not to use ne2001 to get scattering times
        :return:
        """
        if ne2001:
            tau = pyne2001.get_dm_full(glon, glat, distance)['TAU'] * 1e-3
            tau = self.convert_tau1GHz_tau400MHz(tau)
            self.tau_400_NE2001.append(tau)
        distance = distance * 3.086e19
        thetas = (np.sqrt(tau * 3e8 / distance) * 180 / np.pi * 60 * 60) * 10 ** 3 # milliarcseconds
        return thetas

    def get_names_pos_unc(self):
        """
        This method grabs all pulsar names, corrected RA/DEC for PM and the corresponding error.
        It returns two dictionaries with keys as the b or j pulsar names and values as the ra and dec
        error in milliarcseconds.
        """
        j_name_dict = {}
        b_name_dict = {}
        for bname, jname, ra_err, dec_err, pm_ra_err, pm_dec_err, pmra, pmdec, ra, dec, theta_ne2001, theta_ne2001DM, theta_atnf, distance, distance_dm, \
            flux_density, l, b, tau_400, tau_400_ne2001 \
                in zip(self.bnames, self.jnames, self.raj_err_mas, self.decj_err_mas, self.pm_ra_err_mas,
                       self.pm_dec_err_mas, self.pm_ra, self.pm_dec, self.rajd, self.decjd, self.theta_NE2001, self.theta_NE2001_DM, self.theta_atnf, self.distance, self.distance_DM,
                       self.fluxes, self.glon, self.glat, self.tau_400, self.tau_400_NE2001):
            # Ad ding pm and RA/DEC errors in quadrature as total error in RA/DEC

            ra_err_quad = np.sqrt(ra_err**2 + pm_ra_err**2)
            dec_err_quad = np.sqrt(dec_err**2 + pm_dec_err**2)
            b_name_dict[bname] = [ra_err_quad, dec_err_quad, (ra, dec), theta_ne2001, theta_atnf, distance, flux_density, (l, b), (ra_err, dec_err), (pmra, pmdec), (pm_ra_err, pm_dec_err), distance_dm, theta_ne2001DM, tau_400, tau_400_ne2001]
            j_name_dict[jname] = [ra_err_quad, dec_err_quad, (ra, dec), theta_ne2001, theta_atnf, distance, flux_density, (l, b), (ra_err, dec_err), (pmra, pmdec), (pm_ra_err, pm_dec_err), distance_dm, theta_ne2001DM, tau_400, tau_400_ne2001]
        return b_name_dict, j_name_dict

    def get_julian_datetime(self):
        """
        Help from: https://stackoverflow.com/questions/31142181/calculating-julian-date-in-python
        Reference: http://scienceworld.wolfram.com/astronomy/JulianDate.html, equation ()

        Gets current JD time.

        date: datetime object of date in question

        :return: JD time of input datetime object

        Raises:
            TypeError : Incorrect parameter type if not datetime
            ValueError: Date out of range of equation (note between 1901 and 2099)
        """
        date = datetime.datetime.utcnow()

        # Ensures correct format
        if not isinstance(date, datetime.datetime):
            raise TypeError('Invalid type for parameter "date" - expecting datetime')
        elif date.year < 1901 or date.year > 2099:
            raise ValueError('Datetime must be between year 1801 and 2099')

        # Calculated JD using the reference
        julian_datetime = 367 * date.year - int((7 * (date.year + int((date.month + 9) / 12.0))) / 4.0) \
                          + int((275 * date.month) / 9.0) + date.day + 1721013.5 \
                          + (date.hour + date.minute / 60.0 + date.second / 3600) / 24.0

        return julian_datetime

    def correct_pm(self):  # ra and dec arrays in degrees!
        """
        Corrects RA and DEC arrays (in degrees) for proper motion
        """
        ra_new = []
        dec_new = []
        for ra, dec, pmra, pmdec, time_po, time_pe in zip(self.rajd, self.decjd, self.pm_ra, self.pm_dec, self.time_pos, self.time_per):
            now = self.get_julian_datetime() # get JD right now
            # MJD = JD - 2,400,000.5
            now = now - 2400000.5 # convert JD to MJD
            time = time_po
            # If no position time is given (either = 0.0), take period time or now as the time of observation
            if time_po < 1:
                time = time_pe
            elif time_pe < 1:
                time = now

            years = (now - time) / 365

            # Correct for proper motion
            ra = ra + (pmra * years)
            dec = dec + (pmdec * years)
            ra_new.append(ra)
            dec_new.append(dec)
        self.rajd = ra_new
        self.decjd = dec_new
        return

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

        Important note on DEC (+dd:mm:ss):
        DEC in degrees:arcminutes:arcseconds
        1 DEC minute = 1'
        1 DEC second = 1"

        :param typ:
        :return: error fixed and in degrees
        """
        xerrarr_fix = []
        for x, y_J, y_D, xerr in zip(self.raj, self.decj, self.decjd, xerrarr):
            if typ is 'ra':
                xerrarr_fix.append(self.get_err_ra(x, y_D, xerr))
            elif typ is 'dec':
                xerrarr_fix.append(self.get_err_dec(y_J, xerr))
            else:
                print("Invalid type input!")
                sys.exit()
        # np.set_printoptions(suppress=True)
        return np.round(np.asarray(xerrarr_fix, dtype=np.float64), 10)

    def get_err_ra(self, x, y, xerr):
        """
        Convert ra errors in seconds or minutes to degrees
        """
        if x.count(':') == 2:
            return xerr / 240. * np.cos(np.radians(y))  # (RA) seconds to degrees
        elif x.count(':') == 1:
            return xerr / 4. * np.cos(np.radians(y))  # (RA) minutes to degrees
        else:
            print("Invalid input!")

    def get_err_dec(self, x, xerr):
        """
        Convert dec errors in arcseconds or arcminutes to degrees
        """
        if x.count(':') == 2:
            return xerr / 3600.  # arcseconds to degrees
        elif x.count(':') == 1:
            return xerr / 60.  # arcminutes to degrees
        else:
            print("Invalid input!")


class PulsarLimitedList(RadioPulsarCatalog):
    def __init__(self, type, min_snr, max_local, max_theta, filename="psr_snr_list_normalised.txt", fix_automatic=True):
        """
        :param filename: text file output from CHIME pulsar which is used as input
        Columns are: PSR, S/N, sigma(S/N), integration_time, normalised_snr, RA, Dec,
        where PSR is the pulsar name
        :param min_snr: minimum SNR to be considered for list, e.g., 50
        :param max_local: maximum error in MILLIARCSECONDS in RA/DEC to be considered for list, e.g. 100 mas
        :param GBT: True if GBT SNRs wanted
        :param fix_automatic: True if automatically constraining pulsar list with all constraints
        """
        RadioPulsarCatalog.__init__(self)
        self.pulsar_replacements = {"J1327+3423": "J1326+33", "J2228+3041": "J2227+30",
                                    "J1629+43": "J1628+4406", "J2122+54": "J2123+5434",
                                    "J1954+43": "J1954+4357", "J2017+59": "J2017+5906",
                                    "J1748+59": "J1749+59", "J0742+4110": "J0740+41",
                                    "J0325+67": "J0325+6744", "J1941+02": "J1940+0239",
                                    "J1836+51": "J1836+5925", "J1710+49": "J1710+4923", "J1647+66": "J1647+6608",
                                    "J2001+42": "J2001+4258", "J1840-04": "J1840-0445", "J1920-09": "J1920-0950",
                                    "J1900-09": "J1900-0933"}

        self.df = self.get_data_csv(filename)
        self.max_local = max_local
        self.min_snr = min_snr
        self.b_name_dict_atnf, self.j_name_dict_atnf = self.get_names_pos_unc()
        self.type = type
        self.GBT = (self.type == "GBT")
        self.new_df = "empty" # where corrected data will be stored for constrained pulsars
        self.max_theta = max_theta

        self.combine_atnf_chime()
        if fix_automatic:
        # Pulsar name replacements where these is a mismatch between the CHIME/ATNF names
            self.filter_SNR()
            self.filter_correct_local()
            self.new_df["RA"] = self.new_df["RA"]/15  # Convert RA in degrees to RA hours
            self.filter_theta()
            self.save_csv()

    def get_data_csv(self, filename):
        """
        Reads in data from a text file delimited by whitespace
        :param filename: location of text file, assumining delimiter is whitespace
        :return: Pandas dataframe containing file contents
        """
        df = pandas.read_csv(filename, delim_whitespace=True)
        return df

    def helper_combine(self, helper_dict, ra_iftransit, name, name_full, snr):
        self.snr_match.append(snr)
        self.name_match.append(name_full)
        if "_" in name_full:
            self.ra_match.append(ra_iftransit * 15)
        else:
            self.ra_match.append(helper_dict[name][2][0])
        self.ra_err_match.append(helper_dict[name][0])
        self.dec_err_match.append(helper_dict[name][1])
        self.dec_match.append(helper_dict[name][2][1])
        self.theta_match_ne2001.append(helper_dict[name][3])
        self.theta_match_atnf.append(helper_dict[name][4])
        self.distances_match.append(helper_dict[name][5])
        self.flux_densities_match.append(helper_dict[name][6])
        self.l_match.append(helper_dict[name][7][0])
        self.b_match.append(helper_dict[name][7][1])
        self.ra_err_act_match.append(helper_dict[name][8][0])
        self.dec_err_act_match.append(helper_dict[name][8][1])
        self.pmra_match.append(helper_dict[name][9][0])
        self.pmdec_match.append(helper_dict[name][9][1])
        self.pmra_err_match.append(helper_dict[name][10][0])
        self.pmdec_err_match.append(helper_dict[name][10][1])
        self.distance_DM_match.append(helper_dict[name][11])
        self.theta_NE2001_DM_match.append(helper_dict[name][12])
        self.tau_400_match.append(helper_dict[name][13])
        self.tau_400_NE2001_match.append(helper_dict[name][14])


    def combine_atnf_chime(self):
        self.df["PSR"].replace(self.pulsar_replacements, inplace=True)

        match_names_full = self.df["PSR"]
        match_names = [n.split("_")[0] for n in self.df.PSR]

        match_snrs = self.df.normalised_snr
        ra_chime = self.df.RA

        self.ra_err_match = []
        self.dec_err_match = []
        self.theta_match_atnf = []
        self.theta_match_ne2001 = []
        self.theta_NE2001_DM_match = []
        self.flux_densities_match = []
        self.l_match = []
        self.b_match = []
        self.ra_err_act_match = []
        self.dec_err_act_match = []
        self.pmra_match = []
        self.pmdec_match = []
        self.pmra_err_match = []
        self.pmdec_err_match = []

        # Also replace RA/DEC with RA/DEC corrected for PM
        self.ra_match = []
        self.dec_match = []
        self.snr_match = []
        self.name_match = []
        j_name_dict = self.j_name_dict_atnf
        b_name_dict = self.b_name_dict_atnf
        self.distances_match = []
        self.distance_DM_match = []
        self.tau_400_match = []
        self.tau_400_NE2001_match = []

        count = 0
        for name_full, name, snr, ra in zip(match_names_full, match_names, match_snrs, ra_chime):
            if name in j_name_dict.keys():
                count += 1
                self.helper_combine(j_name_dict, ra, name, name_full, snr)
            elif name in b_name_dict.keys():
                count += 1
                self.helper_combine(b_name_dict, ra, name, name_full, snr)
            else:
                self.df[(self.df.PSR != name)]
        print("{} pulsars successfully found".format(count))
        data = {"names" : self.name_match, "RA": self.ra_match, "RA (degrees)": self.ra_match, "Dec": self.dec_match,
                "RA_err (mas)": self.ra_err_act_match, "Dec_err (mas)": self.dec_err_act_match,
                "PM RA (degrees)": self.pmra_match, "PM Dec (degrees)": self.pmdec_match,
                "PM RA_err (mas)": self.pmra_err_match, "PM Dec_err (mas)": self.pmdec_err_match,
                "ra_err_mas": self.ra_err_match, "dec_err_mas": self.dec_err_match,
                "l (degrees)": self.l_match, "b (degrees)": self.b_match,
                "snr": self.snr_match, "Mean Flux Density (400MHz from ATNF) (mJy)": self.flux_densities_match,
                "scat_disk_atnf": self.theta_match_atnf, "scat_disk_ne2001": self.theta_match_ne2001, "scat_disk_ne2001dm": self.theta_NE2001_DM_match,
                "ATNF Distance (kpc)": self.distances_match, "ATNF Distance only from DM (kpc)": self.distance_DM_match,
                "tau ATNF (400 MHz) (ms)": self.tau_400_match, "tau NE2001 (400 MHz) (ms)": self.tau_400_NE2001_match}
        self.new_df = pandas.DataFrame(data)
        return

    def filter_theta(self):
        self.new_df = self.new_df[self.new_df.scat_disk_ne2001 < self.max_theta] # filter in mas
        return

    def filter_SNR(self):
        """
        Filter CHIME pulsar list above min SNR
        :return:
        """
        # Constraining list based on GBT/CHIME SNR
        if self.GBT:
            self.new_df = self.new_df[self.get_gbt_snr_from_CHIME(self.new_df.snr, self.new_df.Dec) > self.min_snr]
        else:
            self.new_df = self.new_df[self.new_df.snr > self.min_snr]
        # manually omitting pulsars that can not be paired to ATNF or are new
        # self.df = self.df[(self.df.PSR != "J0534-13") & (self.df.PSR != "J0414+31") & (self.df.PSR != "J2022+2534") & (self.df.PSR != "J2044+28") & (
        #             self.df.PSR != "J2108+45") & (self.df.PSR != "J0406+3039") & (self.df.PSR != "J1239+32") & (self.df.PSR != "J1907+57") & (self.df.PSR != "J0854+54")
        #             & (self.df.PSR != "J1829+25") & (self.df.PSR != "J1904+33") & (self.df.PSR != "J0405+3347") & (self.df.PSR != "J0355+28") & (self.df.PSR != "J2207+40") & (self.df.PSR != "J2229+64") & (self.df.PSR != "J2311+67") & (self.df.PSR != "J1822+02") & (self.df.PSR != "J1928+28") & (self.df.PSR != "J0104+64") & (self.df.PSR != "J1759+5036") & (self.df.PSR != "J1901-04") & (self.df.PSR != "J1221-0633") & (self.df.PSR != "J1733-01") & (self.df.PSR != "J1629+33") & (self.df.PSR != "J1858-11") & (self.df.PSR != "J1742-0203") & (self.df.PSR != "J1317-0157") & (self.df.PSR != "J1045-0436") & (self.df.PSR != "J1304+12") & (self.df.PSR != "J2018-0414") & (self.df.PSR != "J1831-04")]
        return

    def filter_correct_local(self):
        """
        Filters CHIME pulsar list for precise enough localization and corrects RA/DEC for PM
        """
        new_df = self.new_df
        # Filtering out pulsars with errors greater than max local or that are equal to 0
        if not np.isinf(self.max_local):
            self.new_df = self.new_df[(np.sqrt(new_df.ra_err_mas**2 + new_df.dec_err_mas**2) < self.max_local) & ((new_df.ra_err_mas > 0) | (new_df.dec_err_mas > 0))]
        return

    def get_RAs(self, integer=True):
        # RETURNS RA IN Hour Angle
        if integer:
            self.new_df["RA"] = self.new_df["RA"].astype(int)
        return self.new_df["RA"]

    def get_SNRs(self):
        return np.asarray(self.new_df.snr)

    def get_loc_err(self):
        return np.asarray(np.sqrt(self.new_df.ra_err_mas**2 + self.new_df.dec_err_mas**2))

    def get_theta(self, atnf = False):
        if atnf:
            return np.asarray(self.new_df.scat_disk_atnf)
        else:
            return np.asarray(self.new_df.scat_disk_ne2001)

    def get_decs(self):
        return np.asarray(self.new_df.Dec)

    def get_transit_times(self):
        FWHM = 0.31  # deg.
        dec = self.get_decs()  # R2, deg.
        FWHM_dec = 0.31 / np.cos(np.deg2rad(dec))
        n_beams = 4
        t_mins = 1440. * (n_beams * FWHM_dec / 360.)  # transit time in mins
        return t_mins

    def save_csv(self):
        self.new_df.sort_values(by="RA", inplace=True)
        print(self.new_df["RA"].astype(int).value_counts())
        saving_df = self.new_df.rename(columns={"snr":"CHIME SNR (normalized to 10 min integration time)",
                                "RA":"RA (hours)", "Dec":"DEC (degrees)", "ra_err_mas":'"Total" RA_err (mas): combining both location and PM error',
                                "dec_err_mas":'"Total" DEC_err (mas): combining both location and PM error', "scat_disk_atnf":"Scat. Disk (ATNF) (mas)",
                                                "scat_disk_ne2001dm": "Scat. Disk (NE2001 from ATNF DM distance) (mas)",
                                                "scat_disk_ne2001":"Scat. Disk (NE2001 from ATNF Distance) (mas)"})
        # del saving_df["PSR"]
        saving_df.to_csv("pulsars_snr_{}_loc_{}_scat_{}_{}.csv".format(self.min_snr, self.max_local, self.max_theta, self.type), index=False)

    def get_gbt_snr_from_CHIME(self, normalized_snr, dec):
        """
        Scale normalized SNR from CHIME to scaled SNR at GBT latitude
        """
        gbt_latitude = 38.4330
        chime_latitude = 49.3208
        gbt_snr = normalized_snr * (np.cos(np.degrees(dec - gbt_latitude)) / np.cos(np.degrees(dec - chime_latitude)))
        return gbt_snr

    def plotter(self, other_cat, name, bin=48, plot_name=None):
        ra_1 = self.get_RAs()
        ra_2 = other_cat.get_RAs()

        plt.xlim([0, 23])
        plt.xticks(ticks=np.linspace(0, 24, bin/2 + 1))
        ra_1.plot.hist(bins=bin, color="blue", edgecolor = 'black', label="{} S/N > {}, $loc_{{err}}$ < {} mas".format(self.type, self.min_snr, self.max_local))
        ra_2.plot.hist(bins=bin, color="orange", edgecolor='black', label="{} S/N > {}, $loc_{{err}}$ < {} mas".format(other_cat.type, other_cat.min_snr, other_cat.max_local))
        plt.xlabel("RA (hours)")
        plt.ylabel("# of Pulsars")
        plt.legend()
        if plot_name is not None:
            plt.title(plot_name)
        plt.tight_layout()
        plt.savefig("{}.png".format(name), dpi=600)
        plt.savefig("{}.pdf".format(name))
        plt.close()

    def plotter_alone(self, name, bin=48, loc=True):
        plt.xlim([0, 23])
        plt.xticks(ticks=np.linspace(0, 24, bin/2 + 1))
        if bin == 24:
            ra_1 = self.get_RAs(integer=True)
        else:
            ra_1 = self.get_RAs()
        print("Total number of pulsars: {} for {} S/N > {}, $loc_{{err}}$ < {} mas, $theta_{{scat}}$ < {} mas".format(len(ra_1), self.type, self.min_snr, self.max_local, self.max_theta))
        if loc:
            print()
            ra_1.plot.hist(bins=bin, color ="blue", edgecolor = 'black', label="{} S/N > {}, $loc_{{err}}$ < {} mas, $theta_{{scat}}$ < {} mas".format(self.type, self.min_snr, self.max_local, self.max_theta))
        else:
            ra_1.plot.hist(bins=bin, color ="blue", edgecolor = 'black', label="{} S/N > {}, $theta_{{scat}}$ < {} mas".format(self.type, self.min_snr, self.max_local, self.max_theta))
        plt.xlabel("RA (hours)")
        plt.ylabel("# of Pulsars")
        plt.legend()
        plt.tight_layout()
        plt.savefig("{}.png".format(name), dpi=600)
        plt.savefig("{}.pdf".format(name))
        plt.close()

    def snr_ra_color_plotter_alone(self, name, color="theta"):
        plt.close()
        fig = plt.figure(figsize=(10, 5))
        ax = plt.gca()
        ra_1 = self.get_RAs()
        dec_1 = self.get_decs()
        snr_1 = self.get_SNRs()
        locs_1 = self.get_loc_err()
        thetas_1 = self.get_theta(atnf=False)
        ts_1 = self.get_transit_times()
        names = np.asarray(self.new_df["names"])
        ts_1 = np.asarray(ts_1)/60
        quick = np.argmax(np.asarray(snr_1), axis=0)
        ts_1[quick] = 3
        quicker = np.argmax(np.asarray(thetas_1), axis=0)
        print("max: {}".format(names[quicker]))
        cm = plt.cm.get_cmap('viridis')
        plt.xlim([0, 23])
        plt.xticks(ticks=np.linspace(0, 24, 25))
        if color == "loc":
            pulses = ax.scatter(dec_1, ra_1, c=locs_1, s=25, cmap=cm)
            cbar1 = fig.colorbar(pulses)
            cbar1.set_label("$loc_{{err}}$ (mas)", labelpad=20)
            ax.errorbar(dec_1, ra_1, yerr=ts_1, linestyle="None", c='k')
        if color == "theta":
            pulses = ax.scatter(ra_1, snr_1, c=thetas_1, s=25, cmap=cm)
            cbar1 = fig.colorbar(pulses)
            ax.errorbar(dec_1, ra_1, yerr=ts_1, linestyle="None", c='k')
            cbar1.set_label("Scattering Disk (mas)", labelpad=20)
        # plt.hlines(50, np.min(dec_1), np.max(dec_1))
        # plt.text(70, 55, 'SNR = 50')
        plt.text(55, 10100, names[quick])

        ax.set_yscale("log")
        ax.set_ylabel("SNR")
        ax.set_xlabel("RA (hours)")
        ax.grid(True)

        plt.savefig("{}.png".format(name), dpi=600)
        plt.savefig("{}.pdf".format(name))
        plt.close()


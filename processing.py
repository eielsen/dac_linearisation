from utils.inl_processing import gen_physcal_lut, plot_inl
from utils.quantiser_configurations import qs
from utils.results import JSON_results
from LM.lin_method_util import lm, dm


def update_JSON():
    JR = JSON_results()

    DC = qs.w_10bit_ARTI
    DM = dm.SPICE
    LM = lm.SHPD
    fs = 226719135.13513514400
    fc = 100000
    f0 = 1000
    Ncyc = 7
    ENOB = 7.19

    JR.add( DC=DC,      # DAC configuration
            DM=DM,      # STATIC, SPICE
            LM=LM,      # Linearisation method
            fs=fs,      # SAMPLING FREQUENCY
            fc=fc,      # FILTER CORNER FREQUENCY
            f0=f0,      # CARRIER / FUNDAMENTAL
            Ncyc=Ncyc,  # Periods / cycles of carrier
            ENOB=ENOB)  # Resulting ENOB

    JR.print(DC, DM, LM)
    JR.save()
    JR.save_to_html()

if __name__ == '__main__':
    # PLOT INL BY UNVOMMENTING THE DESIRED PART OF THE CODE
    # plot_inl(QConfig=qws.w_16bit_6t_ARTI, Ch_sel=0)


    # This part of the script is used to update the JSON that contains the ENOB results. The JSON file may also be updated manually by hand.
    # Uncomment this part if you want to use it.
    update_JSON()
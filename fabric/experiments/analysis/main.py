from AD import main as AD_main
from RCD import main as RCD_main


def main():
    """
    Main function to analyse the energy data. Make sure that the data-collection
    is done before running this file. This code assumes that a file for the energy data is present.
    """
    # Perform Anomaly Detection (AD)
    AD_main()

    # Perform Root Cause Analysis (RCA)
    RCD_main()

    print("Finished data analysis")

if __name__ == '__main__':
    main()

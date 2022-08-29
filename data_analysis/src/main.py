from utils        import *
from gas_analysis import *

debug_mode  = True

def main():
    debug("\n----------------------------------------------------------------------------", "cyan", debug_mode)
    debug("----------------------- INICIALIZACAO DO EXPERIMENTO -----------------------", "cyan", debug_mode)
    debug("----------------------------------------------------------------------------\n", "cyan", debug_mode)

    gas_analysis = Gas_Analysis(debug_mode)
    gas_analysis.remove_unused_columns()

    print(gas_analysis.gas_df.head())

if __name__ == "__main__":
    main()
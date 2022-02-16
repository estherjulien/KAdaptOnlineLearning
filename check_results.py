import pandas as pd

metadata = pd.DataFrame()
extra_info = pd.DataFrame(columns=["accuracy", "type_1", "type_2", "sum_errors", 0, 1])
for I in [100, 200, 500, 1000]:
    for K in [2, 3, 4]:
        for ct in [None, 30, 70]:
            if ct:
                new = pd.read_pickle(f"CapitalBudgetingResults/ModelResults/Models/Info/rf_class_info_cb_p5_N10_K{K}_I{I}_ct{ct}_all.pickle")
            else:
                new = pd.read_pickle(f"CapitalBudgetingResults/ModelResults/Models/Info/rf_class_info_cb_p5_N10_K{K}_I{I}_all.pickle")
            if len(metadata) == 0:
                metadata = pd.DataFrame(columns=new.index)
            metadata.loc[f"K{K}_I{I}_ct{ct}"] = new
            extra_info.loc[f"K{K}_I{I}_ct{ct}"] = [new["accuracy"], new["type_1"], new["type_2"], new["type_1"] + new["type_2"], new["0"], new["1"]]
import os
import numpy as np
import pandas as pd

from .util import load_pickle_compat

# current file path
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))


def load_data(
    data_name: str,
    onehot_encode: bool = False,
) -> pd.DataFrame:
    if data_name == "METABRIC":
        return make_metabric()
    elif data_name == "NACD":
        return make_nacd()
    elif data_name == "GBSG2":
        return make_gbsg2()
    elif data_name == "NWTCO":
        return make_nwtco()
    elif data_name == "NPC":
        return make_npc()
    elif data_name == "AIDS":
        return make_aids()
    elif data_name == "HFCR":
        return make_heart_failure()
    elif data_name == "leukemia":
        return make_leukemia()
    elif data_name == "Rossi":
        return make_rossi()
    elif data_name == "COVID":
        return make_covid()
    # below is datasets with missing values
    elif data_name == "SUPPORT":
        return make_support(onehot_encode=onehot_encode)
    elif data_name == "FLCHAIN":
        return make_flchain(onehot_encode=onehot_encode)
    elif data_name == "PBC":
        return make_pbc()
    elif data_name == "GBM":
        return make_gbm(onehot_encode=onehot_encode)
    elif data_name == "WPBC":
        return make_wpbc()
    elif data_name == "BMT":
        return make_bmt(onehot_encode=onehot_encode)
    else:
        raise ValueError("Dataset name not recognized.")


def maybe_onehot_encode(
    df: pd.DataFrame,
    categorical_cols: list[str],
    onehot_encode: bool,
) -> pd.DataFrame:
    if onehot_encode:
        return pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    df = df.copy()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def make_support(onehot_encode: bool = False) -> pd.DataFrame:
    """Downloads and preprocesses the SUPPORT dataset from [1]_.

    The missing values are filled using either the recommended
    standard values, the mean (for continuous variables) or the mode
    (for categorical variables).
    Refer to the dataset description at
    https://hbiostat.org/data/repo/supportdesc for more information.
    Download from https://hbiostat.org/data/repo/support2csv.zip
    Returns
    -------
    pd.DataFrame
        Processed covariates for one patient in each row.
    list[str]
        List of columns to standardize.

    References
    ----------
    [1] W. A. Knaus et al., The SUPPORT Prognostic Model: Objective Estimates of Survival
    for Seriously Ill Hospitalized Adults, Ann Intern Med, vol. 122, no. 3, p. 191, Feb. 1995.
    """
    # url = "https://hbiostat.org/data/repo/support2csv.zip"

    # Remove other target columns and other model predictions
    cols_to_drop = [
        "hospdead",
        "slos",
        "charges",
        "totcst",
        "totmcst",
        "avtisst",
        "sfdm2",
        "adlp",
        "adls",
        "dzgroup",  # "adlp", "adls", and "dzgroup" were used in other preprocessing steps,
        # see https://github.com/autonlab/auton-survival/blob/master/auton_survival/datasets.py
        "sps",
        "aps",
        "surv2m",
        "surv6m",
        "prg2m",
        "prg6m",
        "dnr",
        "dnrday",
        "hday",
    ]

    # `death` is the overall survival event indicator
    # `d.time` is the time to death from any cause or censoring
    # df = pd.read_csv(url).drop(cols_to_drop, axis=1).rename(columns={"d.time": "time", "death": "event"})
    df = (
        pd.read_csv(f"{CURRENT_PATH}/support2.csv")
        .drop(cols_to_drop, axis=1)
        .rename(columns={"d.time": "time", "death": "event"})
    )
    df["event"] = df["event"].astype(int)

    df["ca"] = (df["ca"] == "metastatic").astype(int)

    # use recommended default values from official dataset description ()
    # or mean (for continuous variables)/mode (for categorical variables) if not given
    fill_vals = {
        "alb": 3.5,
        "pafi": 333.3,
        "bili": 1.01,
        "crea": 1.01,
        "bun": 6.51,
        "wblc": 9,
        "urine": 2502,
        # "edu": df["edu"].mean(),
        # "ph": df["ph"].mean(),
        # "glucose": df["glucose"].mean(),
        # "scoma": df["scoma"].mean(),
        # "meanbp": df["meanbp"].mean(),
        # "hrt": df["hrt"].mean(),
        # "resp": df["resp"].mean(),
        # "temp": df["temp"].mean(),
        # "sod": df["sod"].mean(),
        # "income": df["income"].mode()[0],
        # "race": df["race"].mode()[0],
    }
    df = df.fillna(fill_vals)

    with pd.option_context("future.no_silent_downcasting", True):
        df.sex = df.sex.replace({"male": 1, "female": 0}).infer_objects()
        df.income = df.income.replace({"under $11k": 0, "$11-$25k": 1, "$25-$50k": 2, ">$50k": 3}).infer_objects()

    # one-hot encode categorical variables
    onehot_cols = ["dzclass", "race"]
    df = maybe_onehot_encode(df, onehot_cols, onehot_encode=onehot_encode)
    if onehot_encode:
        df = df.rename(columns={"dzclass_COPD/CHF/Cirrhosis": "dzclass_COPD"})

    df.reset_index(drop=True, inplace=True)
    return df


def make_nacd() -> pd.DataFrame:
    cols_to_drop = ["PERFORMANCE_STATUS", "STAGE_NUMERICAL", "AGE65"]
    df = pd.read_csv(f"{CURRENT_PATH}/NACD_Full.csv").drop(cols_to_drop, axis=1).rename(columns={"delta": "event"})

    df = df.drop(df[df["time"] <= 0].index)  # remove patients with negative or zero survival time
    df.reset_index(drop=True, inplace=True)
    return df


def make_metabric() -> pd.DataFrame:
    df = pd.read_csv(f"{CURRENT_PATH}/metabric.csv").rename(columns={"delta": "event", "duration": "time"})
    return df


def make_flchain(onehot_encode: bool = False) -> pd.DataFrame:
    # flchain dataset: relationship between serum free light chain (FLC) and mortality
    # see: https://vincentarelbundock.github.io/Rdatasets/doc/survival/flchain.html
    cols_to_drop = ["chapter"]  # only dead patients has chapter information
    df = (
        pd.read_csv(f"{CURRENT_PATH}/flchain.csv")
        .drop(cols_to_drop, axis=1)
        .rename(columns={"futime": "time", "death": "event"})
    )
    df = df.drop(df[df["time"] <= 0].index)  # remove patients with negative or zero survival time
    df.reset_index(drop=True, inplace=True)
    with pd.option_context("future.no_silent_downcasting", True):
        df.sex = df.sex.replace({"M": 1, "F": 0}).infer_objects()
    # processing see: https://github.com/paidamoyo/adversarial_time_to_event/blob/master/data/flchain/flchain_data.py
    # data = data.fillna({"creatinine": data["creatinine"].median()})
    onehot_cols = ["sample.yr", "flc.grp"]
    df = maybe_onehot_encode(df, onehot_cols, onehot_encode=onehot_encode)
    return df


def make_nwtco() -> pd.DataFrame:
    """
    Tumor histology predicts survival. Downloaded and preprocessed from [1]_.

    Check the data description at https://vincentarelbundock.github.io/Rdatasets/doc/survival/nwtco.html
    Download from https://vincentarelbundock.github.io/Rdatasets/csv/survival/nwtco.csv

    References
    ----------
    [1] NE Breslow and N Chatterjee (1999), Design and analysis of two-phase studies with binary outcome applied to
    Wilms tumour prognosis. Applied Statistics 48, 457–68.
    """
    cols_to_drop = ["rownames", "seqno"]
    df = (
        pd.read_csv(f"{CURRENT_PATH}/nwtco.csv")
        .drop(cols_to_drop, axis=1)
        .rename(columns={"rel": "event", "edrel": "time"})
    )
    return df


def make_gbsg2() -> pd.DataFrame:
    """
    German Breast Cancer Study Group (GBSG)

    This dataset is downloaded from `survival` package in R.
    The data description can be found at https://rdrr.io/cran/survival/man/gbsg.html
    """
    cols_to_drop = ["pid"]
    df = (
        pd.read_csv(f"{CURRENT_PATH}/GBSG.csv")
        .drop(cols_to_drop, axis=1)
        .rename(columns={"status": "event", "rfstime": "time"})
    )
    return df


def make_gbm(onehot_encode: bool = False) -> pd.DataFrame:
    df = pd.read_csv(f"{CURRENT_PATH}/GBM.clin.merged.picked.csv").rename(columns={"delta": "event"})
    df.drop(columns=["Composite Element REF", "tumor_tissue_site"], inplace=True)  # Columns with only one value
    df = df[df.time.notna()]  # Unknown censor/event time
    df = df.drop(df[df["time"] <= 0].index)  # remove patients with negative or zero survival time
    df.reset_index(drop=True, inplace=True)

    # Preprocess and fill missing values
    with pd.option_context("future.no_silent_downcasting", True):
        df.gender = df.gender.replace({"male": 1, "female": 0}).infer_objects()
        df.radiation_therapy = df.radiation_therapy.replace({"yes": 1, "no": 0}).infer_objects()
        df.ethnicity = df.ethnicity.replace({"not hispanic or latino": 0, "hispanic or latino": 1}).infer_objects()
    # one-hot encode categorical variables
    onehot_cols = ["histological_type", "race"]
    df = maybe_onehot_encode(df, onehot_cols, onehot_encode=onehot_encode)
    # fill_vals = {
    #     "radiation_therapy": data["radiation_therapy"].median(),
    #     "karnofsky_performance_score": data["karnofsky_performance_score"].median(),
    #     "ethnicity": data["ethnicity"].median()
    # }
    # data = data.fillna(fill_vals)
    df.columns = df.columns.str.replace(" ", "_")
    return df


def make_npc() -> pd.DataFrame:
    """
    nasopharyngeal carcinoma (NPC) prognostic dataset collected from
    Sun Yat-sen University Cancer Center, Guangzhou, China.

    End time is disease-free (Progression-free) survival time (PFSmonths), which was calculated
    from the date of diagnosis to the date of the first relapse at any site, death from any cause,
    or the date of the last follow-up visit.

    The original dataset are split into fixed training and testing set.
    The training set contains 4,630 consecutive NPC patients between 2007.01-2009.12.
    The testing set contains 1,819 NPC patients between 2011.01-2012.06.
    There we combine the training and testing set together.

    More details can be found in [1]_. The dataset is downloaded from [2]_.
    [1] Tang LQ, Li CF, Li J, et al. Establishment and Validation of Prognostic Nomograms
    for Endemic Nasopharyngeal Carcinoma. J Natl Cancer Inst. 2016. 108(1)
    [2] https://github.com/sysucc-ailab/RankDeepSurv/tree/master
    """
    df_train = pd.read_csv(f"{CURRENT_PATH}/npc_train.csv")
    df_test = pd.read_csv(f"{CURRENT_PATH}/npc_test.csv")
    df = pd.concat([df_train, df_test]).reset_index(drop=True)

    df.rename(
        columns={
            "PFSmonths": "time",
            "outcome": "event",
            "TUICC": "T_stage",
            "NUICC": "N_stage",
        },
        inplace=True,
    )
    return df


def make_aids() -> pd.DataFrame:
    """
    Preprocess the AIDS Clinical Trials Group Study 175 dataset.

    Data link: https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175
    Paper: https://www.nejm.org/doi/pdf/10.1056/NEJM199610103351501
    """
    aids_clinical_trials_group_study_175 = load_pickle_compat(
        f"{CURRENT_PATH}/aids_clinical_trials_group_study_175.pkl"
    )

    X = aids_clinical_trials_group_study_175.data.features
    y = aids_clinical_trials_group_study_175.data.targets.rename(columns={"cid": "event"})
    df = pd.concat([X, 1 - y], axis=1)
    return df


def make_pbc() -> pd.DataFrame:
    """
    Preprocess the Cirrhosis Patient Survival Prediction dataset.

    Link: https://archive.ics.uci.edu/dataset/878/cirrhosis+patient+survival+prediction+dataset-1
    Paper: https://pubmed.ncbi.nlm.nih.gov/2737595/
    """
    cirrhosis = load_pickle_compat(f"{CURRENT_PATH}/cirrhosis.pkl")

    cols_to_drop = ["ID"]
    df = cirrhosis.data.original.drop(cols_to_drop, axis=1).rename(columns={"Status": "event", "N_Days": "time"})
    with pd.option_context("future.no_silent_downcasting", True):
        df = df.replace({"NaNN": np.nan}).infer_objects()
        df.event = df.event.replace({"C": 0, "CL": 0, "D": 1}).infer_objects()
        df.Drug = df.Drug.replace({"D-penicillamine": 0, "Placebo": 1}).infer_objects()
        df.Sex = df.Sex.replace({"M": 1, "F": 0}).infer_objects()
        df.Ascites = df.Ascites.replace({"N": 0, "Y": 1}).infer_objects()
        df.Hepatomegaly = df.Hepatomegaly.replace({"N": 0, "Y": 1}).infer_objects()
        df.Spiders = df.Spiders.replace({"N": 0, "Y": 1}).infer_objects()
        df.Edema = df.Edema.replace({"N": 0, "Y": 1, "S": 0.5}).infer_objects()
    df.Cholesterol = pd.to_numeric(df.Cholesterol, errors="coerce")
    df.Copper = pd.to_numeric(df.Copper, errors="coerce")
    df.Tryglicerides = pd.to_numeric(df.Tryglicerides, errors="coerce")
    df.Platelets = pd.to_numeric(df.Platelets, errors="coerce")

    # fill_vals = {
    #     "Drug": data.Drug.mode()[0],
    #     "Ascites": data.Ascites.mode()[0],
    #     "Hepatomegaly": data.Hepatomegaly.mode()[0],
    #     "Spiders": data.Spiders.mode()[0],
    #     "Cholesterol": data.Cholesterol.mean(),
    #     "Copper": data.Copper.mean(),
    #     "Alk_Phos": data.Alk_Phos.mean(),
    #     "SGOT": data.SGOT.mean(),
    #     "Tryglicerides": data.Tryglicerides.mean(),
    #     "Platelets": data.Platelets.mean(),
    #     "Prothrombin": data.Prothrombin.mean(),
    #     "Stage": data.Stage.mode()[0],
    # }
    #
    # data = data.fillna(fill_vals)
    df.reset_index(drop=True, inplace=True)
    return df


def make_heart_failure() -> pd.DataFrame:
    """
    Preprocess the Heart Failure Prediction dataset.

    Link: https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records
    Paper: https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5
    """
    heart_failure = load_pickle_compat(f"{CURRENT_PATH}/heart_failure.pkl")

    X = heart_failure.data.features
    y = heart_failure.data.targets
    df = pd.concat([X, y], axis=1).rename(columns={"death_event": "event"})
    return df


def make_wpbc() -> pd.DataFrame:
    """
    Preprocess the Wisconsin Prognostic Breast Cancer dataset.

    Event is recurrence of breast cancer.

    Link: https://archive.ics.uci.edu/dataset/16/breast+cancer+wisconsin+prognostic
    """
    wpbc = load_pickle_compat(f"{CURRENT_PATH}/wpbc.pkl")

    cols_to_drop = ["ID"]
    df = wpbc.data.original.drop(cols_to_drop, axis=1).rename(columns={"Outcome": "event", "Time": "time"})
    with pd.option_context("future.no_silent_downcasting", True):
        # N: no recurrence, R: recurrence
        df.event = df.event.replace({"N": 0, "R": 1}).infer_objects()
    # # fill missing values with the median for lymph_node_status
    # data = data.fillna({"lymph_node_status": data["lymph_node_status"].median()})
    return df


def make_bmt(onehot_encode: bool = False) -> pd.DataFrame:
    """
    Preprocess for the Bone Marrow Transplant dataset.

    Link: https://archive.ics.uci.edu/dataset/565/bone+marrow+transplant+children
    Paper: https://www.astctjournal.org/article/S1083-8791(10)00148-5/fulltext
    """
    bmt = load_pickle_compat(f"{CURRENT_PATH}/bmt.pkl")

    cols_to_drop = [
        "Donorage35",
        "Recipientage10",
        "Recipientageint",
    ]
    df = bmt.data.original.drop(cols_to_drop, axis=1).rename(
        columns={"survival_status": "event", "survival_time": "time"}
    )

    blood_type = {0: "O", 1: "A", -1: "B", 3: "AB"}
    with pd.option_context("future.no_silent_downcasting", True):
        df.DonorABO = df.DonorABO.replace(blood_type).infer_objects()
        df.RecipientABO = df.RecipientABO.replace(blood_type).infer_objects()

    cat_cols = ["DonorABO", "RecipientABO", "Disease"]
    df = maybe_onehot_encode(df, cat_cols, onehot_encode=onehot_encode)

    # # fill missing values with the median
    # data = data.fillna(data.median())
    return df


def make_leukemia() -> pd.DataFrame:
    """
    Leukemia dataset.
    From lifelines package.
    which is originally from
    http://web1.sph.emory.edu/dkleinb/allDatasets/surv2datasets/anderson.dat
    """
    leukemia = pd.read_csv(f"{CURRENT_PATH}/anderson.csv", sep=" ").rename(columns={"status": "event", "t": "time"})
    return leukemia


def make_rossi() -> pd.DataFrame:
    """
    The Rossi dataset pertain to 432 convicts who were released from Maryland state prisons in the 1970s
    and who were followed up for one year after release.
    Half the released convicts were assigned at random to an experimental treatment in which
    they were given financial aid; half did not receive aid.

    Rossi, P.H., R.A. Berk, and K.J. Lenihan (1980). Money, Work, and Crime: Some Experimental Results.
    New York: Academic Press.

    John Fox, Marilia Sa Carvalho (2012). The RcmdrPlugin.survival Package: Extending the R Commander
    Interface to Survival Analysis. Journal of Statistical Software, 49(7), 1-32.
    """
    rossi = pd.read_csv(f"{CURRENT_PATH}/rossi.csv").rename(columns={"week": "time", "arrest": "event"})
    return rossi


def make_covid():
    """
    Load the COVID-19 dataset.

    This dataset aims to investigate the discharge time of COVID-19 patients in Asian.

    Data link: https://github.com/kuan0911/ISDEvaluation-covid/blob/master/Data/covid/asian_discharge_exp3.csv
    Paper: https://www.nature.com/articles/s41598-022-08601-6#MOESM1
    """
    covid = pd.read_csv(f"{CURRENT_PATH}/asian_discharge_exp3.csv")

    covid = covid.drop(covid[covid["time"] <= 0].index)  # remove patients with negative or zero survival time
    covid.reset_index(drop=True, inplace=True)

    # change population density to float
    covid["population_density_city"] = covid["population_density_city"].str.replace(",", "").astype(float)
    return covid


if __name__ == "__main__":
    # test loading each dataset
    dataset_names = [
        "SUPPORT",
        "METABRIC",
        "NACD",
        "FLCHAIN",
        "GBSG2",
        "NWTCO",
        "PBC",
        "GBM",
        "NPC",
        "AIDS",
        "HFCR",
        "WPBC",
        "BMT",
        "leukemia",
        "Rossi",
        "COVID",
    ]
    # get missing rate for each dataset
    for dataset_name in dataset_names:
        df = load_data(dataset_name)
        missing_rate = df.isnull().mean().mean()
        print(f"{dataset_name}: {missing_rate:.2%} missing values")
        has_missing = df.isnull().values.any()
        print(f"{dataset_name}: {'has missing values' if has_missing else 'no missing values'}")

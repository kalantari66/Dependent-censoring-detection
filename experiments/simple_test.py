from cmi import detect_dependent_censoring
from data import dgp


def main():
    df = dgp(
        kind="copula",
        n_subjects=500,
        n_features=3,
        copula="clayton",
        theta=4.0,
        gamma=0.5,
        seed=1
    )

    p_global = detect_dependent_censoring(
        df,
        quantiles=[0.3, 0.5, 0.7, 0.9],
        B=200,
        seed=123,
        min_stratum_size=30,
        variance_threshold=1e-3,
        t_col="time",
        e_col="event",
    )

    print("Global p-value:", p_global)


if __name__ == "__main__":
    main()
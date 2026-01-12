from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt


def tv_download(tickers: list[tuple[str, str, str]], interval: str, n_bars) -> pd.DataFrame: #tickers in format: symbol, exchange, name (optional)
    tv = TvDatafeed()
    returns_df = pd.DataFrame()

    for i in range(len(tickers)):
        df: pd.DataFrame = tv.get_hist(
            symbol=tickers[i][0],
            exchange=tickers[i][1],
            interval=interval,
            n_bars=n_bars,
            fut_contract=1
            )
        
        returns_df[f"{tickers[i][0]}"] = df["close"].values
        returns_df.set_index(df.index.values, inplace=True)

    return returns_df.dropna()

def tv_import(path: str) -> pd.DataFrame:
    df = pd.read_csv(f"./data/{path}")
    df.set_index(pd.to_datetime(df.iloc[:, 0]), inplace=True)
    df = df.drop(df.columns[0], axis=1)
    return df

def get_pnl(price_df: pd.DataFrame, weights: list[float], investment) -> pd.Series:
    simple_returns = price_df.pct_change().dropna() * investment
    return simple_returns @ weights

def get_risk_hist(pnl: pd.Series, days, ci):
    range_pnl = pnl.rolling(window=days).sum() 
    range_pnl = range_pnl.dropna()

    var = -np.quantile(range_pnl, 1 - ci)
    es = -range_pnl[range_pnl <= -var].mean()

    return float(var), float(es), pd.Series(range_pnl)

def get_risk_t(df: pd.DataFrame, days, ci, investment, weights, num_sims = 100_000, nu = 4, volatility_scaler = 1):
    # Generate multi-day PnL simulations:
    #   1. Draw standard normals for each asset and day
    #   2. Apply Cholesky to introduce correlations
    #   3. Scale by chi-square for t-distribution tails
    #   4. Compute weighted portfolio daily PnL and sum over days
    # Compute VaR and ES

    pnl = df.pct_change().dropna() * investment

    mu = pnl.mean().values
    Sigma = pnl.cov().values
    n_assets = len(mu)
    
    shape = Sigma * (nu - 2) / nu
    shape = shape * volatility_scaler ** 2
    L = np.linalg.cholesky(shape)

    # Generating all normal draws (our numerator)
    rng = np.random.default_rng(None) #Creating the random seed
    Z = rng.standard_normal(size=(num_sims, days, n_assets))
    daily_normal = Z @ L.T #shape: (num_sims, days, n_assets)

    # Generating 1 chi-square per simulation and scaling it (our denominator)
    U = rng.chisquare(df=nu, size=num_sims) #shape: (num_sims,)
    scale = np.sqrt(U / nu)
    scale = scale[:, None, None] #shape: (num_sims, 1, 1)

    # Combining them into multivariate Student t ( mu + Z/(sqrt(U/nu)) )
    daily_pnl = mu + daily_normal / scale #shape: (num_sims, days, n_assets)
    
    portfolio_daily = np.einsum('ijk,k->ij', daily_pnl, weights) # daily_pnl @ weights -> (num_sims, days)
    total_pnl = portfolio_daily.sum(axis=1) #shape: (num_sims,)


    var = -np.quantile(total_pnl, 1 - ci)
    es = -total_pnl[total_pnl <= -var].mean()

    return float(var), float(es), pd.Series(total_pnl)



def create_hist_plot(data: pd.Series, title: str, lines: tuple[float], lines_labels: tuple[str], line_colors: tuple[str]):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, alpha=0.75, color="blue", edgecolor="black")

    if lines != None:
        for line, label, color in zip(lines, lines_labels, line_colors):
            plt.axvline(-line, linestyle = "--", label = f"{label}: ${round(line):,}", color = color)

    plt.title(title)
    plt.xlabel("Returns")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def pnl_chart(pnl, capital, title):
    equity = capital + pnl.cumsum()
    plt.figure()
    plt.plot(equity.index, equity)
    plt.axhline(0)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.tight_layout()
    plt.show()

def scenario_hist(df, weights, investment, start_date: dt.datetime, end_date: dt.datetime, title):
    days=5
    ci=0.95
    nu = 5

    scenario_df = df[df.index.to_series().between(start_date, end_date)]

    scenario_pnl = get_pnl(scenario_df, weights, investment)



    VaR_roll, ES_roll, total_pnl_roll = get_risk_hist(scenario_pnl, days=days, ci=ci)

    VaR_scale, ES_scale, total_pnl_scale = get_risk_hist(scenario_pnl, days=1, ci=ci)

    VaR_scale = VaR_scale * np.sqrt(days)
    ES_scale = ES_scale * np.sqrt(days)


    total_loss_gain = scenario_pnl.sum()

    pnl_chart(scenario_pnl, 250_000_000, title=f"{title}. Portfolio Equity")
    VaR_mc, ES_mc, total_pnl = get_risk_t(scenario_df,
        days=days,
        ci=ci,
        investment=investment,
        weights=weights,
        nu=nu
    )


    create_hist_plot(total_pnl, title=f"{title} Monte Carlo Simulation over {days} days", lines=(VaR_mc, ES_mc), lines_labels=("VaR", "Expected Shortfall"), line_colors=("yellow", "red"))


    print(f"""
        ---{title}---\n
            Parameters:\n
                Start date: {start_date.strftime('%Y-%m-%d')}\n
                End date: {end_date.strftime('%Y-%m-%d')}\n
            Results:\n
                Historical:\n\n
                5-day VaR (rolling): ${VaR_roll:,.0f}\n
                5-day ES (rolling): ${ES_roll:,.0f}\n
                5-day VaR (scaled): ${VaR_scale:,.0f}\n
                5-day ES (scaled): ${ES_scale:,.0f}\n
                Total gain/loss: ${total_loss_gain:,.0f}\n\n
                Monte Carlo:\n\n
                5-day VaR (Student t): ${VaR_mc:,.0f}\n
                5-day ES (Student t): ${ES_mc:,.0f}
        """)


def main():
    tickers_list = [
        ("CL", "NYMEX", "WTI Crude Oil"), 
        ("RB", "NYMEX", "RBOB Gasoline"),
        ("NG", "NYMEX", "Natural Gas"),
        ("GC", "COMEX", "Gold"),
        ("SI", "COMEX", "Silver"),
        ("PL", "NYMEX", "Platinum"),
        ("ALI", "COMEX", "Aluminum (NOT Aluminium)"),
        ("HG", "COMEX", "Copper"),
        ("LE", "CME", "Live Cattle"),
        ("ZS", "CBOT", "Soybeans")
    ]

    df = tv_import("tv_data.csv")

#    weights = np.array([1/len(tickers_list)]*len(tickers_list)) 

    weights = [-0.1, 0.1, -0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]
    investment = 5_000_000_000 * 0.95

#   Scenario 1. Oil Price Crash 2020
    start_date = dt.datetime(2020, 2, 20)
    end_date = dt.datetime(2020, 4, 30)
    
    scenario_hist(df, weights, investment, start_date, end_date, "Stress Test Scenario 1. Oil Price Crash 2020")
    

#   Scenario 2. Energy Crisis 2022

    start_date = dt.datetime(2022, 2, 24)
    end_date = dt.datetime(2022, 10, 31)
    
    scenario_hist(df, weights, investment, start_date, end_date, "Stress Test Scenario 2. Energy Crisis 2022")

#   Scenario 3. Increased Volatility
    ci = 0.95
    nu = 5
    volatility_coefficient = 2

    VaR_5, ES_5, total_pnl_5 = get_risk_t(
        df,
        days=5,
        ci=ci,
        investment=investment,
        weights=weights,
        nu=nu,
        volatility_scaler=volatility_coefficient
        )

    create_hist_plot(total_pnl_5,
        title=f"Stress Test 3. Monte Carlo Simulation over {5} days",
        lines=(VaR_5, ES_5),
        lines_labels=("VaR", "Expected Shortfall"),
        line_colors=("yellow", "red")
    )


    VaR_10, ES_10, total_pnl_10 = get_risk_t(
        df,
        days=10,
        ci=ci,
        investment=investment,
        weights=weights,
        nu=nu,
        volatility_scaler=volatility_coefficient
    )

    create_hist_plot(total_pnl_10,
        title=f"Stress Test 3. Monte Carlo Simulation over {10} days",
        lines=(VaR_10, ES_10), 
        lines_labels=("VaR", "Expected Shortfall"),
        line_colors=("yellow", "red")
    )


    print(f"""
        ---Stress Test Scenario 3. Increased Volatility---\n
            Parameters:\n
                Volatility increase: {volatility_coefficient*100}%\n
                Degrees of freedom: {nu}\n
            Results:\n
                5-day VaR (Student t): ${VaR_5:,.0f}\n
                5-day ES (Student t): ${ES_5:,.0f}\n
                10-day VaR (Student t): ${VaR_10:,.0f}\n
                10-day ES (Student t): ${ES_10:,.0f}
    """)

#   Scenario 4. Increased Tail Risk
    ci = 0.95
    nu = 3

    VaR_5, ES_5, total_pnl_5 = get_risk_t(
        df,
        days=5,
        ci=ci,
        investment=investment,
        weights=weights,
        nu=nu
    )

    create_hist_plot(
        total_pnl_5,
        title=f"Stress Test 4. Monte Carlo Simulation over {5} days",
        lines=(VaR_5, ES_5),
        lines_labels=("VaR", "Expected Shortfall"),
        line_colors=("yellow", "red")
    )


    VaR_10, ES_10, total_pnl_10 = get_risk_t(df, days=10, ci=ci, investment=investment, weights=weights, nu=nu)

    create_hist_plot(
        total_pnl_10,
        title=f"Stress Test 4. Monte Carlo Simulation over {10} days",
        lines=(VaR_10, ES_10), 
        lines_labels=("VaR", "Expected Shortfall"),
        line_colors=("yellow", "red")
    )


    print(f"""
        ---Stress Test Scenario 4. Increased Tail Risk---\n
            Parameters:\n
                Degrees of freedom: {nu}\n
            Results:\n
                5-day VaR (Student t): ${VaR_5:,.0f}\n
                5-day ES (Student t): ${ES_5:,.0f}\n
                10-day VaR (Student t): ${VaR_10:,.0f}\n
                10-day ES (Student t): ${ES_10:,.0f}
    """)
    
#   Scenario 5. Extreme Volatility and Increased Tail Risk
    ci = 0.95
    nu = 3
    volatility_coefficient = 4

    VaR_5, ES_5, total_pnl_5 = get_risk_t(
        df,
        days=5,
        ci=ci,
        investment=investment,
        weights=weights, 
        nu=nu,
        volatility_scaler=volatility_coefficient
    )

    create_hist_plot(
        total_pnl_5,
        title=f"Stress Test 5. Monte Carlo Simulation over {5} days",
        lines=(VaR_5, ES_5),
        lines_labels=("VaR", "Expected Shortfall"),
        line_colors=("yellow", "red")
    )


    VaR_10, ES_10, total_pnl_10 = get_risk_t(
        df,
        days=10,
        ci=ci,
        investment=investment,
        weights=weights,
        nu=nu,
        volatility_scaler=volatility_coefficient
    )

    create_hist_plot(
        total_pnl_10,
        title=f"Stress Test 5. Monte Carlo Simulation over {10} days",
        lines=(VaR_10, ES_10), 
        lines_labels=("VaR", "Expected Shortfall"),
        line_colors=("yellow", "red")
    )


    print(f"""
        ---Stress Test Scenario 5. Extreme Volatility and Increased Tail Risk---\n
            Parameters:\n
                Volatility increase: {volatility_coefficient*100}%\n
                Degrees of freedom: {nu}\n
            Results:\n
                5-day VaR (Student t): ${VaR_5:,.0f}\n
                5-day ES (Student t): ${ES_5:,.0f}\n
                10-day VaR (Student t): ${VaR_10:,.0f}\n
                10-day ES (Student t): ${ES_10:,.0f}
    """)

if __name__ == "__main__":
    main()


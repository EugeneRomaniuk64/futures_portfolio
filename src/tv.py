from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

#Used to download new data. NOTE: API is sometimes faulty
def tv_download(tickers: pd.DataFrame, interval = Interval.in_daily, n_bars=2900) -> pd.DataFrame: #tickers in format: symbol, exchange
    tv = TvDatafeed()
    returns_df = pd.DataFrame()

    for i in range(len(tickers)):
        df: pd.DataFrame = tv.get_hist(
            symbol=tickers.index[i],
            exchange=tickers["Exchange"].iloc[i],
            interval=interval,
            n_bars=n_bars,
            fut_contract=1
            )
        
        returns_df[f"{tickers.index[i]}"] = df["close"].values
        returns_df = returns_df.set_index(df.index.values)

    return returns_df.dropna()

#Imports data from a file (Preferred method)
def tv_import(path: str) -> pd.DataFrame:
    df = pd.read_csv(f"./data/{path}")
    df.set_index(pd.to_datetime(df.iloc[:, 0]), inplace=True)
    df = df.drop(df.columns[0], axis=1)
    return df

def get_pnl(price_df, num_contracts, multipliers) -> pd.Series:
    delta_price = price_df.diff().dropna() 
    return delta_price * num_contracts * multipliers

#Computes historical VaR and ES
def get_risk_hist(pnl: pd.Series, days, ci):
    range_pnl = pnl.rolling(window=days).sum() 
    range_pnl = range_pnl.dropna()

    var = -np.quantile(range_pnl, 1 - ci)
    es = -range_pnl[range_pnl <= -var].mean()

    return float(var), float(es), pd.Series(range_pnl)

def get_risk_t(pnl: pd.DataFrame, days, ci, num_sims = 100_000, nu = 4, volatility_scaler = 1):
    # Generate multi-day PnL simulations:
    #   1. Draw standard normals for each asset and day
    #   2. Apply Cholesky to introduce correlations
    #   3. Scale by chi-square for t-distribution tails
    #   4. Compute weighted portfolio daily PnL and sum over days
    # Compute VaR and ES

    mu = pnl.mean().values #shape: (n_assets,)
    Sigma = pnl.cov().values #shape: (n_assets, n_assets)
    n_assets = len(mu)
    
    shape = Sigma * (nu - 2) / nu
    shape = shape * volatility_scaler ** 2
    L = np.linalg.cholesky(shape)

    # Generating all normal draws (our numerator)
    rng = np.random.default_rng(123) #Creating the random seed
    Z = rng.standard_normal(size=(num_sims, days, n_assets))
    daily_normal = Z @ L.T #shape: (num_sims, days, n_assets)

    # Generating 1 chi-square per simulation and scaling it (our denominator)
    U = rng.chisquare(df=nu, size=num_sims) #shape: (num_sims,)
    scale = np.sqrt(U / nu)
    scale = scale[:, None, None] #shape: (num_sims, 1, 1)

    # Combining them into multivariate Student t ( mu + Z/(sqrt(U/nu)) )
    daily_pnl = mu + daily_normal / scale #shape: (num_sims, days, n_assets)
    
    portfolio_daily = daily_pnl.sum(axis=2) #shape: (num_sims, days)
    total_pnl = portfolio_daily.sum(axis=1) #shape: (num_sims,)


    var = -np.quantile(total_pnl, 1 - ci)
    es = -total_pnl[total_pnl <= -var].mean()

    return float(var), float(es), pd.Series(total_pnl)


#Plots returns with several vertical lines which can show VaR, ES or other metrics
def create_hist_plot(data: pd.Series, title: str, lines: tuple[float], lines_labels: tuple[str], line_colors: tuple[str]):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=120, alpha=0.75, color="blue", edgecolor="black")

    if lines != None:
        for line, label, color in zip(lines, lines_labels, line_colors):
            plt.axvline(-line, linestyle = "--", label = f"{label}: ${round(line):,}", color = color)

    plt.title(title)
    plt.xlabel("Returns")
    plt.ylabel("Frequency")
    plt.xlim(-1_000_000_000, 1_000_000_000)
    plt.legend()
    plt.show()

#Shows cumulative P&L cash flows as a chart
def pnl_chart(pnl, capital, title):
    starting_capital = pd.Series([capital], index=[pnl.index[0]-dt.timedelta(days=1)])
    equity = capital + pnl.cumsum()
    combined_equity = pd.concat([starting_capital, equity])

    plt.figure()
    plt.plot(combined_equity.index, combined_equity)
    plt.axhline(0)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.tight_layout()
    plt.show()

#Historical stress test. Computes historical as well as Monte Carlo VaR and ES
def scenario_hist(df, n_contracts, multipliers, start_date: dt.datetime, end_date: dt.datetime, title, initial_margin):
    days = 5
    ci = 0.95
    nu = 5

    scenario_df = df[df.index.to_series().between(start_date, end_date)]

    
    scenario_pnl_df = pd.DataFrame(get_pnl(scenario_df, n_contracts, multipliers))
    scenario_pnl = scenario_pnl_df.sum(axis=1)

    

    VaR_roll, ES_roll, total_pnl_roll = get_risk_hist(scenario_pnl, days=days, ci=ci)

    total_loss_gain = scenario_pnl.sum()

    pnl_chart(scenario_pnl, initial_margin, title=f"{title}. Portfolio Equity")

    VaR_mc, ES_mc, total_pnl = get_risk_t(
        pnl=scenario_pnl_df,
        days=days,
        ci=ci,
        nu=nu
    )


    create_hist_plot(
        total_pnl,
        title=f"{title} Monte Carlo Simulation over {days} days",
        lines=(VaR_mc, ES_mc),
        lines_labels=("VaR", "Expected Shortfall"),
        line_colors=("yellow", "red")
    )


    print(f"""
        ---{title}---\n
            Parameters:\n
                Start date: {start_date.strftime('%Y-%m-%d')}\n
                End date: {end_date.strftime('%Y-%m-%d')}\n
            Results:\n
                Historical:\n\n
                5-day VaR (rolling): ${VaR_roll:,.0f}\n
                5-day ES (rolling): ${ES_roll:,.0f}\n
                Total gain/loss: ${total_loss_gain:,.0f}\n\n
                Monte Carlo:\n\n
                5-day VaR (Student t): ${VaR_mc:,.0f}\n
                5-day ES (Student t): ${ES_mc:,.0f}
        """)


def main():
    tickers_list = pd.DataFrame(
        [
            ["NYMEX", "WTI Crude Oil", 1000, 145, 41.63], 
            ["NYMEX", "RBOB Gasoline", 42_000, 206, 1.8033],
            ["NYMEX", "Natural Gas", 10_000, 756, 1.583],
            ["COMEX", "Gold", 100, 57, 4319.7],
            ["COMEX", "Silver", 5000, 69, 49.82],
            ["NYMEX", "Platinum", 50, 363, 1348.2],
            ["COMEX", "Aluminum (NOT Aluminium)", 25, 911, 2093.75],
            ["COMEX", "Copper", 25_000, 477, 3.1315],
            ["CME", "Live Cattle", 40_000, 194, 204.325],
            ["CBOT", "Soybeans", 5000, 335, 1060]
        ], 
        columns=["Exchange", "Description", "Multiplier", "Amount", "Entry Price"],
        index=["CL", "RB", "NG", "GC", "SI", "PL", "ALI", "HG", "LE", "ZS"]
    )

    initial_margin = 114000000

    df = tv_import("tv_data.csv")

    multipliers = tickers_list.loc[:, "Multiplier"].values
    n_contracts = tickers_list.loc[:, "Amount"].values


    prices_today = tickers_list["Entry Price"]
    tickers_list["Exposure"] = tickers_list["Multiplier"] * tickers_list["Amount"] * prices_today
    print(f"Total Notional Exposure: ${tickers_list["Exposure"].sum():,.0f}")


    pnl = get_pnl(df, n_contracts, multipliers)
    portfolio_pnl = pnl.sum(axis=1)

    #Hisorical risk metrics
    VaR, ES, rolling_pnl = get_risk_hist(portfolio_pnl, 10, 0.95)

    print(f"""
        ---Common Risk Metrics---\n
            10-days VaR (historical): ${VaR:,.0f}\n
            10-days ES (historical): ${ES:,.0f}
    """)    

    create_hist_plot(
        rolling_pnl,
        title="Common Risk Metrics",
        lines=(VaR, ES), 
        lines_labels=("VaR", "ES"),
        line_colors=("yellow", "red")
    )

#   Scenario 1. Oil Price Crash 2020
    start_date = dt.datetime(2020, 2, 20)
    end_date = dt.datetime(2020, 4, 30)
    
    scenario_hist(
        df=df,
        n_contracts=n_contracts, 
        multipliers=multipliers, 
        start_date=start_date, 
        end_date=end_date, 
        title="Stress Test Scenario 1. Oil Price Crash 2020", 
        initial_margin=initial_margin
        )
    

#   Scenario 2. Energy Crisis 2022

    start_date = dt.datetime(2022, 2, 24)
    end_date = dt.datetime(2022, 10, 31)
    
    scenario_hist(
        df=df,
        n_contracts=n_contracts,
        multipliers=multipliers,
        start_date=start_date,
        end_date=end_date,
        title="Stress Test Scenario 2. Energy Crisis 2022",
        initial_margin=initial_margin
        )

#   Scenario 3. Increased Volatility
    ci = 0.95
    nu = 5
    volatility_coefficient = 2

    VaR_5, ES_5, total_pnl_5 = get_risk_t(
        pnl=pnl,
        days=5,
        ci=ci,
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
        pnl=pnl,
        days=10,
        ci=ci,
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
        pnl=pnl,
        days=5,
        ci=ci,
        nu=nu
    )

    create_hist_plot(
        total_pnl_5,
        title=f"Stress Test 4. Monte Carlo Simulation over {5} days",
        lines=(VaR_5, ES_5),
        lines_labels=("VaR", "Expected Shortfall"),
        line_colors=("yellow", "red")
    )


    VaR_10, ES_10, total_pnl_10 = get_risk_t(
        pnl=pnl, 
        days=10, 
        ci=ci, 
        nu=nu
    )

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
        pnl=pnl,
        days=5,
        ci=ci,
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
        pnl=pnl,
        days=10,
        ci=ci,
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
    



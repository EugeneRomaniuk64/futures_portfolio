from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import datetime as dt


def tv_download(tickers: list[tuple[str, str, str]], interval: str, n_bars: int) -> pd.DataFrame:
    tv = TvDatafeed()
    returns_df = pd.DataFrame()

    for i in range(len(tickers)):
        df: pd.DataFrame = tv.get_hist(symbol=tickers[i][0], exchange=tickers[i][1], interval=interval, n_bars=n_bars, fut_contract=1)
        returns_df[f"{tickers[i][0]}"] = df["close"].values
        returns_df.set_index(df.index.values, inplace=True)

    return returns_df.dropna()


def tv_import(path: str) -> pd.DataFrame:
    df = pd.read_csv(f"./data/{path}")
    df.set_index(pd.to_datetime(df.iloc[:, 0]), inplace=True)
    df = df.drop(df.columns[0], axis=1)
    return df
        

def get_pnl(price_df: pd.DataFrame, weights: list[float], investment: float) -> pd.Series:
    simple_returns = price_df.pct_change().dropna() * investment
    return simple_returns @ weights

def get_risk_hist(pnl: pd.Series, days, ci):
    range_pnl = pnl.rolling(window=days).sum()
    range_pnl = range_pnl.dropna()
    var = -np.quantile(range_pnl, 1 - ci)
    es = -range_pnl[range_pnl <= -var].mean()

    return (float(var), float(es), pd.Series(range_pnl))


def get_risk_t(mu, std, days, ci, num_sims = 100_000, df = 5):
    sim_pnl = t.rvs(df=df, size=(num_sims, days)) 
    sim_pnl = sim_pnl * std / np.sqrt(df / (df - 2)) + mu

    total_pnl = sim_pnl.sum(axis=1) #We are summing up simple returns, but that's correct in the case of PnL modelling
    var = -np.quantile(total_pnl, 1 - ci) #Value at Risk -- our worst case loss in n days with a certain confidence
    es = -total_pnl[total_pnl <= -var].mean() #Expected Shortfall -- mean losses beyond our VaR

    return (float(var), float(es), pd.DataFrame(sim_pnl), pd.Series(total_pnl))



def get_risk(mu, std, days, ci, num_sims = 100_000, df = 5, get_sim_returns = False):
    res = get_risk_t(mu, std, days, ci, num_sims, df)
    if get_sim_returns:
        return res
    else:
        return (res[0], res[1])


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

def pnl_chart(pnl, capital):
    equity = capital + pnl.cumsum()
    plt.figure()
    plt.plot(equity.index, equity)
    plt.axhline(0)
    plt.title("Portfolio Equity During 2020 Oil Crash Stress Scenario")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.tight_layout()
    plt.show()

def scenario_hist(start_date: dt.datetime, end_date: dt.datetime, title):
    days=5
    ci=0.95

    scenario_df = df[df.index.to_series().between(start_date, end_date)]

    scenario_pnl = get_pnl(scenario_df, weights, investment)



    VaR_roll, ES_roll, total_pnl_roll = get_risk_hist(scenario_pnl, days=days, ci=ci)

    VaR_scale, ES_scale, total_pnl_scale = get_risk_hist(scenario_pnl, days=1, ci=ci)

    VaR_scale = VaR_scale * np.sqrt(days)
    ES_scale = ES_scale * np.sqrt(days)


    total_loss_gain = scenario_pnl.sum()

    pnl_chart(scenario_pnl, 250_000_000)
    VaR_mc, ES_mc, sim_pnl, total_pnl = get_risk(scenario_pnl.mean(), scenario_pnl.std(), days=days, ci=ci, df=4, get_sim_returns=True)


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

if __name__ == "__main__":
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

    weights: np.ndarray = np.array([1/len(tickers_list)]*len(tickers_list)) 
    investment = 5_000_000_000 * 0.95

#   Scenario 1. Oil Price Crash 2020
    start_date = dt.datetime(2020, 2, 20)
    end_date = dt.datetime(2020, 4, 30)
    
    scenario_hist(start_date, end_date, "Stress Test Scenario 1. Oil Price Crash 2020")
    

#   Scenario 2. Energy Crisis 2022

    start_date = dt.datetime(2022, 2, 24)
    end_date = dt.datetime(2022, 10, 31)
    
    scenario_hist(start_date, end_date, "Stress Test Scenario 2. Energy Crisis 2022")

#   Scenario 3. Increased Volatility
    ci = 0.95

    pnl = get_pnl(df, weights, investment)

    volatility_coefficient = 2

    VaR_5, ES_5, sim_pnl_5, total_pnl_5 = get_risk(pnl.mean(), pnl.std() * volatility_coefficient, days=5, ci=ci, df=4, get_sim_returns=True)

    create_hist_plot(total_pnl_5, title=f"Stress Test 3. Monte Carlo Simulation over {5} days", lines=(VaR_5, ES_5),
                    lines_labels=("VaR", "Expected Shortfall"), line_colors=("yellow", "red"))


    VaR_10, ES_10, sim_pnl_10, total_pnl_10 = get_risk(pnl.mean(), pnl.std() * volatility_coefficient, days=10, ci=ci, df=4, get_sim_returns=True)

    create_hist_plot(total_pnl_10, title=f"Stress Test 3. Monte Carlo Simulation over {10} days", lines=(VaR_10, ES_10), 
                    lines_labels=("VaR", "Expected Shortfall"), line_colors=("yellow", "red"))


    print(f"""
        ---Stress Test Scenario 3. Increased Volatility---\n
            Parameters:\n
                Volatility increase: {volatility_coefficient*100}%\n
            Results:\n
                5-day VaR (Student t): ${VaR_5:,.0f}\n
                5-day ES (Student t): ${ES_5:,.0f}\n
                10-day VaR (Student t): ${VaR_10:,.0f}\n
                10-day ES (Student t): ${ES_10:,.0f}
        """)
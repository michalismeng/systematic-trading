import pandas as pd
import datetime
from lightweight_charts import Chart, AbstractChart

with open("activity.csv", "rb") as f:
    results = pd.read_csv(f, parse_dates=["date"], index_col="date")

def set_weights_diff_table(chart: AbstractChart, time, price):
    date = datetime.datetime.fromtimestamp(time).date()
    df = pd.DataFrame([(s.drift["weights_diff"]) for s in results.simulation], index=[s.date for s in results.simulation])
    rebalances = pd.DataFrame([s.rebalance_params[0] for s in results.simulation], columns=["no_trade_zone"], index=[s.date for s in results.simulation])
    weights_diff_table.clear()
    weights_diff_table.footer[0] = f"Weight Diff on {date} - No trade zone: {rebalances.loc[date, 'no_trade_zone'] / 2:.2%}"
    for key, value in df.loc[date].items():
        row = weights_diff_table.new_row(key, f"{value:.2%}")

def create_max_drift_line(chart: AbstractChart):
    df = pd.DataFrame([(s.drift["weights_diff"].abs().max() * 100) for s in results.simulation[1:]], index=[s.date for s in results.simulation[1:]])
    df.rename(columns={0: "Max Drift Weight (%)"}, inplace=True)
    df["Max Drift"] = df.max().iloc[0]
    line = chart.create_line(name="Max Drift Weight (%)")
    line.set(df[["Max Drift Weight (%)"]])
    line = chart.create_line(name="Max Drift", color="red", style="dashed")
    line.set(df[["Max Drift"]])
    line = chart.create_line(name="No Trade Zone", color="green")
    df = pd.DataFrame([(s.rebalance_params[0] / 2 * 100) for s in results.simulation[1:]], index=[s.date for s in results.simulation[1:]])
    df.rename(columns={0: "No Trade Zone"}, inplace=True)
    line.set(df[["No Trade Zone"]])
    return line

def create_cash_line(chart: AbstractChart):
    df = pd.DataFrame([(s.market_value.loc["$CASH", "market_value"]) for s in results.simulation[1:]], index=[s.date for s in results.simulation[1:]])
    df.rename(columns={0: "Cash"}, inplace=True)
    line = chart.create_line(name="Cash")
    line.set(df[["Cash"]])
    return line

def create_cash_percentage_line(chart: AbstractChart):
    df = pd.DataFrame([(s.market_value.loc["$CASH", "current_weight"] * 100) for s in results.simulation[1:]], index=[s.date for s in results.simulation[1:]])
    df.rename(columns={0: "Cash Weight (%)"}, inplace=True)
    df["Max Cash Weight"] = df.max().iloc[0]
    line = chart.create_line(name="Cash Weight (%)")
    line.set(df[["Cash Weight (%)"]])
    line = chart.create_line(name="Max Cash Weight", color="red", style="dashed")
    line.set(df[["Max Cash Weight"]])
    return line

def create_market_value_line(chart: AbstractChart):
    df = pd.DataFrame([(s.market_value["market_value"]) for s in results.simulation[1:]], index=[s.date for s in results.simulation[1:]])
    for key in df.columns:
        line = chart.create_line(name=key)
        line.set(df[[key]])
    return line

def create_nav_line(chart: AbstractChart):
    df = pd.DataFrame(results["nav"].tolist(), columns=[f"Nav"], index=results.index.tolist())
    line = chart.create_line(name=f"Nav")
    line.set(df)
    return line

def create_forecast_line(chart: AbstractChart):
    df = pd.DataFrame(results["forecast"].tolist(), columns=[f"Forecast"], index=results.index.tolist())
    line = chart.create_line(name=f"Forecast")
    line.set(df)
    return line

def create_volatility_line(chart: AbstractChart):
    df = pd.DataFrame(results["volatility"].tolist(), columns=[f"Volatility"], index=results.index.tolist())
    line = chart.create_line(name=f"Volatility")
    line.set(df)
    return line

def create_volatility_scalar_line(chart: AbstractChart):
    df = pd.DataFrame(results["instrument_volatility_scalar"].tolist(), columns=[f"Volatility Scalar"], index=results.index.tolist())
    line = chart.create_line(name=f"Volatility Scalar")
    line.set(df)
    return line

def create_nav_volatility_line(chart: AbstractChart):
    vol = results["nav"].pct_change().dropna().rolling(window=25).std().dropna() * 16
    df = pd.DataFrame(vol.tolist(), columns=[f"Nav Volatility"], index=vol.index.tolist())
    line = chart.create_line(name=f"Nav Volatility")
    line.set(df)
    return line

if __name__ == '__main__':
    chart = Chart(inner_width=0.5, inner_height=0.33, title=f"Backtest", maximize=True)
    # create_nav_line(chart)


    # table = chart.create_table(width=0.3, height=0.5,
    #               headings=('Parameter', 'Value'),
    #               widths=(0.2, 0.1),
    #               alignments=('center', 'center'),
    #               position='right', func=lambda row: None)

    # table.new_row("Initial cash", results.initial_value)
    # for key in results.analysis:
    #     table.new_row(key, results.analysis[key])

    # forecast_chart = chart.create_subchart(position='bottom', width=0.5, height=0.33, sync=True)
    # create_forecast_line(forecast_chart)
    # forecast_chart.legend(True)

    # volatility_chart = chart.create_subchart(position='bottom', width=0.5, height=0.33, sync=True)
    # create_volatility_line(volatility_chart)
    # volatility_chart.legend(True)

    volatility_scalar_chart = chart.create_subchart(position='right', width=0.5, height=0.33, sync=True)
    create_nav_volatility_line(volatility_scalar_chart)
    volatility_scalar_chart.legend(True)

    # tx_table = chart.create_table(width=0.3, height=0.5,
    #               headings=('Date', 'Ticker', 'Quantity', 'Total'),
    #               widths=(0.2, 0.1, 0.2, 0.2, 0.3),
    #               alignments=('center', 'center', 'right', 'right'),
    #               position='right', func=lambda row: None)

    # for t in results.transactions:
    #     tx_table.new_row(t.date.strftime("%d %b '%y"), t.ticker, int(t.quantity), f"{t.price * t.quantity:.0f}")

    # tx_table.header(1)
    # tx_table.header[0] = "Transactions Table"

    # max_drift_chart = chart.create_subchart(position='left', width=0.7, height=0.33, sync=True)
    # create_max_drift_line(max_drift_chart)
    # max_drift_chart.events.click += set_weights_diff_table
    # max_drift_chart.legend(True, ohlc=False, percent=True)

    # weights_diff_table = chart.create_table(width=0.3, height=0.5,
    #               headings=('Ticker', 'Weight Diff'),
    #               widths=(0.2, 0.3),
    #               alignments=('center', 'center'),
    #               position='right', func=lambda row: None)

    # weights_diff_table.header(1)
    # weights_diff_table.header[0] = "Weights Diff Table"
    # weights_diff_table.footer(1)


    chart.show(block=True)
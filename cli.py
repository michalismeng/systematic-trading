"""A command-line interface for running systematic trading backtests using the Nautilus Trader framework."""
import cmd
import datetime
import shutil
from pathlib import Path

import requests
from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.config import (
    BacktestDataConfig,
    BacktestEngineConfig,
    BacktestRunConfig,
    BacktestVenueConfig,
    ImportableStrategyConfig,
)
from nautilus_trader.model import Venue
from nautilus_trader.model.data import Bar
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.persistence.loaders import CSVBarDataLoader
from nautilus_trader.persistence.wranglers_v2 import BarDataWranglerV2
from nautilus_trader.test_kit.providers import TestInstrumentProvider

CATALOG_PATH = Path.cwd() / "catalog"
DATA_DIR = Path.cwd() / "data"


def configure_backtest(strategy_config: ImportableStrategyConfig, venue: str = "US",
                       start_date: datetime.datetime = datetime.datetime(2010, 1, 1),
                       starting_balance: str = "1_000_000 USD") -> BacktestRunConfig:
    """Configure a backtest run for a given strategy and venue."""
    catalog = ParquetDataCatalog(CATALOG_PATH)
    venue_config = BacktestVenueConfig(
        name=venue,
        oms_type="NETTING",
        account_type="CASH",
        base_currency="USD",
        starting_balances=[starting_balance],
    )

    data_config = BacktestDataConfig(
        catalog_path=str(CATALOG_PATH),
        data_cls=Bar,
        instrument_ids=[
            instrument.id
            for instrument in catalog.instruments()
            if instrument.venue.value == venue
        ],
        start_time=start_date.isoformat(),
        end_time=None,
    )

    engine_config = BacktestEngineConfig(strategies=[strategy_config])

    config = BacktestRunConfig(
        engine=engine_config,
        data=[data_config],
        venues=[venue_config],
    )

    return config


class SystematicTradingCLI(cmd.Cmd):
    """A command-line interface for running systematic trading backtests using the Nautilus Trader framework."""
    intro = "Welcome to the Systematic Trading CLI. Type help or ? to list commands."

    def do_download_data(self, _):
        """Download historical end-of-day prices for AAPL from EODHD API (demo account)."""
        url = "https://eodhd.com/api/eod/AAPL.US?api_token=demo&fmt=csv"
        output_path = DATA_DIR / "AAPL.US.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading data from {url} to {output_path}...")
        response = requests.get(url)
        with Path.open(output_path, "wb") as f:
            f.write(response.content)

    def do_ingest_data(self, _):
        """Ingest dataset into a Nautilus catalog.

        Notes:
            - The data should reside in the ./data directory.
              Each CSV file should be named in the format SYMBOL.VENUE.csv (e.g., AAPL.US.csv).
            - The data is expected to be in CSV format with columns: date, open, high, low, close, adjusted_close,
                                                                     volume.
            - Creates Nautilus file-based catalog under ./catalog. If a catalog already exists, it is re-created.
        """
        if CATALOG_PATH.exists():
            shutil.rmtree(CATALOG_PATH)
        CATALOG_PATH.mkdir(parents=True)
        catalog = ParquetDataCatalog(CATALOG_PATH)

        for path in DATA_DIR.glob("*.csv"):
            df_snap = CSVBarDataLoader.load(
                path,
                header=0,
                index_col="date",
                names=[
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "adjusted_close",
                    "volume",
                ],
                usecols=["date", "open", "high", "low", "close"],
                parse_dates=True,
            )
            df_snap = df_snap.sort_index()
            df_snap["timestamp"] = df_snap.index
            instrument = path.stem
            symbol, venue = instrument.split(".")
            wrangler = BarDataWranglerV2(f"{instrument}-1-DAY-LAST-EXTERNAL", 2, 0)
            bars = wrangler.from_pandas(df_snap)
            catalog.write_data([TestInstrumentProvider().equity(symbol, venue)])
            catalog.write_data(bars)

        print("Instruments ingested:", len(catalog.instruments()))
        print("Bars ingested: ", len(catalog.bars()))

    def do_list_instruments(self, _):
        """List instruments available in the catalog."""
        catalog = ParquetDataCatalog(CATALOG_PATH)
        print(catalog.instruments())

    def do_count_bars(self, _):
        """Cound the number of bars available in the catalog."""
        catalog = ParquetDataCatalog(CATALOG_PATH)
        bars = catalog.bars()
        print(len(bars))

    def do_backtest(self, instrument=None):
        """Run a backtest against the specified instrument.

        Notes:
            - Currently only one instrument can be backtested at a time.
              The instrument should be in the format SYMBOL.VENUE (e.g., AAPL.US).
            - If no instrument is specified, the backtest will default to AAPL.US.
            - No commissions are applied.
        """
        if not instrument:
            instrument = "AAPL.US"

        strategy = ImportableStrategyConfig(
            strategy_path="strategy:VolTargetingStrategy",
            config_path="strategy:VolTargetingStrategyConfig",
            config={
                "bar_type": f"{instrument}-1-DAY-LAST-EXTERNAL",
            },
        )

        config = configure_backtest(strategy)
        node = BacktestNode(configs=[config])
        results = node.run()

        print("Backtest completed. Results:")
        print(results)

        engine = node.get_engine(config.id)
        if engine is None:
            raise Exception(f"Invalid engine for config id: {config.id}")

        fills = engine.trader.generate_order_fills_report()
        positions = engine.trader.generate_positions_report()
        account = engine.trader.generate_account_report(Venue("US"))

        fills.to_csv("backtest_fills.csv")
        positions.to_csv("backtest_positions.csv")
        account.to_csv("backtest_account.csv")

    def do_exit(self, _):
        """Exit the REPL."""
        print("Goodbye!")
        return True

    do_quit = do_exit
    do_EOF = do_exit  # Ctrl-D to exit  # noqa: N815


if __name__ == "__main__":
    SystematicTradingCLI().cmdloop()

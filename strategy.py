"""A simple systematic trading strategy that demonstrates how to use the Nautilus Trader framework."""

import pandas as pd
from nautilus_trader.accounting.accounts.base import Account
from nautilus_trader.cache import Cache
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import Bar, BarType, Currency, Money, Position
from nautilus_trader.model.enums import OrderSide, PriceType, TimeInForce
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Quantity
from nautilus_trader.model.orders.market import MarketOrder
from nautilus_trader.trading.strategy import Strategy
from synthetic_forecasts import generate_synthetic_forecasts

LONG_TERM_FORECAST_AVERAGE = 10
FORECAST_CAP = 20

class VolTargetingCalculator:
    """A class to perform volatility standardization calculations for an instrument."""

    def __init__(self, cache: Cache, instrument: Instrument, bar: Bar, vol_days: int, target_risk: float,
                 forecast: float, forecast_average: float):
        """A class to perform volatility standardization calculations for an instrument.

        Args:
            cache: Nautilus trader cache to access data and metadata.
            instrument: The instrument for which to perform the calculations.
            bar: The bar for which to perform the calculations.
            vol_days: The number of days to use for 'recent' volatility calculations.
            target_risk: The annualized target risk for the portfolio (e.g., 0.12 for 12%).
            forecast: The forecast for the instrument, properly scaled and capped.
            forecast_average: The long-term average of the absolute value of the forecast.
        """
        self.cache = cache
        self.instrument = instrument
        self.bar = bar
        self.vol_days = vol_days
        self.target_risk = target_risk
        self.forecast = forecast
        self.forecast_average = forecast_average

    @property
    def account(self) -> Account:
        """Get the single account."""
        accounts = self.cache.accounts()
        assert len(accounts) == 1, "Expected exactly one account in the cache."
        return accounts[0]

    @property
    def balance(self) -> Money:
        """Get the current account balance (cash available for trading)."""
        balances: dict[Currency, Money] = self.account.balances_total()
        assert len(balances) == 1, "Mutil-currency accounts not supported."
        return list(balances.values())[0]

    @property
    def nav(self) -> Money:
        """Get the NAV, as the account balance plus mark-to-market of open positions across instruments."""
        positions = [p for p in self.cache.positions() if p.is_open]
        nav = self.balance
        # XXX: `sum(p.notional_value())` returns float instead of Money.
        for p in positions:
            nav += p.notional_value(self.bar.close)
        return nav

    def annual_cash_volatility(self) -> Money:
        """Get the annualized cash volatility for the account.

        Shows how much cash we should be risking, given the current account size and the annual volatility target.

        Notes:
            - Annualized cash volatility = Current account value * annualized target risk
        """
        return Money(self.nav.as_double() * self.target_risk, self.nav.currency)

    def daily_cash_volatility(self) -> Money:
        """Get the daily cash volatility for the account.

        Shows how much cash we should be risking on a single day, given the current account size and
        the annual volatility target.

        Notes:
            - Daily cash volatility = Annualized cash volatility / 16
                                    = Current account value * annualized target risk / 16
            - Division by 16 accounts for 256 business days.
        """
        acv = self.annual_cash_volatility()
        return Money(acv.as_double() / 16, acv.currency)

    def volatility(self):
        """Get recent volatility of the instrument.

        Calculated as the standard deviation of daily returns over the specified number of most recent days.
        """
        bars: list[Bar] = self.cache.bars(self.bar.bar_type)
        if len(bars) <= self.vol_days:
            raise ValueError(f"Not enough data points to calculate volatility(N={self.vol_days}).")
        bars = bars[:self.vol_days + 1]
        # Reverse to have oldest first, since that's how Pandas expects the data.
        prices = list(reversed([b.close.as_double() for b in bars]))
        return pd.to_numeric(pd.Series(prices).pct_change().dropna().rolling(self.vol_days).std().dropna().squeeze())

    def block_value(self) -> Money:
        """Get the block value of the instrument.

        The block value shows the exposure we'd get if we were exposed to "one" unit of an instrument
        using today's prices.
        For stocks, it's the price of one share. For futures, it's the notional value of one contract.
        """
        quantity = self.instrument.make_qty(1)
        price = self.cache.price(self.instrument.id, PriceType.LAST)
        return self.instrument.notional_value(quantity, price)

    def currency_volatility(self) -> Money:
        """Daily cash volatility for a block of the instrument in its quote currency.

        Shows the volatility-adjusted exposure we'd get if we were exposed to "one" unit of an instrument
        using today's prices, in the quote currency of the instrument.
        """
        volatility = self.volatility()
        bvalue = self.block_value()
        return Money(bvalue.as_double() * volatility, bvalue.currency)

    def value_volatility(self) -> Money:
        """Daily cash volatility in the account currency.

        Shows the volatility-adjusted exposure we'd get if we were exposed to "one" unit of an instrument
        using today's prices, in the account's main currency.

        Notes:
            - Currently no fx conversion is applied,
              so only instruments with the same quote currency as the account currency are supported.
        """
        cvol = self.currency_volatility()
        if self.balance.currency != cvol.currency:
            raise NotImplementedError("Currency conversion not implemented yet."
                                      " Only accounts with the same currency as the instrument are supported.")
        fx = 1
        return Money(cvol.as_double() * fx, self.balance.currency)

    def volatility_scalar(self) -> float:
        """Scaling factor between instrument's natural volatility and portfolio's required volatility.

        Shows how many blocks of the instrument we should hold to achieve the portfolio's target volatility,
        if we were to bet all available capital at this instrument and have an average buy forecast.
        """
        vvol = self.value_volatility()
        return self.daily_cash_volatility().as_double() / vvol.as_double()

    def subsystem_position(self) -> tuple[Quantity, OrderSide]:
        """The position for the instrument, assuming all capital is allocated to that instrument.

        The position is calculated as a scaling of the volatility scalar by the forecast,
        so that we take larger positions for stronger signals and smaller positions for weaker signals.

        Returns:
            - The optimal position and the side (LONG => BUY / SHORT => SELL).
              Assumes all capital is allocated to the instrument.
        """
        vol_scalar = self.volatility_scalar()
        signed_quantity = vol_scalar * self.forecast / self.forecast_average
        return Quantity(abs(signed_quantity), 4), OrderSide.BUY if signed_quantity > 0 else OrderSide.SELL

    def diagnostics(self) -> dict:
        """Get all values of the methods as diagnostics for the instrument."""
        return {
            "balance": self.balance.as_double(),
            "nav": self.nav.as_double(),
            "target_risk": self.target_risk,
            "forecast": self.forecast,
            "volatility": self.volatility() * 16,
            "daily_cash_volatility": self.daily_cash_volatility().as_double(),
            "block_value": self.block_value().as_double(),
            "currency_volatility": self.currency_volatility().as_double(),
            "value_volatility": self.value_volatility().as_double(),
            "instrument_volatility_scalar": self.volatility_scalar(),
            "subsystem_position": self.subsystem_position(),
        }


class VolTargetingStrategyConfig(StrategyConfig, frozen=True):
    """Configuration for the VolTargetingStrategy."""
    bar_type: BarType
    """Bar type to subscribe to, e.g., AAPL.US-1-DAY-LAST-EXTERNAL."""
    target_risk: float = 0.12
    """Annualized target risk for the portfolio, e.g., 0.12 for 12%."""
    forecast_average: float = LONG_TERM_FORECAST_AVERAGE
    """Long-term average of the absolute value of the forecast, used for scaling.
    Default is 10, which means that a forecast of +10 means a buy signal, -10 means a sell signal,
    and 0 means we should not take any position. Forecasts above the forecast average indicate a strong signal,
    while forecasts below the forecast average indicate a weak signal."""
    forecast_cap: float = FORECAST_CAP
    """Cap for the forecast. Default is 20, which means that any forecast above 20 will be treated as 20,
    and any forecast below -20 will be treated as -20. Useful to avoid taking excessively large positions
    in case of extreme forecasts."""


class VolTargetingStrategy(Strategy):
    """A simple systematic trading strategy that demonstrates how to use the Nautilus Trader framework.

    The strategy generates forecasts for the subscribed instruments and uses volatility targeting
    to determine optimal position sizes.
    It then compares with the current position and submits market orders to adjust the position accordingly.

    Notes:
        - The strategy supports trading a single instrument for now.
        - Current forecasts are synthetic and thus completely unrelated to the actual prices movements.
    """
    def __init__(self, config: VolTargetingStrategyConfig) -> None:
        """Initialize the strategy with the given configuration."""
        super().__init__(config)
        self.activity = []

    def on_start(self):
        """Start strategy."""
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar):
        """Receive new bar data and act on it."""
        self.act(bar)

    def act(self, bar: Bar) -> None:
        """Act on the newly arrived data."""
        # Get forecast for each traded instrument (currently single instrument only)
        # The forecast should be a signal of how likely it is for the price to change
        # in the direction of the forecast (positive = buy signal, negative = sell signal).
        forecasts = generate_synthetic_forecasts(
            instruments=[bar.bar_type.instrument_id.value],
            seed=int(bar.ts_event) % (2**31 - 1),
            scale=self.config.forecast_average,
            cap=self.config.forecast_cap,
        )

        # Negative forecasts mean short positions. We do not support this yet (CASH account).
        for k in forecasts:
            if forecasts[k] < 0:
                forecasts[k] = 0

        # Wait for 30 bars to have (more than) enough data to calculate volatility before trading.
        if len(self.cache.bars(self.config.bar_type)) < 30:
            self.log.info("Hydrating bars for volatility calculations...")
            return

        instrument: Instrument = self.cache.instrument(self.config.bar_type.instrument_id)
        calculator = VolTargetingCalculator(
            self.cache,
            instrument,
            bar,
            25,
            self.config.target_risk,
            forecasts[bar.bar_type.instrument_id.value],
            self.config.forecast_average,
        )

        positions: list[Position] = self.cache.positions(instrument_id=instrument.id)
        assert len(positions) <= 1, "Current system supports trading only a single instrument"

        optimal_position, side = calculator.subsystem_position()
        signed_pos = optimal_position.as_double() if side == OrderSide.BUY else -optimal_position.as_double()
        should_trade = True

        # If a position is open, calculate the difference between the optimal position and the current position,
        # and trade the difference.
        if len(positions) > 0:
            current_position = positions[0]
            signed_pos = signed_pos - current_position.signed_qty
            side = OrderSide.BUY if signed_pos > 0 else OrderSide.SELL
            if (instrument.notional_value(Quantity(int(abs(signed_pos)), 0), bar.close) >= calculator.balance
                and side == OrderSide.BUY):
                self.log.warning(f"Notional value of trade greater than current balance. Leverage required,"
                                 f" but not supported in CASH account. Skipping trade."
                                 f" Trade: {instrument.notional_value(Quantity(int(abs(signed_pos)), 0), bar.close)},"
                                 f" balance: {calculator.balance}")
                self.log.warning(f"Printing diagnostics:\n{calculator.diagnostics()}")
                should_trade = False

        if should_trade and int(abs(signed_pos)) > 0:
            order: MarketOrder = self.order_factory.market(
                instrument_id=instrument.id,
                order_side=side,
                quantity=Quantity(int(abs(signed_pos)), 0),
                time_in_force=TimeInForce.GTC,
            )
            self.submit_order(order)

        # TODO:
        # - Multiple instruments: calculate weights across multiple instruments using standard optimization.
        # - Apply position inertia (e.g., trade only if diff in positions > threshold).
        # - Apply commissions (and slippage).
        # - Multi-currency balances. Fx data needed.

        diagnostics = calculator.diagnostics()
        del diagnostics["subsystem_position"]
        diagnostics["order_side"] = side
        diagnostics["difference"] = signed_pos
        diagnostics["optimal_position"] = optimal_position.as_double()
        self.activity.append(diagnostics)

    def on_stop(self):
        """Stop strategy."""
        self.unsubscribe_bars(self.config.bar_type)
        pd.DataFrame(self.activity).to_csv("activity.csv", index=False)

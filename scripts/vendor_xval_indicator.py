"""
@pyne edge

Plot the canonical Pine Script TA built-ins, as PyneCore implements them, over
the bars tvlab feeds its own indicators.

Hand-written rather than PyneComp output, but it must carry the ``@pyne``
directive or ``pynecore.standalone.run`` refuses to load it (error R0047).

This is the reference half of the cross-validation in
``tests/test_pynecore_reference.py``. It is deliberately a plain Pine-style script
rather than one of pynecore-strategies' 471 compiled indicators: those are
third-party scripts with their own input defaults, smoothing and plotting choices,
so a disagreement could come from the author's parameters rather than from a real
difference in the underlying formula. Calling ``ta.*`` directly with tvlab's own
parameters isolates the semantics, which is the only thing worth comparing.

Run:  python vendor_xval_indicator.py <path/to/BTC_USDT.ohlcv>
Out:  vendor_xval_indicator.csv  (written next to this file)
"""
from pynecore.lib import close, high, low, plot, script, ta
from pynecore.types import Series


@script.indicator(shorttitle='tvlabXVal', title='tvlab indicator cross-validation',
                  overlay=False, timeframe='', na_bool=True)
def main():
    # --- the claims tvlab's docstrings actually make -------------------------
    # atr(): "wilder (default) matches TradingView's ta.atr - an RMA, i.e. an
    # EMA with alpha = 1/period"
    plot(ta.atr(14), 'atr_14')

    # rsi(): Wilder's RSI on close
    plot(ta.rsi(close, 14), 'rsi_14')

    # ema()/sma()/rma(): the three smoothing primitives
    plot(ta.ema(close, 20), 'ema_20')
    plot(ta.sma(close, 20), 'sma_20')
    plot(ta.rma(close, 14), 'rma_14')

    # macd(): (macd, signal, histogram)
    m, s, h = ta.macd(close, 12, 26, 9)
    plot(m, 'macd_12_26_9')
    plot(s, 'macd_signal_9')
    plot(h, 'macd_hist')

    # bollinger(): tvlab uses ddof=0 (population std) to match Pine's biased=True
    bb_mid, bb_up, bb_lo = ta.bb(close, 20, 2.0)
    plot(bb_mid, 'bb_mid_20')
    plot(bb_up, 'bb_upper_20')
    plot(bb_lo, 'bb_lower_20')
    plot(ta.stdev(close, 20), 'stdev_20_biased')

    # supertrend(factor, atrPeriod) - note Pine's argument order is the reverse
    # of tvlab's supertrend(period, multiplier)
    st, direction = ta.supertrend(3.0, 10)
    plot(st, 'supertrend_3_10')
    plot(direction, 'supertrend_dir_3_10')

    # adx(): Wilder ADX with +DI/-DI, whose docstring claims it "matches
    # TradingView's ta.dmi". Pine has no ta.adx, but ta.dmi returns the whole
    # triple and is the function that claim is about. tvlab uses one period for
    # both the DI length and the ADX smoothing, so the mapping is dmi(p, p) --
    # which is also TradingView's own ta.dmi() default of (14, 14).
    plus_di, minus_di, adx_ = ta.dmi(14, 14)
    plot(plus_di, 'dmi_plus_di_14')
    plot(minus_di, 'dmi_minus_di_14')
    plot(adx_, 'dmi_adx_14')

    # donchian(): tvlab shifts the bands by one bar so bar i's band excludes bar
    # i. ta.highest/lowest do NOT shift, so both are plotted and the test
    # compares tvlab's band against the *shifted* Pine value.
    plot(ta.highest(high, 20), 'hh_20')
    plot(ta.lowest(low, 20), 'll_20')

    # true range, via the ATR period-1 identity is not available, so plot the
    # pieces the tvlab true_range() formula is built from
    plot(high - low, 'hl')
    plot(ta.change(close), 'change_close')


if __name__ == "__main__":
    from pynecore.standalone import run
    run(__file__)

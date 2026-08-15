"""The data-quality engine.

Strategies must never silently operate on corrupt data. Every series that
enters the platform is graded PASS / WARNING / FAIL here, and the grade travels
with the data so a backtest report can state the quality of what it was run on.

Design notes on two checks that are easy to get wrong:

*Missing candles.* Counting them exactly requires a session calendar per venue,
which no free source provides reliably. Rather than pretend, the engine
separates two things: small gaps inside a session (1-3 bar intervals), which
are genuinely dropped bars, and large gaps, which are session breaks, weekends
and holidays. Only the former is reported as "missing"; the latter is reported
as gap statistics. Daily series use exact business-day arithmetic instead,
where holidays remain a known and documented source of over-counting.

*Price jumps.* Outlier detection uses median absolute deviation, not standard
deviation. A single bad print inflates the standard deviation enough to hide
itself; the median does not move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
import pandas as pd

from app.config.models import AssetConfig, QualityThresholds
from app.data.schema import CLOSE, HIGH, LOW, OPEN, REQUIRED_COLUMNS, VOLUME, MarketData
from app.utils.logging import get_logger
from app.utils.timeutils import Timeframe, format_utc, normalize_timeframe, utcnow

logger = get_logger(__name__)


class QualityStatus(IntEnum):
    """Ordered so that ``max()`` gives the worst status seen."""

    PASS = 0
    WARNING = 1
    FAIL = 2

    def __str__(self) -> str:
        return self.name


class DataQualityError(RuntimeError):
    """Raised when a series fails validation and the config forbids using it."""

    def __init__(self, report: DataQualityReport) -> None:
        self.report = report
        super().__init__(
            f"Data quality FAIL for {report.symbol} {report.timeframe}: "
            + "; ".join(i.message for i in report.issues if i.severity is QualityStatus.FAIL)
        )


@dataclass(frozen=True)
class QualityIssue:
    """One finding."""

    code: str
    severity: QualityStatus
    message: str
    count: int = 0
    ratio: float | None = None
    samples: tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code}: {self.message}"


@dataclass
class DataQualityReport:
    """The full verdict on one series."""

    symbol: str
    timeframe: str
    rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    timezone: str = "UTC"
    provider: str = "unknown"
    issues: list[QualityIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    generated_at: pd.Timestamp = field(default_factory=utcnow)

    # --- verdict -----------------------------------------------------------
    @property
    def status(self) -> QualityStatus:
        return max((i.severity for i in self.issues), default=QualityStatus.PASS)

    @property
    def passed(self) -> bool:
        return self.status is not QualityStatus.FAIL

    @property
    def is_clean(self) -> bool:
        return self.status is QualityStatus.PASS

    def issues_of(self, severity: QualityStatus) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity is severity]

    def add(self, issue: QualityIssue) -> None:
        self.issues.append(issue)

    # --- rendering ----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialisable form, stored alongside experiments for reproducibility."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "provider": self.provider,
            "rows": self.rows,
            "start": self.start.isoformat() if self.start is not None else None,
            "end": self.end.isoformat() if self.end is not None else None,
            "timezone": self.timezone,
            "status": str(self.status),
            "generated_at": self.generated_at.isoformat(),
            "stats": dict(self.stats),
            "issues": [
                {
                    "code": i.code,
                    "severity": str(i.severity),
                    "message": i.message,
                    "count": i.count,
                    "ratio": i.ratio,
                    "samples": list(i.samples),
                }
                for i in self.issues
            ],
        }

    def render(self) -> str:
        """Human-readable report."""
        width = 64
        lines = [
            "=" * width,
            "DATA QUALITY REPORT",
            "=" * width,
            "",
            f"Symbol:      {self.symbol}",
            f"Timeframe:   {self.timeframe}",
            f"Provider:    {self.provider}",
            f"Rows:        {self.rows:,}",
        ]
        if self.start is not None and self.end is not None:
            lines += [
                f"Start:       {format_utc(self.start)}",
                f"End:         {format_utc(self.end)}",
            ]
        lines += [f"Timezone:    {self.timezone}", ""]

        key_stats = [
            ("Missing candles", "missing_candles"),
            ("Duplicates", "duplicates"),
            ("Invalid OHLC", "invalid_ohlc"),
            ("Non-positive prices", "nonpositive_prices"),
            ("Suspected bad prints", "price_jumps"),
            ("Large gaps", "large_gaps"),
            ("Stale bars", "stale_bars"),
            ("Bars repaired", "bars_repaired"),
            ("Bars dropped", "bars_dropped_in_cleaning"),
        ]
        for label, key in key_stats:
            if key in self.stats:
                lines.append(f"{label + ':':<22}{self.stats[key]:,}")

        if "median_bar_interval" in self.stats:
            lines.append(f"{'Median interval:':<22}{self.stats['median_bar_interval']}")
        if "data_age_hours" in self.stats:
            lines.append(f"{'Data age (hours):':<22}{self.stats['data_age_hours']:.1f}")

        if self.issues:
            lines += ["", "-" * width, "FINDINGS", "-" * width]
            for severity in (QualityStatus.FAIL, QualityStatus.WARNING):
                for issue in self.issues_of(severity):
                    lines.append(f"  [{severity}] {issue.message}")
                    if issue.samples:
                        lines.append(f"           e.g. {', '.join(issue.samples[:3])}")
        else:
            lines += ["", "No issues detected."]

        lines += ["", "-" * width, f"Status: {self.status}", "=" * width]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


class DataQualityEngine:
    """Runs the check suite and grades a series."""

    def __init__(self, thresholds: QualityThresholds | None = None) -> None:
        self.thresholds = thresholds or QualityThresholds()

    def validate(
        self,
        data: MarketData,
        asset: AssetConfig | None = None,
        cleaning: Any | None = None,
    ) -> DataQualityReport:
        """Grade ``data``, optionally using ``asset`` for venue-specific rules.

        Args:
            data: the series to grade, normally already cleaned.
            asset: asset config, for venue-specific rules such as whether
                volume is meaningful for this instrument.
            cleaning: the :class:`~app.data.processors.normalize.CleaningReport`
                from cleaning this series, if any. Passing it matters: cleaning
                repairs defects *before* grading, so without this the report
                would describe the repaired series and quietly understate how
                damaged the vendor data was.
        """
        df = data.df
        tf = normalize_timeframe(data.timeframe)

        report = DataQualityReport(
            symbol=data.symbol,
            timeframe=tf.code,
            rows=len(df),
            start=data.start,
            end=data.end,
            provider=data.provider,
            timezone=str(df.index.tz) if isinstance(df.index, pd.DatetimeIndex) else "unknown",
        )

        if df.empty:
            report.add(
                QualityIssue(
                    code="empty_dataset",
                    severity=QualityStatus.FAIL,
                    message="Dataset is empty - no bars returned for the requested range",
                )
            )
            return report

        self._check_row_count(df, report)
        self._check_index(df, report)
        self._check_missing_values(df, report)
        self._check_price_validity(df, report)
        self._check_ohlc_consistency(df, report)
        self._check_gaps_and_missing(df, tf, asset, report)
        self._check_price_jumps(df, report)
        self._check_stale_bars(df, report)
        self._check_volume(df, asset, report)
        self._check_freshness(df, tf, report)
        self._check_repairs(cleaning, len(df), report)

        logger.info(
            "Data quality checked",
            extra={
                "symbol": data.symbol,
                "timeframe": tf.code,
                "status": str(report.status),
                "rows": len(df),
                "issues": len(report.issues),
            },
        )
        return report

    # ------------------------------------------------------------ individual checks

    def _check_row_count(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        n = len(df)
        t = self.thresholds
        if n < t.min_rows_fail:
            report.add(
                QualityIssue(
                    code="insufficient_history",
                    severity=QualityStatus.FAIL,
                    message=(
                        f"Only {n} bars available; at least {t.min_rows_fail} are required for "
                        "any meaningful analysis"
                    ),
                    count=n,
                )
            )
        elif n < t.min_rows_warn:
            report.add(
                QualityIssue(
                    code="short_history",
                    severity=QualityStatus.WARNING,
                    message=(
                        f"Only {n} bars available (below {t.min_rows_warn}); statistics computed "
                        "on this sample will be unreliable and backtest results should not be "
                        "trusted"
                    ),
                    count=n,
                )
            )

    def _check_index(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        if df.index.tz is None:
            report.add(
                QualityIssue(
                    code="naive_timestamps",
                    severity=QualityStatus.FAIL,
                    message="Index is timezone-naive; all timestamps must be UTC-aware",
                )
            )
        elif str(df.index.tz) != "UTC":
            report.add(
                QualityIssue(
                    code="non_utc_timezone",
                    severity=QualityStatus.WARNING,
                    message=f"Index timezone is {df.index.tz}, expected UTC",
                )
            )

        if not df.index.is_monotonic_increasing:
            report.add(
                QualityIssue(
                    code="unordered_timestamps",
                    severity=QualityStatus.FAIL,
                    message="Timestamps are not in chronological order",
                )
            )

        dupes = int(df.index.duplicated().sum())
        report.stats["duplicates"] = dupes
        if dupes:
            ratio = dupes / len(df)
            severity = self._grade(
                ratio,
                self.thresholds.max_duplicate_ratio_warn,
                self.thresholds.max_duplicate_ratio_fail,
            )
            samples = tuple(str(ts) for ts in df.index[df.index.duplicated()][:5])
            report.add(
                QualityIssue(
                    code="duplicate_timestamps",
                    severity=severity,
                    message=f"{dupes:,} duplicate timestamp(s) ({ratio:.3%} of rows)",
                    count=dupes,
                    ratio=ratio,
                    samples=samples,
                )
            )

    def _check_missing_values(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        nan_counts = {c: int(df[c].isna().sum()) for c in REQUIRED_COLUMNS}
        total = sum(nan_counts.values())
        report.stats["nan_ohlc_cells"] = total
        if total:
            worst = max(nan_counts.values()) / len(df)
            severity = self._grade(
                worst,
                self.thresholds.max_missing_ratio_warn,
                self.thresholds.max_missing_ratio_fail,
            )
            detail = ", ".join(f"{c}={n}" for c, n in nan_counts.items() if n)
            report.add(
                QualityIssue(
                    code="nan_prices",
                    severity=severity,
                    message=f"Missing price values ({detail})",
                    count=total,
                    ratio=worst,
                )
            )

    def _check_price_validity(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        prices = df[list(REQUIRED_COLUMNS)]
        nonpositive = (prices <= 0).any(axis=1) & prices.notna().all(axis=1)
        count = int(nonpositive.sum())
        report.stats["nonpositive_prices"] = count
        if count:
            ratio = count / len(df)
            report.add(
                QualityIssue(
                    code="nonpositive_prices",
                    severity=QualityStatus.FAIL,
                    message=f"{count:,} bar(s) contain a zero or negative price ({ratio:.3%})",
                    count=count,
                    ratio=ratio,
                    samples=tuple(str(ts) for ts in df.index[nonpositive][:5]),
                )
            )

        infinite = ~np.isfinite(prices.to_numpy(dtype="float64", na_value=0.0))
        inf_count = int(infinite.sum())
        if inf_count:
            report.add(
                QualityIssue(
                    code="infinite_prices",
                    severity=QualityStatus.FAIL,
                    message=f"{inf_count:,} non-finite price value(s) present",
                    count=inf_count,
                )
            )

    def _check_ohlc_consistency(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        valid = df[list(REQUIRED_COLUMNS)].notna().all(axis=1)
        sub = df[valid]
        if sub.empty:
            report.stats["invalid_ohlc"] = 0
            return

        body_high = sub[[OPEN, CLOSE]].max(axis=1)
        body_low = sub[[OPEN, CLOSE]].min(axis=1)

        inverted = sub[HIGH] < sub[LOW]
        high_too_low = sub[HIGH] < body_high
        low_too_high = sub[LOW] > body_low
        broken = inverted | high_too_low | low_too_high

        count = int(broken.sum())
        report.stats["invalid_ohlc"] = count
        if count:
            ratio = count / len(df)
            severity = self._grade(
                ratio,
                self.thresholds.max_invalid_ohlc_ratio_warn,
                self.thresholds.max_invalid_ohlc_ratio_fail,
            )
            details = []
            if int(inverted.sum()):
                details.append(f"{int(inverted.sum())} with high < low")
            if int(high_too_low.sum()):
                details.append(f"{int(high_too_low.sum())} with high below open/close")
            if int(low_too_high.sum()):
                details.append(f"{int(low_too_high.sum())} with low above open/close")
            report.add(
                QualityIssue(
                    code="invalid_ohlc",
                    severity=severity,
                    message=f"{count:,} bar(s) violate OHLC relationships ({', '.join(details)})",
                    count=count,
                    ratio=ratio,
                    samples=tuple(str(ts) for ts in sub.index[broken][:5]),
                )
            )

    def _check_gaps_and_missing(
        self,
        df: pd.DataFrame,
        tf: Timeframe,
        asset: AssetConfig | None,
        report: DataQualityReport,
    ) -> None:
        """Distinguish dropped bars inside a session from legitimate session breaks."""
        if len(df) < 3:
            return

        deltas = df.index.to_series().diff().dropna()
        median_delta = deltas.median()
        report.stats["median_bar_interval"] = str(median_delta)

        weekdays_only = asset is None or asset.trading_days == "WEEKDAYS"

        if tf.code == "1D":
            missing, gap_samples = self._missing_daily(df, weekdays_only)
        else:
            missing, gap_samples = self._missing_intraday(tf, deltas)

        report.stats["missing_candles"] = missing
        if missing:
            ratio = missing / (len(df) + missing)
            severity = self._grade(
                ratio,
                self.thresholds.max_missing_ratio_warn,
                self.thresholds.max_missing_ratio_fail,
            )
            note = (
                " (public holidays are counted here; some of these are expected)"
                if tf.code == "1D"
                else ""
            )
            report.add(
                QualityIssue(
                    code="missing_candles",
                    severity=severity,
                    message=f"{missing:,} missing bar(s) ({ratio:.3%} of the expected grid){note}",
                    count=missing,
                    ratio=ratio,
                    samples=gap_samples,
                )
            )

        # Large gaps: reported for visibility, never fatal - they are usually
        # weekends, holidays or an exchange closure.
        tolerance = median_delta * self.thresholds.gap_tolerance_multiple
        large = deltas[deltas > max(tolerance, pd.Timedelta(minutes=1))]
        report.stats["large_gaps"] = int(len(large))
        if len(large):
            biggest = large.nlargest(3)
            samples = tuple(f"{ts:%Y-%m-%d %H:%M} ({d})" for ts, d in biggest.items())
            report.add(
                QualityIssue(
                    code="large_gaps",
                    severity=QualityStatus.WARNING,
                    message=(
                        f"{len(large):,} gap(s) exceed {self.thresholds.gap_tolerance_multiple:g}x "
                        f"the median bar interval (weekends, holidays and closures look like this)"
                    ),
                    count=int(len(large)),
                    samples=samples,
                )
            )

    @staticmethod
    def _missing_daily(df: pd.DataFrame, weekdays_only: bool) -> tuple[int, tuple[str, ...]]:
        """Exact business-day arithmetic for daily series."""
        dates = df.index.tz_convert("UTC").normalize()
        start, end = dates[0], dates[-1]
        if weekdays_only:
            expected = int(
                np.busday_count(start.date(), end.date()) + 1
            )
        else:
            expected = int((end - start).days) + 1
        missing = max(0, expected - len(df.index.unique()))

        samples: tuple[str, ...] = ()
        if missing:
            freq = "B" if weekdays_only else "D"
            full = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
            absent = full.difference(dates)
            samples = tuple(f"{ts:%Y-%m-%d}" for ts in absent[:5])
        return missing, samples

    @staticmethod
    def _missing_intraday(tf: Timeframe, deltas: pd.Series) -> tuple[int, tuple[str, ...]]:
        """Count bars dropped *within* a session.

        A gap of 2-3 bar intervals is a dropped bar. Anything larger is a
        session break, and counting it as missing data would drown the report
        in false positives for any exchange-traded instrument.
        """
        expected = pd.Timedelta(minutes=tf.minutes)
        ratios = deltas / expected
        intra_session = deltas[(ratios > 1.5) & (ratios <= 3.5)]
        missing = int((ratios[(ratios > 1.5) & (ratios <= 3.5)].round() - 1).sum())
        samples = tuple(
            f"{ts:%Y-%m-%d %H:%M}" for ts in intra_session.index[:5]
        )
        return missing, samples

    def _check_price_jumps(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Flag suspected bad prints using a median-absolute-deviation z-score."""
        close = df[CLOSE].dropna()
        if len(close) < 30:
            return

        returns = np.log(close / close.shift(1)).dropna()
        if returns.empty:
            return

        median = returns.median()
        mad = (returns - median).abs().median()
        if mad <= 0:
            return

        # 1.4826 rescales MAD to a standard-deviation equivalent for normal data.
        robust_sigma = 1.4826 * mad
        z = (returns - median).abs() / robust_sigma
        outliers = z[z > self.thresholds.price_jump_sigma]

        count = int(len(outliers))
        report.stats["price_jumps"] = count
        if count:
            ratio = count / len(returns)
            severity = self._grade(
                ratio,
                self.thresholds.max_price_jump_ratio_warn,
                self.thresholds.max_price_jump_ratio_fail,
            )
            worst = outliers.nlargest(3)
            samples = tuple(
                f"{ts:%Y-%m-%d %H:%M} ({returns[ts]:+.2%}, {zv:.0f} sigma)"
                for ts, zv in worst.items()
            )
            report.add(
                QualityIssue(
                    code="price_jumps",
                    severity=severity,
                    message=(
                        f"{count:,} bar(s) move more than {self.thresholds.price_jump_sigma:g} "
                        f"robust sigma - suspected bad prints (or genuine shocks; inspect before "
                        f"discarding)"
                    ),
                    count=count,
                    ratio=ratio,
                    samples=samples,
                )
            )

    def _check_stale_bars(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Bars where O=H=L=C: a frozen feed, or a genuinely untraded period."""
        flat = (
            (df[OPEN] == df[HIGH]) & (df[HIGH] == df[LOW]) & (df[LOW] == df[CLOSE])
        )
        count = int(flat.sum())
        report.stats["stale_bars"] = count
        if not count:
            return
        ratio = count / len(df)
        if ratio > 0.05:
            report.add(
                QualityIssue(
                    code="stale_bars",
                    severity=QualityStatus.WARNING,
                    message=(
                        f"{count:,} bar(s) ({ratio:.2%}) have open=high=low=close, which usually "
                        "means a stalled feed or an illiquid period rather than real trading"
                    ),
                    count=count,
                    ratio=ratio,
                    samples=tuple(str(ts) for ts in df.index[flat][:5]),
                )
            )

    def _check_volume(
        self, df: pd.DataFrame, asset: AssetConfig | None, report: DataQualityReport
    ) -> None:
        volume = df[VOLUME]
        has_any = bool(volume.notna().any())
        report.stats["has_volume"] = has_any

        if asset is not None and not asset.has_reliable_volume:
            # Configured as unreliable: not a defect, but record it so that
            # feature engineering knows to suppress volume-based features.
            report.stats["volume_reliable"] = False
            return

        report.stats["volume_reliable"] = has_any
        if not has_any:
            report.add(
                QualityIssue(
                    code="no_volume",
                    severity=QualityStatus.WARNING,
                    message=(
                        "No volume data present, though the asset is configured as having "
                        "reliable volume. Volume-based features will be unavailable."
                    ),
                )
            )
            return

        negative = int((volume < 0).sum())
        if negative:
            report.add(
                QualityIssue(
                    code="negative_volume",
                    severity=QualityStatus.FAIL,
                    message=f"{negative:,} bar(s) report negative volume, which is impossible",
                    count=negative,
                )
            )

        zero_ratio = float((volume.fillna(0) == 0).mean())
        report.stats["zero_volume_ratio"] = zero_ratio
        if zero_ratio > 0.10:
            report.add(
                QualityIssue(
                    code="zero_volume",
                    severity=QualityStatus.WARNING,
                    message=(
                        f"{zero_ratio:.1%} of bars report zero volume; volume-derived features "
                        "will be unreliable on this series"
                    ),
                    ratio=zero_ratio,
                )
            )

    def _check_freshness(
        self, df: pd.DataFrame, tf: Timeframe, report: DataQualityReport
    ) -> None:
        """How stale the last bar is. Informational for research, critical for paper trading."""
        last = df.index[-1]
        age_hours = (utcnow() - last).total_seconds() / 3600
        report.stats["data_age_hours"] = age_hours

        # Allow for weekends plus a generous slack of several bar intervals.
        expected_max_hours = max(72.0, (tf.minutes / 60.0) * 5)
        if age_hours > expected_max_hours:
            report.add(
                QualityIssue(
                    code="stale_data",
                    severity=QualityStatus.WARNING,
                    message=(
                        f"Most recent bar is {age_hours / 24:.1f} days old "
                        f"({format_utc(last)}). Fine for historical research; not safe for "
                        "paper trading without refreshing."
                    ),
                )
            )

    def _check_repairs(
        self, cleaning: Any | None, rows: int, report: DataQualityReport
    ) -> None:
        """Report how much surgery cleaning had to perform on the vendor data.

        A series that needed thousands of repairs is not equivalent to one that
        needed none, even though both look pristine by the time they are graded.
        """
        if cleaning is None or rows == 0:
            return

        repaired = int(getattr(cleaning, "ohlc_bounds_repaired", 0))
        dropped = int(getattr(cleaning, "rows_dropped", 0))
        report.stats["bars_repaired"] = repaired
        report.stats["bars_dropped_in_cleaning"] = dropped

        if repaired:
            ratio = repaired / rows
            severity = self._grade(
                ratio,
                self.thresholds.max_repaired_ratio_warn,
                self.thresholds.max_repaired_ratio_fail,
            )
            report.add(
                QualityIssue(
                    code="repaired_bars",
                    severity=severity,
                    message=(
                        f"{repaired:,} bar(s) ({ratio:.2%}) had invalid high/low bounds that "
                        "cleaning repaired. The series is usable, but this many defects "
                        "indicates a low-quality vendor feed for this period - prefer a range "
                        "where the raw data is sound"
                    ),
                    count=repaired,
                    ratio=ratio,
                )
            )

        if dropped:
            ratio = dropped / (rows + dropped)
            severity = self._grade(
                ratio,
                self.thresholds.max_missing_ratio_warn,
                self.thresholds.max_missing_ratio_fail,
            )
            report.add(
                QualityIssue(
                    code="dropped_bars",
                    severity=severity,
                    message=(
                        f"{dropped:,} bar(s) ({ratio:.2%}) were unusable and dropped during "
                        "cleaning (duplicates, missing prices, or unrepairable OHLC)"
                    ),
                    count=dropped,
                    ratio=ratio,
                )
            )

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _grade(value: float, warn_at: float, fail_at: float) -> QualityStatus:
        if value > fail_at:
            return QualityStatus.FAIL
        if value > warn_at:
            return QualityStatus.WARNING
        return QualityStatus.PASS

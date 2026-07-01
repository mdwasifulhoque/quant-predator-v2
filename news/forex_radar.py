"""
news/forex_radar.py
===================
ForexFactory high-impact economic event detector for Quant Predator v2.

IMPROVEMENTS vs v1 (news_sentiment.py ForexNewsRadar):
    - XML response cached for 1 hour (NewsConfig.CACHE_DURATION_SECS).
      v1 fetched a fresh request on every H1 bar — risked rate-limiting.
    - Thread-safe cache using a Lock (safe if called from multiple threads).
    - Configurable currency list per symbol (not hardcoded to USD/EUR).
    - Returns a structured NewsEvent dataclass instead of a raw string.
    - Explicit timezone handling with pytz.

The news halt in live_trader.py suspends new entries (and optionally
flattens open positions) within HALT_HOURS_AHEAD of any detected event.
"""

from __future__ import annotations

import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pytz
import requests

from core.config import NewsConfig
from core.logger import get_logger

logger = get_logger(__name__)

_EST_TZ = pytz.timezone("America/New_York")

# Module-level cache — shared across all ForexNewsRadar instances
_cache_lock    = threading.Lock()
_cached_xml:   Optional[bytes] = None
_cache_time:   float = 0.0


@dataclass
class NewsEvent:
    """Represents a single upcoming high-impact economic event."""
    title:      str
    country:    str
    event_time: datetime   # Timezone-aware (America/New_York)
    impact:     str


class ForexNewsRadar:
    """
    Scans the ForexFactory economic calendar for high-impact events
    that could spike volatility and invalidate technical setups.

    Args:
        currencies: List of currency codes to monitor (e.g., ["USD", "EUR"]).
    """

    def __init__(self, currencies: List[str] = NewsConfig.EURUSD_CURRENCIES) -> None:
        self.currencies = currencies

    def fetch_imminent_news(
        self,
        hours_ahead: float = NewsConfig.HALT_HOURS_AHEAD,
    ) -> Tuple[bool, str]:
        """
        Checks whether any high-impact event is imminent for the monitored currencies.

        Uses a 1-hour XML cache to avoid hammering the ForexFactory server.

        Args:
            hours_ahead: How many hours ahead to scan for events.

        Returns:
            (is_imminent: bool, status_message: str)
            status_message is displayed in the dashboard RADAR panel.
        """
        events = self._get_upcoming_events(hours_ahead)
        if not events:
            return False, "Clear Skies"

        soonest = min(events, key=lambda e: e.event_time)
        now_est = datetime.now(_EST_TZ)
        mins    = int((soonest.event_time - now_est).total_seconds() / 60)
        msg     = (
            f"HIGH IMPACT: {soonest.country} {soonest.title} "
            f"in ~{mins} min ({soonest.event_time.strftime('%H:%M')} EST)"
        )
        logger.warning(f"News halt triggered — {msg}")
        return True, msg

    def get_todays_events(self) -> List[NewsEvent]:
        """
        Returns all high-impact events for the monitored currencies today.
        Useful for the dashboard calendar panel.
        """
        xml_content = self._fetch_cached_xml()
        if not xml_content:
            return []
        return self._parse_events(xml_content, hours_ahead=24)

    # ==========================================================================
    # PRIVATE
    # ==========================================================================

    def _get_upcoming_events(self, hours_ahead: float) -> List[NewsEvent]:
        xml_content = self._fetch_cached_xml()
        if not xml_content:
            return []
        return self._parse_events(xml_content, hours_ahead=hours_ahead)

    def _fetch_cached_xml(self) -> Optional[bytes]:
        """Returns cached XML content, refreshing if the cache has expired."""
        global _cached_xml, _cache_time

        with _cache_lock:
            age = time.time() - _cache_time
            if age < NewsConfig.CACHE_DURATION_SECS and _cached_xml is not None:
                return _cached_xml

            # Cache expired or empty — fetch fresh
            logger.info("Refreshing ForexFactory calendar cache...")
            content = self._download_xml()
            if content is not None:
                _cached_xml = content
                _cache_time = time.time()
            return _cached_xml   # Return None if download failed and cache was empty

    def _download_xml(self) -> Optional[bytes]:
        """Downloads the ForexFactory XML calendar. Returns None on failure."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        try:
            resp = requests.get(
                NewsConfig.FOREX_FACTORY_URL,
                headers=headers,
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.content
            logger.warning(f"ForexFactory returned HTTP {resp.status_code}.")
            return None
        except Exception as exc:
            logger.warning(f"ForexFactory fetch failed: {exc}")
            return None

    def _parse_events(self, xml_content: bytes, hours_ahead: float) -> List[NewsEvent]:
        """Parses XML and returns events for monitored currencies within the window."""
        events: List[NewsEvent] = []
        try:
            tree    = ET.fromstring(xml_content)
            now_est = datetime.now(_EST_TZ)
            window  = now_est + timedelta(hours=hours_ahead)

            for event in tree.findall("event"):
                impact  = (event.find("impact").text  or "").strip()
                country = (event.find("country").text or "").strip()

                if impact != "High" or country not in self.currencies:
                    continue

                date_str  = (event.find("date").text  or "").strip()
                time_str  = (event.find("time").text  or "").strip()
                title     = (event.find("title").text or "Unknown Event").strip()

                # Skip all-day and tentative events (no reliable time)
                if "All Day" in time_str or "Tentative" in time_str:
                    continue

                try:
                    event_time = _EST_TZ.localize(
                        datetime.strptime(f"{date_str} {time_str}", "%m-%d-%Y %I:%M%p")
                    )
                except ValueError:
                    continue

                if now_est <= event_time <= window:
                    events.append(
                        NewsEvent(
                            title      = title,
                            country    = country,
                            event_time = event_time,
                            impact     = impact,
                        )
                    )

        except ET.ParseError as exc:
            logger.warning(f"ForexFactory XML parse error: {exc}")

        return events

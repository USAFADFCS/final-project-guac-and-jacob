# vegas_tool.py
# Real-odds Vegas tool for fairlib-based agents.
# - Provider: The Odds API v4 (https://the-odds-api.com)
# - Requires env var: THE_ODDS_API_KEY
# - Sync + Async supported (requests for sync, aiohttp for async).
# - Exposes VegasOddsTool with .use() / .ause() plus __call__ / arun for compatibility.

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple, List
import os
import json
import re
import datetime as dt

# Dependencies:
#   pip install requests aiohttp
import requests

try:
    import aiohttp
except ImportError as e:
    raise ImportError("vegas_tool.py requires aiohttp. Install with: pip install aiohttp") from e


# ===================== Data Model =====================
@dataclass
class VegasLine:
    home: str
    away: str
    # Spread is from HOME perspective: negative means home is favored
    spread_home: Optional[float]
    total: Optional[float]
    moneyline_home: Optional[int]
    moneyline_away: Optional[int]
    # Movement is often unavailable on free plans (kept for future use)
    move_spread_home: Optional[float]
    move_total: Optional[float]
    source: str = "the-odds-api"
    as_of: str = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def to_compact(self) -> str:
        parts = [
            f"{self.away} @ {self.home}",
            f"spread_home={self.spread_home if self.spread_home is not None else 'unknown'}",
            f"total={self.total if self.total is not None else 'unknown'}",
            f"ml_home={self.moneyline_home if self.moneyline_home is not None else 'unknown'}",
            f"ml_away={self.moneyline_away if self.moneyline_away is not None else 'unknown'}",
            f"move_spread={self.move_spread_home if self.move_spread_home is not None else 'unknown'}",
            f"move_total={self.move_total if self.move_total is not None else 'unknown'}",
            f"source={self.source}",
            f"as_of={self.as_of}",
        ]
        return " | ".join(parts)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


# ===================== Helpers =====================
_VS_PAT = re.compile(r"\bvs\.?\b", flags=re.IGNORECASE)
_AT_PAT = re.compile(r"\b@\b", flags=re.IGNORECASE)

def parse_matchup_freeform(s: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse user input into (away, home), handling common formats:
      'Bills vs Jets'   -> ('Bills', 'Jets')  (treated as away vs home)
      'Broncos @ Chiefs'-> ('Broncos', 'Chiefs') (explicit away @ home)
      '49ers at Eagles' -> ('49ers', 'Eagles')
    """
    s = s.strip()
    if _AT_PAT.search(s):
        left, right = _AT_PAT.split(s, maxsplit=1)
        return left.strip(), right.strip()
    if _VS_PAT.search(s):
        left, right = _VS_PAT.split(s, maxsplit=1)
        return left.strip(), right.strip()
    if " at " in s.lower():
        left, right = re.split(r"\sat\s", s, maxsplit=1, flags=re.IGNORECASE)
        return left.strip(), right.strip()
    return None, None


def american_to_int(odd) -> Optional[int]:
    if odd is None:
        return None
    try:
        return int(odd)
    except Exception:
        try:
            return int(float(odd))
        except Exception:
            return None


# ===================== Real Provider (The Odds API) =====================
class RealOddsProvider:
    """
    The Odds API (v4) provider.
    Docs: https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds
    Requires THE_ODDS_API_KEY in environment or passed explicitly.

    Supports:
      - fetch(...)  : async (aiohttp)
      - fetch_sync: sync (requests)  -> safe to call inside an existing event loop
    """
    name = "the-odds-api"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        odds_format: str = "american",
        bookmakers: Optional[List[str]] = None,   # e.g., ["draftkings","fanduel"]
        date_format: str = "iso",
        window_hours: int = 10*24,                # look ahead for upcoming games
        session: Optional[aiohttp.ClientSession] = None,
    ):
        self.api_key = api_key or os.getenv("THE_ODDS_API_KEY")
        if not self.api_key:
            raise ValueError("Missing THE_ODDS_API_KEY in environment or passed to RealOddsProvider.")

        self.regions = regions
        self.markets = markets
        self.odds_format = odds_format
        self.bookmakers = bookmakers
        self.date_format = date_format
        self.window_hours = window_hours
        self._session = session

        # Build canonical alias map for ALL 32 NFL teams
        self.alias: Dict[str, str] = {}
        # NFC North
        self._add_alias("Detroit Lions", "Lions", "DET", "Detroit")
        self._add_alias("Chicago Bears", "Bears", "CHI", "Chicago")
        self._add_alias("Green Bay Packers", "Packers", "GB", "GBP", "Green Bay")
        self._add_alias("Minnesota Vikings", "Vikings", "MIN", "MINN", "Minnesota")
        # NFC East
        self._add_alias("Dallas Cowboys", "Cowboys", "DAL", "Dallas")
        self._add_alias("New York Giants", "Giants", "NYG", "NY Giants", "N.Y. Giants")
        self._add_alias("Philadelphia Eagles", "Eagles", "PHI", "Philly", "Philadelphia")
        self._add_alias("Washington Commanders", "Commanders", "WAS", "WSH", "Washington")
        # NFC South
        self._add_alias("Atlanta Falcons", "Falcons", "ATL", "Atlanta")
        self._add_alias("Carolina Panthers", "Panthers", "CAR", "Carolina")
        self._add_alias("New Orleans Saints", "Saints", "NO", "NOS", "NOLA", "New Orleans")
        self._add_alias("Tampa Bay Buccaneers", "Buccaneers", "Bucs", "TB", "TBB", "Tampa Bay")
        # NFC West
        self._add_alias("Arizona Cardinals", "Cardinals", "Cards", "ARI", "ARZ", "Arizona")
        self._add_alias("Los Angeles Rams", "Rams", "LAR", "LA Rams", "Los Angeles Rams")
        self._add_alias("San Francisco 49ers", "49ers", "Niners", "SF", "SFO", "San Francisco")
        self._add_alias("Seattle Seahawks", "Seahawks", "SEA", "Seattle")
        # AFC North
        self._add_alias("Baltimore Ravens", "Ravens", "BAL", "Baltimore")
        self._add_alias("Cincinnati Bengals", "Bengals", "CIN", "Cincy", "Cincinnati")
        self._add_alias("Cleveland Browns", "Browns", "CLE", "Cleveland")
        self._add_alias("Pittsburgh Steelers", "Steelers", "PIT", "Pittsburgh")
        # AFC East
        self._add_alias("Buffalo Bills", "Bills", "BUF", "Buffalo")
        self._add_alias("Miami Dolphins", "Dolphins", "MIA", "Miami")
        self._add_alias("New England Patriots", "Patriots", "Pats", "NE", "N.E.", "New England")
        self._add_alias("New York Jets", "Jets", "NYJ", "NY Jets", "N.Y. Jets")
        # AFC South
        self._add_alias("Houston Texans", "Texans", "HOU", "Houston")
        self._add_alias("Indianapolis Colts", "Colts", "IND", "Indianapolis")
        self._add_alias("Jacksonville Jaguars", "Jaguars", "Jags", "JAX", "JAC", "Jacksonville")
        self._add_alias("Tennessee Titans", "Titans", "TEN", "Tennessee")
        # AFC West
        self._add_alias("Denver Broncos", "Broncos", "DEN", "Denver")
        self._add_alias("Kansas City Chiefs", "Chiefs", "KC", "KCC", "Kansas City")
        self._add_alias("Las Vegas Raiders", "Raiders", "LV", "LVR", "Vegas", "Las Vegas")
        self._add_alias("Los Angeles Chargers", "Chargers", "LAC", "LA Chargers", "Los Angeles Chargers")

    # ----- alias helpers -----
    def _add_alias(self, canon: str, *alts: str):
        # Map each alias (lowercased) to the canonical name
        for a in (canon, *alts):
            self.alias[a.lower()] = canon

    def _canon(self, s: str) -> Optional[str]:
        # Return canonical if known; otherwise keep as-is for lenient matching
        return self.alias.get(s.strip().lower(), s)

    # ----- async session management -----
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session and not self._session.closed:
            return self._session
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=12))
        return self._session

    async def aclose(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ===================== ASYNC FETCH =====================
    async def fetch(self, home: str, away: str) -> Optional[VegasLine]:
        """
        Async fetch with aiohttp. Use from async code paths (.ause / .arun).
        """
        H, A = self._canon(home), self._canon(away)

        now = dt.datetime.utcnow()
        start = now.isoformat(timespec="seconds") + "Z"
        stop = (now + dt.timedelta(hours=self.window_hours)).isoformat(timespec="seconds") + "Z"

        params = {
            "apiKey": self.api_key,
            "regions": self.regions,
            "markets": self.markets,         # h2h,spreads,totals
            "oddsFormat": self.odds_format,  # american
            "dateFormat": self.date_format,  # iso
            "commenceTimeFrom": start,
            "commenceTimeTo": stop,
        }
        if self.bookmakers:
            params["bookmakers"] = ",".join(self.bookmakers)

        url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"

        ses = await self._get_session()
        async with ses.get(url, params=params) as r:
            if r.status != 200:
                txt = await r.text()
                raise RuntimeError(f"The Odds API error {r.status}: {txt}")
            events = await r.json()

        return self._extract_event_line(events, H, A)

    # ===================== SYNC FETCH =====================
    def fetch_sync(self, home: str, away: str) -> Optional[VegasLine]:
        """
        Sync fetch with requests. Safe to call from within an already-running event loop.
        """
        H, A = self._canon(home), self._canon(away)

        now = dt.datetime.utcnow()
        start = now.isoformat(timespec="seconds") + "Z"
        stop = (now + dt.timedelta(hours=self.window_hours)).isoformat(timespec="seconds") + "Z"

        params = {
            "apiKey": self.api_key,
            "regions": self.regions,
            "markets": self.markets,
            "oddsFormat": self.odds_format,
            "dateFormat": self.date_format,
            "commenceTimeFrom": start,
            "commenceTimeTo": stop,
        }
        if self.bookmakers:
            params["bookmakers"] = ",".join(self.bookmakers)

        url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
        r = requests.get(url, params=params, timeout=12)
        if r.status_code != 200:
            raise RuntimeError(f"The Odds API error {r.status_code}: {r.text}")
        events = r.json()

        return self._extract_event_line(events, H, A)

    # ----- shared extractor -----
    def _extract_event_line(self, events: List[dict], H: str, A: str) -> Optional[VegasLine]:
        """
        Find the matching event, select a bookmaker (respect preference if provided),
        and extract moneyline, spread (home perspective), and total (Over).
        """
        if not events:
            return None

        target = None
        # Try exact orientation
        for ev in events:
            ev_home = self._canon(ev.get("home_team", ""))
            ev_away = self._canon(ev.get("away_team", ""))
            if ev_home and ev_away and ev_home.lower() == H.lower() and ev_away.lower() == A.lower():
                target = ev
                break
        # Try swapped (user may have reversed)
        if not target:
            for ev in events:
                ev_home = self._canon(ev.get("home_team", ""))
                ev_away = self._canon(ev.get("away_team", ""))
                if ev_home and ev_away and ev_home.lower() == A.lower() and ev_away.lower() == H.lower():
                    target = ev
                    H, A = ev_home, ev_away
                    break

        if not target:
            return None

        bks = target.get("bookmakers", []) or []
        if not bks:
            return None

        # Choose bookmaker: use preference list if specified; else first available
        if self.bookmakers:
            book = None
            pref_set = set(self.bookmakers)
            for bk in bks:
                if bk.get("key") in pref_set:
                    book = bk
                    break
            if book is None:
                book = bks[0]
        else:
            book = bks[0]

        spread_home: Optional[float] = None
        total: Optional[float] = None
        ml_home: Optional[int] = None
        ml_away: Optional[int] = None

        for mk in book.get("markets", []):
            key = mk.get("key")
            if key == "h2h":
                # Moneyline prices for home/away
                for oc in mk.get("outcomes", []):
                    name = self._canon(oc.get("name", ""))
                    price = american_to_int(oc.get("price"))
                    if name and price is not None:
                        if name.lower() == H.lower():
                            ml_home = price
                        elif name.lower() == A.lower():
                            ml_away = price

            elif key == "spreads":
                home_point = None
                away_point = None
                for oc in mk.get("outcomes", []):
                    name = self._canon(oc.get("name", ""))
                    pt = oc.get("point")
                    if name and pt is not None:
                        if name.lower() == H.lower():
                            home_point = float(pt)
                        elif name.lower() == A.lower():
                            away_point = float(pt)
                if home_point is None and away_point is not None:
                    home_point = -float(away_point)
                spread_home = home_point if home_point is not None else spread_home

            elif key == "totals":
                # Use Over point as the game total
                over_point = None
                for oc in mk.get("outcomes", []):
                    if str(oc.get("name", "")).lower() == "over":
                        p = oc.get("point")
                        if p is not None:
                            over_point = float(p)
                            break
                total = over_point if over_point is not None else total

        return VegasLine(
            home=H,
            away=A,
            spread_home=spread_home,
            total=total,
            moneyline_home=ml_home,
            moneyline_away=ml_away,
            move_spread_home=None,  # Historical movement needs paid history endpoints
            move_total=None,
            source=f"{self.name}:{book.get('key')}",
        )


# ===================== Tool (fairlib-compatible) =====================
class VegasOddsTool:
    """
    Fairlib-compatible tool using The Odds API.

    Inputs:
      - Free text like "Broncos @ Chiefs" or "Bills vs Jets"
      - JSON string: {"home":"...","away":"...","format":"json|text"}

    Outputs:
      - compact text (default) or JSON (if "format":"json")
    """
    name = "vegas_odds"
    description = (
        "Retrieve real Vegas lines (moneyline, spread, total) for an NFL matchup using The Odds API. "
        "Accepts free text (e.g., 'Bills vs Jets', 'Broncos @ Chiefs') or JSON "
        '{"home":"...","away":"...","format":"json|text"}. Requires THE_ODDS_API_KEY.'
    )

    def __init__(self, provider: RealOddsProvider):
        self.provider = provider

    # ===== fairlib-style sync entrypoint (safe in running event loop) =====
    def use(self, query: str) -> str:
        data = None
        try:
            data = json.loads(query)
        except Exception:
            pass

        if data and isinstance(data, dict):
            home = data.get("home")
            away = data.get("away")
            out_fmt = data.get("format", "text")
        else:
            away, home = parse_matchup_freeform(query)  # returns (away, home)
            out_fmt = "text"

        if not home or not away:
            return "unknown: could not parse teams (expect 'Away @ Home' or 'Away vs Home' or JSON with home/away)."

        try:
            line = self.provider.fetch_sync(home, away)
        except Exception as e:
            return f"error: {type(e).__name__}: {e}"

        if not line:
            return "unknown: no line available for requested matchup."
        return line.to_json() if str(out_fmt).lower() == "json" else line.to_compact()

    # ===== fairlib-style async entrypoint =====
    async def ause(self, query: str) -> str:
        data = None
        try:
            data = json.loads(query)
        except Exception:
            pass

        if data and isinstance(data, dict):
            home = data.get("home")
            away = data.get("away")
            out_fmt = data.get("format", "text")
        else:
            away, home = parse_matchup_freeform(query)
            out_fmt = "text"

        if not home or not away:
            return "unknown: could not parse teams (expect 'Away @ Home' or 'Away vs Home' or JSON with home/away)."

        try:
            line = await self.provider.fetch(home, away)
        except Exception as e:
            return f"error: {type(e).__name__}: {e}"

        if not line:
            return "unknown: no line available for requested matchup."
        return line.to_json() if str(out_fmt).lower() == "json" else line.to_compact()

    # ===== compatibility with other executors =====
    def __call__(self, query: str) -> str:
        return self.use(query)

    async def arun(self, query: str) -> str:
        return await self.ause(query)

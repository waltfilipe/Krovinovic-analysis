"""Player profile enrichment (photo, height, foot, age) via TheSportsDB + Wikidata."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_PATH = ROOT / "data" / "player_profiles_cache.json"
THESPORTSDB_SEARCH = "https://www.thesportsdb.com/api/v1/json/3/searchplayers.php"
THESPORTSDB_LOOKUP = "https://www.thesportsdb.com/api/v1/json/3/lookupplayer.php"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_ENTITY = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
REQUEST_TIMEOUT_SEC = 8
USER_AGENT = "midfielders-passers/1.0"
MIN_PLAYER_AGE = 16
MAX_PLAYER_AGE = 42

GENERAL_PROFILE_LABELS: dict[str, str] = {
    "minutes": "Minutes played",
    "age": "Age",
    "height": "Height",
    "dominant_foot": "Dominant foot",
    "nationality": "Nationality",
}

GENERAL_PROFILE_KEYS: tuple[str, ...] = tuple(GENERAL_PROFILE_LABELS.keys())

PASS_TRADITIONAL_PARTICIPATION_KEYS: tuple[str, ...] = (
    "passes_total",
    "long_balls",
    "progressive_passes",
    "final_third_passes",
    "passes_to_box",
    "key_passes",
    "crosses_total",
)

CARRY_TRADITIONAL_PARTICIPATION_KEYS: tuple[str, ...] = (
    "carry_progressive_carries",
    "very_progressive_carries",
    "dribbles_success",
    "dribbles_final_third",
)

# Normalize common European-club labels before matching TheSportsDB teams.
TEAM_ALIASES: dict[str, str] = {
    "ssc napoli": "napoli",
    "fc barcelona": "barcelona",
    "fc bayern munchen": "bayern munich",
    "fc bayern münchen": "bayern munich",
    "bayern 04 leverkusen": "bayer leverkusen",
    "bayer 04 leverkusen": "bayer leverkusen",
    "rb leipzig": "rasenballsport leipzig",
    "manchester utd": "manchester united",
    "man utd": "manchester united",
    "man united": "manchester united",
    "newcastle utd": "newcastle united",
    "tottenham hotspur": "tottenham",
    "wolverhampton wanderers": "wolves",
    "wolverhampton": "wolves",
    "west ham united": "west ham",
    "brighton hove albion": "brighton",
    "brighton and hove albion": "brighton",
    "nottingham forest": "nottm forest",
    "leeds united": "leeds",
    "inter milan": "inter",
    "ac milan": "milan",
    "as roma": "roma",
    "hellas verona": "verona",
    "us lecce": "lecce",
    "us sassuolo": "sassuolo",
    "us cremonese": "cremonese",
    "atalanta bc": "atalanta",
    "ssc bari": "bari",
    "real madrid cf": "real madrid",
    "atletico madrid": "atletico de madrid",
    "athletic bilbao": "athletic club",
    "real betis balompie": "real betis",
    "rayo vallecano": "rayo vallecano de madrid",
    "borussia dortmund": "dortmund",
    "borussia monchengladbach": "monchengladbach",
    "eintracht frankfurt": "frankfurt",
    "paris saint germain": "paris saint germain",
    "olympique lyonnais": "lyon",
    "olympique de marseille": "marseille",
    "as monaco": "monaco",
    "ogc nice": "nice",
    "rc lens": "lens",
    "stade rennais": "rennes",
    "stade brestois": "brest",
    "fc nantes": "nantes",
    "rc strasbourg": "strasbourg",
    "saint etienne": "saint etienne",
    "le havre ac": "le havre",
    "paris fc": "paris fc",
    "red star fc": "red star",
    "stade de reims": "reims",
    "montpellier hsc": "montpellier",
}


def _normalize_team(value: str | None) -> str:
    if not value:
        return ""
    text = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
    text = re.sub(r"\s+", " ", text)
    return TEAM_ALIASES.get(text, text)


def _normalize_name(value: str | None) -> str:
    if not value:
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _team_match_score(candidate_team: str | None, target_team: str | None) -> float:
    cand = _normalize_team(candidate_team)
    target = _normalize_team(target_team)
    if not cand or not target:
        return 0.0
    if cand == target:
        return 1.0
    if cand in target or target in cand:
        return 0.85
    cand_tokens = set(cand.split())
    target_tokens = set(target.split())
    if not cand_tokens or not target_tokens:
        return 0.0
    overlap = len(cand_tokens & target_tokens) / max(len(target_tokens), 1)
    return overlap


def _name_match_score(row_name: str, target_name: str) -> float:
    if not target_name:
        return 0.0
    if row_name == target_name:
        return 1.0
    if target_name in row_name or row_name in target_name:
        return 0.8
    target_tokens = set(target_name.split())
    row_tokens = set(row_name.split())
    if not target_tokens or not row_tokens:
        return 0.0
    return len(target_tokens & row_tokens) / len(target_tokens)


def _age_from_birthdate(value: str | None) -> int | None:
    if not value:
        return None
    text = str(value).strip()
    if text.startswith("+"):
        text = text[1:]
    try:
        born = datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None
    today = datetime.now(timezone.utc).date()
    age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    if age < MIN_PLAYER_AGE or age > MAX_PLAYER_AGE:
        return None
    return age


def _birthdate_iso(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    if text.startswith("+"):
        text = text[1:]
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date().isoformat()
    except ValueError:
        return None


def _is_soccer_player(row: dict) -> bool:
    sport = str(row.get("strSport", "")).strip().lower()
    return sport in {"soccer", "football"}


def _profile_has_data(profile: dict) -> bool:
    return any(
        profile.get(key) is not None
        for key in ("age", "photo_url", "height", "dominant_foot", "nationality")
    )


def _cached_age_value(profile: dict) -> int | None:
    age = _age_from_birthdate(profile.get("date_of_birth"))
    if age is not None:
        return age
    try:
        cached_age = profile.get("age")
        return int(cached_age) if cached_age is not None else None
    except (TypeError, ValueError):
        return None


def _should_use_cached(cached: dict | None, *, force: bool) -> bool:
    if force or not isinstance(cached, dict):
        return False
    if _cached_age_value(cached) is not None:
        return True
    # Metadata without a usable age should be retried (bad DOB, Wikidata not tried yet).
    if cached.get("nationality") or cached.get("photo_url"):
        return False
    return cached.get("fetch_status") == "not_found"


def _http_json(url: str) -> dict | list | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None


def _load_cache() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    try:
        raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _save_cache(cache: dict[str, dict]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _pick_search_result(results: list[dict], *, player_name: str, team: str) -> dict | None:
    if not results:
        return None
    target_name = _normalize_name(player_name)
    scored: list[tuple[float, dict]] = []
    for row in results:
        if not _is_soccer_player(row):
            continue
        row_name = _normalize_name(row.get("strPlayer"))
        name_score = _name_match_score(row_name, target_name)
        team_score = _team_match_score(row.get("strTeam"), team)
        age = _age_from_birthdate(row.get("dateBorn"))
        age_penalty = 0.0
        if age is None and row.get("dateBorn"):
            age_penalty = 0.25
        # Short names (Pedri, Rodri) need a stronger team signal to avoid collisions.
        min_team = 0.55 if len(target_name.split()) == 1 else 0.2
        if team_score < min_team and name_score < 0.95:
            continue
        scored.append((name_score * 0.65 + team_score * 0.35 - age_penalty, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return None
    best_score, best_row = scored[0]
    return best_row if best_score >= 0.45 else None


def _search_thesportsdb(query: str) -> list[dict]:
    payload = _http_json(f"{THESPORTSDB_SEARCH}?p={urllib.parse.quote(query)}")
    if not isinstance(payload, dict):
        return []
    results = payload.get("player")
    return results if isinstance(results, list) else []


def _profile_from_thesportsdb_row(picked: dict) -> dict:
    player_id = picked.get("idPlayer")
    detail = picked
    if player_id:
        lookup = _http_json(f"{THESPORTSDB_LOOKUP}?id={urllib.parse.quote(str(player_id))}")
        if isinstance(lookup, dict):
            players = lookup.get("players")
            if isinstance(players, list) and players:
                detail = players[0]

    photo = (
        detail.get("strCutout")
        or detail.get("strThumb")
        or picked.get("strCutout")
        or picked.get("strThumb")
    )
    date_born = detail.get("dateBorn") or picked.get("dateBorn")
    return {
        "photo_url": str(photo) if photo else None,
        "height": str(detail.get("strHeight")).strip() if detail.get("strHeight") else None,
        "dominant_foot": str(detail.get("strSide")).strip() if detail.get("strSide") else None,
        "nationality": str(detail.get("strNationality")).strip() if detail.get("strNationality") else None,
        "date_of_birth": _birthdate_iso(date_born),
        "age": _age_from_birthdate(date_born),
        "thesportsdb_id": str(player_id) if player_id else None,
        "source": "thesportsdb",
    }


def _fetch_profile_from_thesportsdb(player_name: str, team: str) -> dict:
    queries = [player_name]
    if team:
        queries.append(f"{player_name} {team}")
    seen_ids: set[str] = set()
    for query in queries:
        for picked in _search_thesportsdb(query):
            player_id = str(picked.get("idPlayer") or "")
            if player_id and player_id in seen_ids:
                continue
            if player_id:
                seen_ids.add(player_id)
            if not _is_soccer_player(picked):
                continue
            candidate = _pick_search_result([picked], player_name=player_name, team=team)
            if not candidate:
                continue
            profile = _profile_from_thesportsdb_row(candidate)
            if profile.get("age") is not None or profile.get("nationality"):
                return profile
    return {}


def _wikipedia_search(query: str, *, limit: int = 5) -> list[dict]:
    url = (
        f"{WIKIPEDIA_API}?{urllib.parse.urlencode({
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'format': 'json',
            'srlimit': limit,
        })}"
    )
    payload = _http_json(url)
    if not isinstance(payload, dict):
        return []
    query_block = payload.get("query")
    if not isinstance(query_block, dict):
        return []
    results = query_block.get("search")
    return results if isinstance(results, list) else []


def _wikidata_label(entity_id: str | None) -> str | None:
    if not entity_id:
        return None
    payload = _http_json(WIKIDATA_ENTITY.format(qid=entity_id))
    if not isinstance(payload, dict):
        return None
    entities = payload.get("entities")
    if not isinstance(entities, dict):
        return None
    entity = entities.get(entity_id)
    if not isinstance(entity, dict):
        return None
    labels = entity.get("labels")
    if isinstance(labels, dict):
        for key in ("en", "pt", "es", "de", "it"):
            label = labels.get(key)
            if isinstance(label, dict) and label.get("value"):
                return str(label["value"])
    return None


def _wikidata_profile_from_title(title: str) -> dict:
    url = (
        f"{WIKIPEDIA_API}?{urllib.parse.urlencode({
            'action': 'query',
            'prop': 'pageprops',
            'titles': title,
            'ppprop': 'wikibase_item',
            'format': 'json',
        })}"
    )
    payload = _http_json(url)
    if not isinstance(payload, dict):
        return {}
    pages = payload.get("query", {}).get("pages")
    if not isinstance(pages, dict):
        return {}
    page = next(iter(pages.values()), {})
    qid = page.get("pageprops", {}).get("wikibase_item")
    if not qid:
        return {}

    entity_payload = _http_json(WIKIDATA_ENTITY.format(qid=qid))
    if not isinstance(entity_payload, dict):
        return {}
    entity = entity_payload.get("entities", {}).get(qid)
    if not isinstance(entity, dict):
        return {}

    claims = entity.get("claims", {})
    dob_claim = (claims.get("P569") or [{}])[0]
    dob_value = dob_claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
    date_born = dob_value.get("time")
    nat_claim = (claims.get("P27") or [{}])[0]
    nat_id = nat_claim.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
    height_claim = (claims.get("P2048") or [{}])[0]
    height_amount = height_claim.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("amount")
    height_txt = None
    if height_amount is not None:
        try:
            height_txt = f"{float(str(height_amount).lstrip('+')):.0f} cm"
        except ValueError:
            height_txt = None

    return {
        "photo_url": None,
        "height": height_txt,
        "dominant_foot": None,
        "nationality": _wikidata_label(nat_id),
        "date_of_birth": _birthdate_iso(date_born),
        "age": _age_from_birthdate(date_born),
        "wikidata_id": qid,
        "source": "wikidata",
    }


def _title_looks_like_footballer(title: str, snippet: str = "") -> bool:
    text = f"{title} {snippet}".lower()
    if "footballer" in text or "soccer" in text:
        return True
    if any(token in text for token in (" fc", "f.c.", "united", "city", "madrid", "milan")):
        return True
    return False


def _fetch_profile_from_wikidata(player_name: str, team: str) -> dict:
    queries = [
        f"{player_name} {team} footballer",
        f"{player_name} footballer",
        player_name,
    ]
    seen_titles: set[str] = set()
    target_name = _normalize_name(player_name)
    for query in queries:
        for hit in _wikipedia_search(query):
            title = str(hit.get("title") or "").strip()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            snippet = str(hit.get("snippet") or "")
            title_name = _normalize_name(title.split("(")[0])
            if _name_match_score(title_name, target_name) < 0.5 and not _title_looks_like_footballer(title, snippet):
                continue
            profile = _wikidata_profile_from_title(title)
            if profile.get("age") is not None:
                return profile
    return {}


def _merge_profiles(base: dict, extra: dict) -> dict:
    out = dict(base)
    if out.get("age") is None and _age_from_birthdate(out.get("date_of_birth")) is None:
        out.pop("date_of_birth", None)
    for key, value in extra.items():
        if value is not None and out.get(key) is None:
            out[key] = value
    if out.get("date_of_birth"):
        out["age"] = _age_from_birthdate(out["date_of_birth"])
    return out


def get_player_profile(
    player_id: str,
    player_name: str,
    team: str,
    *,
    force: bool = False,
) -> dict:
    """Return cached or freshly fetched profile fields for a player."""
    pid = str(player_id or "").strip()
    cache = _load_cache()
    cached = cache.get(pid) if pid else None
    if _should_use_cached(cached, force=force):
        return dict(cached)

    profile: dict = {
        "photo_url": None,
        "height": None,
        "dominant_foot": None,
        "nationality": None,
        "date_of_birth": None,
        "age": None,
        "thesportsdb_id": None,
        "wikidata_id": None,
        "source": None,
        "resolved": True,
        "fetch_status": "not_found",
    }

    if player_name:
        sportsdb = _fetch_profile_from_thesportsdb(player_name, team)
        profile = _merge_profiles(profile, sportsdb)
        if profile.get("age") is None:
            wikidata = _fetch_profile_from_wikidata(player_name, team)
            profile = _merge_profiles(profile, wikidata)

    if _profile_has_data(profile):
        profile["fetch_status"] = "ok"
    else:
        profile["fetch_status"] = "not_found"

    if pid:
        cache[pid] = profile
        _save_cache(cache)
    return profile


def read_cached_profile(player_id: str) -> dict:
    """Return the cached profile for a player without any network call."""
    pid = str(player_id or "").strip()
    if not pid:
        return {}
    cached = _load_cache().get(pid)
    return dict(cached) if isinstance(cached, dict) else {}


def read_cached_age(player_id: str) -> int | None:
    """Cached age (years) or None. No network."""
    return _cached_age_value(read_cached_profile(player_id))


def enrich_player_general_profile(player: dict, *, force: bool = False) -> dict:
    """Attach general profile fields onto a player dict (non-destructive)."""
    out = dict(player)
    profile = get_player_profile(
        str(player.get("player_id", "")),
        str(player.get("player_name", "")),
        str(player.get("team", "")),
        force=force,
    )
    for key in GENERAL_PROFILE_KEYS:
        if key in {"minutes", "minutes_pct"}:
            continue
        value = profile.get(key)
        if value is not None:
            out[key] = value
    return out

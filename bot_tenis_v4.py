# ============================================================
# BOT TENIS PRO IA v7.0 — MODO ESCALERA
# ============================================================
# HEREDADO v6:
# ✅ Arquitectura modular con clases
# ✅ Logging profesional con rotación de archivos
# ✅ Reintentos con backoff exponencial
# ✅ Caché con TTL configurable
# ✅ Múltiples bookmakers (mejor cuota)
# ✅ Kelly Criterion para sizing
# ✅ Validación de features
# ✅ Registro de picks en CSV (backtesting)
# ✅ Rate limiting
# ✅ Configuración centralizada via dataclasses
# ✅ Todas las superficies (no solo Grand Slams)
#
# NUEVO v7 — ESCALERA:
# ✅ Solo partidos del DÍA SIGUIENTE (fecha exacta UTC+0 ajustable)
# ✅ Filtro de cuotas escalera: 1.35 – 1.70
# ✅ Análisis estadístico completo sobre candidatos filtrados
# ✅ Resumen diario al final con total de picks encontrados
# ============================================================

import os
import re
import csv
import time
import logging
import pickle
import hashlib
import requests
import pandas as pd

from datetime import datetime, timedelta
from dataclasses import dataclass, field
from functools import wraps
from typing import Optional
from bs4 import BeautifulSoup
from logging.handlers import RotatingFileHandler

# ============================================================
# CONFIGURACIÓN CENTRALIZADA
# ============================================================

@dataclass
class Config:
    odds_api_key: str = field(
        default_factory=lambda: os.environ.get("ODDS_API_KEY", "")
    )
    telegram_token: str = field(
        default_factory=lambda: os.environ.get("TELEGRAM_TOKEN", "")
    )
    telegram_chat_id: str = field(
        default_factory=lambda: os.environ.get("TELEGRAM_CHAT_ID", "")
    )

    modelo_path: str = "modelo_tenis_v5.pkl"
    picks_log_path: str = "picks_log.csv"

    # ── Filtro de fecha ─────────────────────────────────────
    # Corre el bot cada noche y solo analiza partidos
    # cuya fecha de inicio sea exactamente el día siguiente.
    # Ajusta utc_offset_horas si tu zona horaria difiere de UTC.
    # Ejemplos: Colombia = -5, Argentina = -3, España = +1/+2
    utc_offset_horas: int = -5  # Colombia (UTC-5)

    # ── Rango de cuotas ESCALERA ────────────────────────────
    cuota_escalera_min: float = 1.35
    cuota_escalera_max: float = 1.70

    # ── Umbrales de value betting ───────────────────────────
    value_threshold: float = 1.04   # Edge mínimo sobre cuota justa
    min_prob: float = 0.59          # ~59% prob implícita en cuota 1.70
    kelly_fraccion: float = 0.25    # Kelly fraccionado (conservador)
    bankroll: float = 1000.0        # Bankroll base para Kelly

    # Red
    request_timeout: int = 15
    max_retries: int = 3
    retry_backoff: float = 2.0     # Segundos base para backoff

    # Caché
    cache_ttl_seconds: int = 3600  # 1 hora
    rate_limit_delay: float = 1.5  # Segundos entre requests al scraper

    # Competiciones
    ligas: list = field(default_factory=lambda: [
        # ── ATP ──────────────────────────────────────────────
        "tennis_atp_french_open",
        "tennis_atp_wimbledon",
        "tennis_atp_us_open",
        "tennis_atp_aus_open_singles",
        "tennis_atp_indian_wells",
        "tennis_atp_miami_open",
        "tennis_atp_madrid_open",
        "tennis_atp_italian_open",
        "tennis_atp_canadian_open",
        "tennis_atp_cincinnati_open",
        "tennis_atp_shanghai_masters",
        "tennis_atp_paris_masters",
        "tennis_atp_monte_carlo_masters",
        "tennis_atp_barcelona_open",
        "tennis_atp_hamburg_open",
        "tennis_atp_dubai",
        "tennis_atp_qatar_open",
        "tennis_atp_munich",
        "tennis_atp_china_open",
        # ── WTA ──────────────────────────────────────────────
        "tennis_wta_french_open",
        "tennis_wta_wimbledon",
        "tennis_wta_us_open",
        "tennis_wta_aus_open_singles",
        "tennis_wta_indian_wells",
        "tennis_wta_miami_open",
        "tennis_wta_madrid_open",
        "tennis_wta_italian_open",
        "tennis_wta_canadian_open",
        "tennis_wta_cincinnati_open",
        "tennis_wta_dubai",
        "tennis_wta_qatar_open",
        "tennis_wta_charleston_open",
        "tennis_wta_strasbourg",
        "tennis_wta_stuttgart_open",
        "tennis_wta_china_open",
        "tennis_wta_wuhan_open",
    ])
    regions: str = "eu"

    # Superficies de torneos conocidos
    torneo_superficie: dict = field(default_factory=lambda: {
        "australian open": "hard",
        "roland garros": "clay",
        "french open": "clay",
        "wimbledon": "grass",
        "us open": "hard",
        "madrid open": "clay",
        "monte carlo": "clay",
        "rome": "clay",
        "miami open": "hard",
        "indian wells": "hard",
        "canada": "hard",
        "cincinnati": "hard",
        "paris masters": "hard",
        "dubai": "hard",
        "doha": "hard",
        "halle": "grass",
        "queen's": "grass",
        "eastbourne": "grass",
        "stuttgart": "grass",
    })

CFG = Config()

# ============================================================
# LOGGING
# ============================================================

def setup_logger(name: str = "bot_tenis") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Consola
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # Archivo rotativo (5 MB x 3 backups)
    fh = RotatingFileHandler(
        "bot_tenis.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

log = setup_logger()

# ============================================================
# UTILIDADES
# ============================================================

def retry(max_retries: int = CFG.max_retries, backoff: float = CFG.retry_backoff):
    """Decorador de reintentos con backoff exponencial."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for intento in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    wait = backoff ** intento
                    if intento == max_retries:
                        log.error(f"{func.__name__} falló tras {max_retries} intentos: {e}")
                        raise
                    log.warning(f"{func.__name__} intento {intento}/{max_retries} fallido: {e}. Reintentando en {wait}s…")
                    time.sleep(wait)
        return wrapper
    return decorator


def kelly_criterion(prob: float, cuota: float, fraccion: float = CFG.kelly_fraccion) -> float:
    """Calcula el % del bankroll a apostar según Kelly fraccionado."""
    b = cuota - 1
    q = 1 - prob
    kelly = (b * prob - q) / b
    return max(0.0, round(kelly * fraccion, 4))


def prob_implicita(cuota: float) -> float:
    return round(1 / cuota, 4) if cuota > 1 else 0.0


def detectar_superficie(torneo: str) -> str:
    torneo_lower = torneo.lower()
    for nombre, sup in CFG.torneo_superficie.items():
        if nombre in torneo_lower:
            return sup
    return "hard"  # Default estadístico más común


def fecha_manana_local() -> datetime.date:
    """Retorna la fecha de mañana según el offset de zona horaria configurado."""
    ahora_local = datetime.utcnow() + timedelta(hours=CFG.utc_offset_horas)
    return (ahora_local + timedelta(days=1)).date()


def es_partido_de_manana(commence_time_str: str) -> bool:
    """
    Valida que el partido empiece exactamente el día de mañana (hora local).
    commence_time_str viene en formato ISO 8601 de la Odds API, ej:
    '2025-06-15T14:00:00Z'
    """
    try:
        # La Odds API siempre retorna UTC (sufijo Z)
        dt_utc = datetime.strptime(commence_time_str, "%Y-%m-%dT%H:%M:%SZ")
        dt_local = dt_utc + timedelta(hours=CFG.utc_offset_horas)
        return dt_local.date() == fecha_manana_local()
    except Exception:
        return False


def hash_cache(key: str) -> str:
    return hashlib.md5(key.encode()).hexdigest()

# ============================================================
# CACHÉ CON TTL
# ============================================================

class CacheTTL:
    def __init__(self, ttl: int = CFG.cache_ttl_seconds):
        self._store: dict = {}
        self._ttl = ttl

    def get(self, key: str):
        entry = self._store.get(key)
        if entry is None:
            return None
        valor, ts = entry
        if time.time() - ts > self._ttl:
            del self._store[key]
            return None
        return valor

    def set(self, key: str, valor):
        self._store[key] = (valor, time.time())

    def clear(self):
        self._store.clear()

cache_jugadores = CacheTTL()

# ============================================================
# MODELO IA
# ============================================================

class ModeloIA:
    FEATURE_COLUMNS = [
        "delta_elo", "delta_elo_surface", "delta_hold", "delta_break",
        "delta_dr", "delta_forma", "delta_streak", "delta_tb",
        "delta_h2h", "delta_rank", "delta_fatiga", "delta_market_prob",
        "surface"
    ]

    def __init__(self, path: str = CFG.modelo_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Modelo no encontrado: {path}")
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        log.info(f"✅ Modelo cargado desde {path}")

    def predecir(self, df: pd.DataFrame) -> tuple[float, float]:
        """Retorna (prob_a, prob_b)."""
        df = df[self.FEATURE_COLUMNS]
        if df.isnull().any().any():
            log.warning("Features con NaN detectados, imputando con 0")
            df = df.fillna(0)
        probs = self._model.predict_proba(df)[0]
        return float(probs[1]), float(probs[0])

# ============================================================
# SCRAPER DE JUGADORES
# ============================================================

# Rotación de User-Agents para evitar bloqueos 403
_UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
]
_ua_index = 0

def _next_ua() -> dict:
    global _ua_index
    ua = _UA_LIST[_ua_index % len(_UA_LIST)]
    _ua_index += 1
    return {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }
SURFACE_MAP = {"hard": 0, "clay": 1, "grass": 2}

STATS_DEFAULT = {
    "elo": 1500,
    "elo_hard": 1500,
    "elo_clay": 1500,
    "elo_grass": 1500,
    "hold_pct": 78.0,
    "break_pct": 22.0,
    "dr": 1.00,
    "forma_10": 50.0,
    "streak": 0,
    "tb_win_pct": 50.0,
    "ranking": 200,
    "matches_last_7d": 0,
    "h2h_vs_rival": 0,
}


def _fetch(url: str, params: dict = None) -> requests.Response:
    """Fetch con User-Agent rotativo y backoff ante 403/429."""
    for intento in range(3):
        headers = _next_ua()
        res = requests.get(url, headers=headers, params=params, timeout=CFG.request_timeout)
        if res.status_code == 403:
            wait = 3 * (intento + 1)
            log.debug(f"403 en {url} — esperando {wait}s y rotando UA")
            time.sleep(wait)
            continue
        return res
    return res  # devuelve el último 403 para que el caller lo maneje


def extraer_jugador(nombre: str) -> dict:
    cached = cache_jugadores.get(nombre)
    if cached:
        return cached

    stats = STATS_DEFAULT.copy()

    try:
        nombre_fmt = nombre.replace(" ", "_")
        url = f"https://www.tennisabstract.com/cgi-bin/player.cgi?p={nombre_fmt}"
        res = _fetch(url)

        if res.status_code == 403:
            log.debug(f"tennisabstract bloqueó (403) a {nombre} — usando stats por defecto")
            cache_jugadores.set(nombre, stats)
            return stats
        if res.status_code != 200:
            log.debug(f"tennisabstract HTTP {res.status_code} para {nombre} — usando stats por defecto")
            cache_jugadores.set(nombre, stats)
            return stats

        text = BeautifulSoup(res.text, "html.parser").get_text(" ")

        def buscar(patron, default):
            m = re.search(patron, text, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
            return default

        stats["elo"]       = buscar(r"ELO[:\s]+(\d+(?:\.\d+)?)", 1500)
        stats["elo_hard"]  = buscar(r"Hard\s+ELO[:\s]+(\d+(?:\.\d+)?)", stats["elo"])
        stats["elo_clay"]  = buscar(r"Clay\s+ELO[:\s]+(\d+(?:\.\d+)?)", stats["elo"])
        stats["elo_grass"] = buscar(r"Grass\s+ELO[:\s]+(\d+(?:\.\d+)?)", stats["elo"])
        stats["hold_pct"]  = buscar(r"Hold\s*%\s*(\d+\.?\d*)", 78.0)
        stats["break_pct"] = buscar(r"Break\s*%\s*(\d+\.?\d*)", 22.0)
        stats["ranking"]   = buscar(r"Current Rank[:\s]+(\d+)", 200)

        denom = 100 - stats["break_pct"]
        stats["dr"] = round(stats["hold_pct"] / denom, 2) if denom != 0 else 1.0

        resultados = re.findall(r"\b([WL])\b", text)
        if resultados:
            ultimos = resultados[:10]
            wins = ultimos.count("W")
            stats["forma_10"] = round((wins / len(ultimos)) * 100, 1)
            streak = 0
            for r in ultimos:
                if r == "W":
                    streak += 1
                else:
                    break
            stats["streak"] = streak

        tb = re.search(r"Tiebreaks\s*[:\s]*(\d+)-(\d+)", text, re.IGNORECASE)
        if tb:
            w, l = int(tb.group(1)), int(tb.group(2))
            if (w + l) > 0:
                stats["tb_win_pct"] = round((w / (w + l)) * 100, 1)

        log.debug(f"Stats extraídos para {nombre}: ELO={stats['elo']} rank={stats['ranking']}")

    except Exception as e:
        log.error(f"Error scraping {nombre}: {e}")

    cache_jugadores.set(nombre, stats)
    time.sleep(CFG.rate_limit_delay)
    return stats


def calcular_h2h(jugador_a: str, jugador_b: str) -> int:
    url = f"https://www.tennisabstract.com/cgi-bin/player.cgi?p={jugador_a.replace(' ', '_')}"
    try:
        res = _fetch(url)
    except Exception:
        return 0
    if res.status_code != 200:
        return 0
    text = BeautifulSoup(res.text, "html.parser").get_text(" ")
    apellido = jugador_b.split()[-1]
    wins   = len(re.findall(rf"W.*?{re.escape(apellido)}", text, re.IGNORECASE))
    losses = len(re.findall(rf"L.*?{re.escape(apellido)}", text, re.IGNORECASE))
    return wins - losses

# ============================================================
# ODDS API
# ============================================================

class OddsClient:
    BASE_URL = "https://api.the-odds-api.com/v4/sports"

    def __init__(self, api_key: str = CFG.odds_api_key):
        self._key = api_key

    def _get_odds(self, liga: str) -> list:
        """
        Consulta la API para un torneo específico.
        Retorna lista vacía si el torneo no está en temporada (404)
        sin lanzar excepción ni reintentar (es comportamiento esperado).
        """
        url = f"{self.BASE_URL}/{liga}/odds/"
        ahora_utc = datetime.utcnow()
        params = {
            "apiKey":           self._key,
            "regions":          CFG.regions,
            "markets":          "h2h",
            "oddsFormat":       "decimal",
            "commenceTimeFrom": ahora_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "commenceTimeTo":   (ahora_utc + timedelta(hours=60)).strftime("%Y-%m-%dT%H:%M:%SZ"),  # 60h cubre holgadamente el día siguiente en Colombia
        }
        try:
            res = requests.get(url, params=params, timeout=CFG.request_timeout)
            if res.status_code == 404:
                # Torneo fuera de temporada — normal, no es error
                log.debug(f"  [{liga}] fuera de temporada (404), omitiendo")
                return []
            res.raise_for_status()
            data = res.json()
            if data:
                log.info(f"  [{liga}] {len(data)} partidos encontrados ✅")
            return data
        except requests.exceptions.HTTPError as e:
            log.warning(f"  [{liga}] HTTP {e.response.status_code}, omitiendo")
            return []
        except Exception as e:
            log.error(f"  [{liga}] Error de red: {e}")
            return []

    @staticmethod
    def _mejor_cuota(bookmakers: list, nombre: str) -> float:
        """Devuelve la MEJOR cuota disponible entre todos los bookmakers."""
        mejor = 1.0
        for bm in bookmakers:
            for market in bm.get("markets", []):
                for outcome in market.get("outcomes", []):
                    if outcome["name"] == nombre:
                        mejor = max(mejor, float(outcome["price"]))
        return mejor

    @staticmethod
    def _en_rango_escalera(cuota: float) -> bool:
        return CFG.cuota_escalera_min <= cuota <= CFG.cuota_escalera_max

    def obtener_partidos(self) -> tuple[list[dict], int, int]:
        """
        Retorna (partidos_en_rango, total_fecha_manana, total_fuera_rango).
        """
        partidos = []
        manana = fecha_manana_local()
        total_fecha_global   = 0
        total_fuera_rango    = 0
        log.info(f"📅 Buscando partidos para mañana: {manana} (Colombia UTC-5)")

        for liga in CFG.ligas:
            try:
                data = self._get_odds(liga)
            except Exception as e:
                log.error(f"Odds API error ({liga}): {e}")
                continue

            total_liga      = len(data)
            pasaron_fecha   = 0
            pasaron_cuota   = 0

            descartados_cuota = []  # para diagnóstico

            for match in data:
                # ── Filtro 1: Solo partidos de mañana (hora Colombia) ─
                commence_time = match.get("commence_time", "")
                if not es_partido_de_manana(commence_time):
                    continue
                pasaron_fecha += 1

                torneo    = match.get("sport_title", "")
                surface   = detectar_superficie(torneo)
                bms       = match.get("bookmakers", [])
                jugador_a = match.get("home_team", "")
                jugador_b = match.get("away_team", "")

                if not jugador_a or not jugador_b:
                    continue

                cuota_a = self._mejor_cuota(bms, jugador_a)
                cuota_b = self._mejor_cuota(bms, jugador_b)

                # ── Filtro 2: Al menos uno en rango escalera 1.35–1.70 ─
                if not (self._en_rango_escalera(cuota_a) or self._en_rango_escalera(cuota_b)):
                    descartados_cuota.append(
                        f"{jugador_a}[{cuota_a}] vs {jugador_b}[{cuota_b}]"
                    )
                    continue
                pasaron_cuota += 1

                partidos.append({
                    "jugador_a":     jugador_a,
                    "jugador_b":     jugador_b,
                    "cuota_a":       cuota_a,
                    "cuota_b":       cuota_b,
                    "surface":       surface,
                    "torneo":        torneo,
                    "tour":          "WTA" if "wta" in liga else "ATP",
                    "commence_time": commence_time,
                })

            total_fecha_global += pasaron_fecha
            total_fuera_rango  += len(descartados_cuota)

            # Log diagnóstico por liga
            log.info(
                f"  [{liga}] total_api={total_liga} | "
                f"fecha_mañana={pasaron_fecha} | "
                f"en_rango_escalera={pasaron_cuota} | "
                f"fuera_de_rango={len(descartados_cuota)}"
            )
            for desc in descartados_cuota:
                log.info(f"    ↳ cuota fuera rango 1.35–1.70: {desc}")

        log.info(
            f"✅ Resumen global: fecha_mañana={total_fecha_global} | "
            f"en_rango={len(partidos)} | fuera_rango={total_fuera_rango}"
        )
        return partidos, total_fecha_global, total_fuera_rango

# ============================================================
# CONSTRUCCIÓN DE FEATURES
# ============================================================

def construir_features(
    jugador_a: str,
    jugador_b: str,
    surface: str,
    cuota_a: float,
    cuota_b: float,
) -> pd.DataFrame:

    a = extraer_jugador(jugador_a)
    b = extraer_jugador(jugador_b)

    a["h2h_vs_rival"] = calcular_h2h(jugador_a, jugador_b)
    b["h2h_vs_rival"] = calcular_h2h(jugador_b, jugador_a)

    elo_a = a.get(f"elo_{surface}", a["elo"])
    elo_b = b.get(f"elo_{surface}", b["elo"])

    features = {
        "delta_elo":          a["elo"] - b["elo"],
        "delta_elo_surface":  elo_a - elo_b,
        "delta_hold":         a["hold_pct"] - b["hold_pct"],
        "delta_break":        a["break_pct"] - b["break_pct"],
        "delta_dr":           a["dr"] - b["dr"],
        "delta_forma":        a["forma_10"] - b["forma_10"],
        "delta_streak":       a["streak"] - b["streak"],
        "delta_tb":           a["tb_win_pct"] - b["tb_win_pct"],
        "delta_h2h":          a["h2h_vs_rival"] - b["h2h_vs_rival"],
        "delta_rank":         b["ranking"] - a["ranking"],
        "delta_fatiga":       b["matches_last_7d"] - a["matches_last_7d"],
        "delta_market_prob":  prob_implicita(cuota_a) - prob_implicita(cuota_b),
        "surface":            SURFACE_MAP.get(surface, 0),
    }

    return pd.DataFrame([features])

# ============================================================
# TELEGRAM
# ============================================================

class TelegramNotifier:
    def __init__(self, token: str = CFG.telegram_token, chat_id: str = CFG.telegram_chat_id):
        self._token = token
        self._chat_id = chat_id
        self._url = f"https://api.telegram.org/bot{token}/sendMessage"

    @retry(max_retries=2)
    def enviar(self, mensaje: str):
        if not self._token or not self._chat_id:
            log.warning("Telegram no configurado, omitiendo notificación")
            return
        payload = {
            "chat_id": self._chat_id,
            "text": mensaje,
            "parse_mode": "Markdown"
        }
        res = requests.post(self._url, json=payload, timeout=10)
        res.raise_for_status()

# ============================================================
# REGISTRO DE PICKS (CSV para backtesting)
# ============================================================

class PicksLogger:
    CAMPOS = [
        "timestamp", "torneo", "tour", "jugador_a", "jugador_b",
        "pick", "surface", "cuota_casa", "cuota_justa",
        "prob_ia", "edge_pct", "kelly_pct", "stake_sugerido"
    ]

    def __init__(self, path: str = CFG.picks_log_path):
        self._path = path
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.CAMPOS).writeheader()

    def registrar(self, pick: dict):
        with open(self._path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CAMPOS)
            writer.writerow({k: pick.get(k, "") for k in self.CAMPOS})

# ============================================================
# FORMATEADOR DE MENSAJES
# ============================================================

def formatear_mensaje(p: dict, jugador: str, cuota_casa: float, cuota_justa: float,
                       prob: float, edge: float, kelly_pct: float, stake: float) -> str:
    ventaja = round((edge - 1) * 100, 1)
    estrellas = "⭐" * min(5, max(1, int(ventaja / 2)))

    # Hora local del partido si está disponible
    hora_str = ""
    if p.get("commence_time"):
        try:
            dt_utc = datetime.strptime(p["commence_time"], "%Y-%m-%dT%H:%M:%SZ")
            dt_local = dt_utc + timedelta(hours=CFG.utc_offset_horas)
            hora_str = f"\n🕐 Hora: `{dt_local.strftime('%H:%M')}`"
        except Exception:
            pass

    return (
        f"🎾 *BOT TENIS IA v7.0 — ESCALERA*\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"🏆 {p['torneo'].upper()} ({p['tour']})\n"
        f"🎯 {p['jugador_a']} vs {p['jugador_b']}\n"
        f"🌍 Superficie: `{p['surface'].capitalize()}`"
        f"{hora_str}\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"🔥 PICK: *{jugador.upper()}*  {estrellas}\n"
        f"💰 Cuota Casa: `{cuota_casa}`\n"
        f"📐 Cuota IA: `{cuota_justa}`\n"
        f"📈 Probabilidad IA: `{round(prob * 100, 1)}%`\n"
        f"⚡ Edge: `+{ventaja}%`\n"
        f"📊 Kelly: `{round(kelly_pct * 100, 1)}%` → Stake: `${round(stake, 2)}`\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"🧠 XGBoost + ELO + H2H + Surface + Kelly"
    )

# ============================================================
# MOTOR PRINCIPAL
# ============================================================

class BotTenis:
    def __init__(self):
        self.modelo          = ModeloIA()
        self.odds            = OddsClient()
        self.telegram        = TelegramNotifier()
        self.picks_log       = PicksLogger()
        self._picks_enviados = 0   # contador para resumen final

    def analizar_partido(self, p: dict):
        try:
            df = construir_features(
                p["jugador_a"], p["jugador_b"],
                p["surface"], p["cuota_a"], p["cuota_b"]
            )
            prob_a, prob_b = self.modelo.predecir(df)

            candidatos = [
                (p["jugador_a"], p["cuota_a"], prob_a),
                (p["jugador_b"], p["cuota_b"], prob_b),
            ]

            for jugador, cuota_casa, prob in candidatos:

                # ── Filtro A: cuota del candidato en rango escalera ──
                if not (CFG.cuota_escalera_min <= cuota_casa <= CFG.cuota_escalera_max):
                    continue

                # ── Filtro B: probabilidad mínima del modelo ─────────
                if prob < CFG.min_prob:
                    log.debug(f"{jugador}: prob IA {round(prob*100,1)}% < mínimo {CFG.min_prob*100}%")
                    continue

                cuota_justa = round(1 / prob, 2)
                edge = cuota_casa / cuota_justa

                # ── Filtro C: edge positivo sobre cuota justa ────────
                if edge < CFG.value_threshold:
                    log.debug(f"{jugador}: edge {round(edge,3)} < umbral {CFG.value_threshold}")
                    continue

                kelly_pct = kelly_criterion(prob, cuota_casa)
                stake = round(CFG.bankroll * kelly_pct, 2)

                mensaje = formatear_mensaje(
                    p, jugador, cuota_casa, cuota_justa,
                    prob, edge, kelly_pct, stake
                )

                self.telegram.enviar(mensaje)
                self._picks_enviados += 1

                pick_data = {
                    "timestamp":      datetime.now().isoformat(),
                    "torneo":         p["torneo"],
                    "tour":           p["tour"],
                    "jugador_a":      p["jugador_a"],
                    "jugador_b":      p["jugador_b"],
                    "pick":           jugador,
                    "surface":        p["surface"],
                    "cuota_casa":     cuota_casa,
                    "cuota_justa":    cuota_justa,
                    "prob_ia":        round(prob, 4),
                    "edge_pct":       round((edge - 1) * 100, 2),
                    "kelly_pct":      round(kelly_pct * 100, 2),
                    "stake_sugerido": stake,
                }
                self.picks_log.registrar(pick_data)
                log.info(
                    f"✅ PICK ESCALERA: {jugador} @ {cuota_casa} | "
                    f"prob={round(prob*100,1)}% | edge=+{round((edge-1)*100,1)}% | stake=${stake}"
                )

        except Exception as e:
            log.error(f"Error analizando {p.get('jugador_a')} vs {p.get('jugador_b')}: {e}")

    def run(self):
        manana = fecha_manana_local()
        log.info(f"🎾 BOT TENIS IA v7.0 — Análisis escalera para {manana}")

        partidos, total_manana, fuera_rango = self.odds.obtener_partidos()

        if not partidos:
            sin_picks = (
                f"🎾 *BOT TENIS IA v7.0*\n"
                f"━━━━━━━━━━━━━━━━━━━\n"
                f"📅 Partidos para: `{manana}`\n"
                f"📊 Partidos encontrados para mañana: `{total_manana}`\n"
                f"❌ En rango escalera (1.35–1.70): `0`\n"
                f"⚠️ Todos tienen cuotas fuera del rango. Sin picks de escalera."
            )
            self.telegram.enviar(sin_picks)
            log.warning(f"0 en rango escalera (mañana={total_manana}, fuera_rango={fuera_rango})")
            return

        # Mensaje de inicio con diagnóstico completo
        inicio = (
            f"🎾 *BOT TENIS IA v7.0 — ESCALERA*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"📅 Analizando partidos para: `{manana}`\n"
            f"📊 Total partidos mañana: `{total_manana}`\n"
            f"🎯 En rango escalera (1.35–1.70): `{len(partidos)}`\n"
            f"⏭ Fuera de rango (cuota muy baja/alta): `{fuera_rango}`\n"
            f"⏳ Procesando estadísticas IA…"
        )
        self.telegram.enviar(inicio)

        for p in partidos:
            self.analizar_partido(p)
            time.sleep(CFG.rate_limit_delay)

        # Resumen final
        resumen = (
            f"✅ *ANÁLISIS COMPLETADO*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"📅 Fecha: `{manana}`\n"
            f"📊 Total partidos mañana: `{total_manana}`\n"
            f"🎯 En rango escalera: `{len(partidos)}`\n"
            f"⏭ Fuera de rango: `{fuera_rango}`\n"
            f"🔥 Picks válidos enviados: `{self._picks_enviados}`\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"🧠 XGBoost + ELO + H2H + Surface + Kelly"
        )
        self.telegram.enviar(resumen)
        log.info(f"✅ Análisis completado — {self._picks_enviados} picks enviados")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    BotTenis().run()

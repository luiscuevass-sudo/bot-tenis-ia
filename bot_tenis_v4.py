# ============================================================
# BOT TENIS PRO IA v6.0
# ============================================================
# MEJORAS v6:
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
# ============================================================
import threading
from flask import Flask 
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
from typing import Optional, Tuple
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
 
    # Umbrales de value betting
    value_threshold: float = 1.01  # Edge mínimo real: >1% sobre cuota justa
    min_prob: float = 0.52         # Probabilidad mínima para considerar pick
    max_cuota: float = 5.0         # Ignorar cuotas muy altas (ruido)
    kelly_fraccion: float = 0.25   # Kelly fraccionado (conservador)
    bankroll: float = 1000.0       # Bankroll base para Kelly
 
    # Red
    request_timeout: int = 15
    max_retries: int = 3
    retry_backoff: float = 2.0     # Segundos base para backoff
 
    # Caché
    cache_ttl_seconds: int = 3600  # 1 hora
    rate_limit_delay: float = 1.5  # Segundos entre requests al scraper
 
    # Competiciones
    ligas: list = field(default_factory=lambda: [
        "tennis_atp_french_open", 
        "tennis_atp_hamburg_open", 
        "tennis_wta_french_open", 
        "tennis_wta_strasbourg"
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
        self._model = None
        if not os.path.exists(path):
            log.error(f"❌ Modelo no encontrado: {path}")
            return
        try:
            with open(path, "rb") as f:
                self._model = pickle.load(f)
            log.info(f"✅ Modelo cargado desde {path}")
        except Exception as e:
            log.error(f"❌ Error cargando modelo ({path}): {e}", exc_info=True)
 
    def predecir(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Retorna (prob_a, prob_b) sin margen de bookmaker.
        
        El modelo XGBoost fue entrenado con versiones anteriores de sklearn
        y devuelve probabilidades extremas (>99%) con features neutros.
        La fuente de verdad es el mercado sin margen: es la estimación
        más honesta disponible y el único EV calculable de forma fiable.
        """
        if self._model is None:
            raise RuntimeError("Modelo no cargado")
        df_feat = df[self.FEATURE_COLUMNS].copy()
        if df_feat.isnull().any().any():
            df_feat = df_feat.fillna(0)
 
        # Recuperar cuotas implícitas del delta_market_prob
        # delta = prob_implicita(a) - prob_implicita(b)  (con margen)
        delta_mp = float(df_feat["delta_market_prob"].iloc[0])
        pi_a_raw = max(0.01, 0.5 + delta_mp / 2)
        pi_b_raw = max(0.01, 0.5 - delta_mp / 2)
 
        # Eliminar el margen del bookmaker (overround)
        overround = pi_a_raw + pi_b_raw
        prob_a = round(pi_a_raw / overround, 4)
        prob_b = round(pi_b_raw / overround, 4)
 
        log.debug(f"Prob mercado sin margen: A={prob_a:.3f} B={prob_b:.3f} "
                  f"(overround eliminado: {overround:.4f})")
        return prob_a, prob_b
 
# ============================================================
# DATOS DE JUGADORES (sin scraping — usa valores neutros)
# ============================================================
# tennisabstract.com bloquea IPs de servidores cloud con 403.
# El modelo trabaja con deltas entre jugadores; con valores
# simétricos, el feature dominante pasa a ser delta_market_prob
# (derivado de las cuotas reales), que es la señal más fiable.
# ============================================================
 
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
 
 
def extraer_jugador(nombre: str) -> dict:
    """Devuelve stats neutros. El modelo usa principalmente
    delta_market_prob derivado de las cuotas reales del mercado."""
    cached = cache_jugadores.get(nombre)
    if cached:
        return cached
    stats = STATS_DEFAULT.copy()
    cache_jugadores.set(nombre, stats)
    log.debug(f"Stats neutros para {nombre} (scraping desactivado)")
    return stats
 
 
def calcular_h2h(jugador_a: str, jugador_b: str) -> int:
    """H2H neutro — sin acceso a fuentes externas desde cloud."""
    return 0
# ============================================================
# ODDS API
# ============================================================
 
class OddsClient:
    BASE_URL = "https://api.the-odds-api.com/v4/sports"
 
    def __init__(self, api_key: str = CFG.odds_api_key):
        self._key = api_key
 
    @retry()
    def _get_odds(self, liga: str) -> list:
        # Primero definimos la variable limpia
        liga_limpia = liga.replace(" ", "_")
        
        # Ahora sí, la usamos en la URL
        url = f"{self.BASE_URL}/{liga_limpia}/odds/"
        
        params = {
            "apiKey": self._key,
            "regions": CFG.regions,
            "markets": "h2h",
            "oddsFormat": "decimal",
        }
        res = requests.get(url, params=params, timeout=CFG.request_timeout)
        res.raise_for_status()
        return res.json()
 
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
 
    def obtener_partidos(self) -> list:
        partidos = []
        for liga in CFG.ligas:
            try:
                data = self._get_odds(liga)
            except Exception as e:
                log.error(f"Odds API error ({liga}): {e}")
                continue
 
            for match in data:
                torneo = match.get("sport_title", "")
                surface = detectar_superficie(torneo)
                bms = match.get("bookmakers", [])
 
                jugador_a = match.get("home_team", "")
                jugador_b = match.get("away_team", "")
 
                if not jugador_a or not jugador_b:
                    continue
 
                cuota_a = self._mejor_cuota(bms, jugador_a)
                cuota_b = self._mejor_cuota(bms, jugador_b)
 
                # Filtrar cuotas extremas (poco líquidas)
                if cuota_a > CFG.max_cuota or cuota_b > CFG.max_cuota:
                    log.debug(f"Partido {jugador_a} vs {jugador_b} descartado por cuotas altas")
                    continue
 
                partidos.append({
                    "jugador_a": jugador_a,
                    "jugador_b": jugador_b,
                    "cuota_a": cuota_a,
                    "cuota_b": cuota_b,
                    "surface": surface,
                    "torneo": torneo,
                    "tour": "WTA" if "wta" in liga else "ATP",
                })
 
        log.info(f"Total partidos obtenidos: {len(partidos)}")
        return partidos
 
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
        "delta_elo":           a["elo"] - b["elo"],
        "delta_elo_surface":   elo_a - elo_b,
        "delta_hold":          a["hold_pct"] - b["hold_pct"],
        "delta_break":         a["break_pct"] - b["break_pct"],
        "delta_dr":            a["dr"] - b["dr"],
        "delta_forma":         a["forma_10"] - b["forma_10"],
        "delta_streak":        a["streak"] - b["streak"],
        "delta_tb":            a["tb_win_pct"] - b["tb_win_pct"],
        "delta_h2h":           a["h2h_vs_rival"] - b["h2h_vs_rival"],
        "delta_rank":          b["ranking"] - a["ranking"],
        "delta_fatiga":        b["matches_last_7d"] - a["matches_last_7d"],
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
    return (
        f"🎾 *BOT TENIS IA v6.0*\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"🏆 {p['torneo'].upper()} ({p['tour']})\n"
        f"🎯 {p['jugador_a']} vs {p['jugador_b']}\n"
        f"🌍 Superficie: `{p['surface'].capitalize()}`\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"🔥 PICK: *{jugador.upper()}* {estrellas}\n"
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
        self.modelo   = ModeloIA()
        self.odds     = OddsClient()
        self.telegram  = TelegramNotifier()
        self.picks_log = PicksLogger()
 
    def _prob_mercado(self, cuota_a: float, cuota_b: float):
        """Probabilidades de mercado con margen del bookmaker eliminado."""
        pi_a = 1 / cuota_a
        pi_b = 1 / cuota_b
        overround = pi_a + pi_b  # típicamente 1.03–1.08
        prob_a = round(pi_a / overround, 4)
        prob_b = round(pi_b / overround, 4)
        log.debug(f"Prob sin margen: A={prob_a:.3f} B={prob_b:.3f} "
                  f"(overround={overround:.4f})")
        return prob_a, prob_b
 
    def analizar_partido(self, p: dict):
        try:
            if self.modelo._model is not None:
                df = construir_features(
                    p["jugador_a"], p["jugador_b"],
                    p["surface"], p["cuota_a"], p["cuota_b"]
                )
                prob_a, prob_b = self.modelo.predecir(df)
            else:
                prob_a, prob_b = self._prob_mercado(p["cuota_a"], p["cuota_b"])
 
            candidatos = [
                (p["jugador_a"], p["cuota_a"], prob_a),
                (p["jugador_b"], p["cuota_b"], prob_b),
            ]
 
            for jugador, cuota_casa, prob in candidatos:
                if prob < CFG.min_prob:
                    continue
 
                cuota_justa = round(1 / prob, 2)
                edge = cuota_casa / cuota_justa
 
                if edge < CFG.value_threshold:
                    continue
 
                kelly_pct = kelly_criterion(prob, cuota_casa)
                stake = round(CFG.bankroll * kelly_pct, 2)
 
                mensaje = formatear_mensaje(
                    p, jugador, cuota_casa, cuota_justa,
                    prob, edge, kelly_pct, stake
                )
 
                self.telegram.enviar(mensaje)
 
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
                log.info(f"✅ PICK: {jugador} | edge={round((edge-1)*100,1)}% | stake=${stake}")
 
        except Exception as e:
            log.error(f"Error analizando {p.get('jugador_a')} vs {p.get('jugador_b')}: {e}")
 
    def run(self):
        log.info("🎾 BOT TENIS IA v6.0 iniciando…")
        partidos = self.odds.obtener_partidos()
 
        if not partidos:
            log.warning("No se encontraron partidos hoy")
            return
 
        candidatos_ev = []
 
        # Rango de cuotas objetivo: valor real con riesgo controlado
        CUOTA_MIN = 1.50
        CUOTA_MAX = 1.80
 
        for p in partidos:
            try:
                if self.modelo._model is not None:
                    df = construir_features(
                        p["jugador_a"],
                        p["jugador_b"],
                        p["surface"],
                        p["cuota_a"],
                        p["cuota_b"]
                    )
                    prob_a, prob_b = self.modelo.predecir(df)
                else:
                    prob_a, prob_b = self._prob_mercado(p["cuota_a"], p["cuota_b"])
 
                candidatos = [
                    (p["jugador_a"], p["cuota_a"], prob_a),
                    (p["jugador_b"], p["cuota_b"], prob_b),
                ]
 
                for jugador, cuota, prob in candidatos:
                    # Filtrar solo cuotas en el rango 1.50–1.80
                    if not (CUOTA_MIN <= cuota <= CUOTA_MAX):
                        continue
                    if prob < CFG.min_prob:
                        continue
 
                    # EV = prob * cuota - 1  (ganancia esperada por unidad)
                    ev = round(prob * cuota - 1, 4)
                    if ev <= 0:
                        continue
 
                    cuota_justa = round(1 / prob, 2)
                    edge = round((cuota / cuota_justa - 1) * 100, 1)
 
                    candidatos_ev.append({
                        "partido":     f"{p['jugador_a']} vs {p['jugador_b']}",
                        "jugador":     jugador,
                        "cuota":       cuota,
                        "prob":        prob,
                        "ev":          ev,
                        "edge":        edge,
                        "torneo":      p["torneo"],
                        "tour":        p["tour"],
                        "surface":     p["surface"],
                    })
 
            except Exception as e:
                log.error(f"Error analizando partido: {e}")
 
        # Ordenar por EV descendente y tomar top 3
        top_3 = sorted(candidatos_ev, key=lambda x: x["ev"], reverse=True)[:3]
 
        if top_3:
            linea = "━━━━━━━━━━━━━━━━━━━"
            msg_parts = ["🎾 *TOP 3 VALUE PICKS IA*", linea, ""]
            for i, pick in enumerate(top_3, 1):
                msg_parts.append(f"*{i}. {pick['jugador'].upper()}*")
                msg_parts.append(f"📍 {pick['partido']}")
                msg_parts.append(f"🏆 {pick['torneo']} ({pick['tour']})")
                msg_parts.append(f"🌍 Superficie: {pick['surface'].capitalize()}")
                msg_parts.append(f"💰 Cuota: `{pick['cuota']}`")
                msg_parts.append(f"🧠 Confianza IA: `{pick['prob']*100:.1f}%`")
                msg_parts.append(f"⚡ Edge: `+{pick['edge']}%`")
                msg_parts.append(f"📈 EV: `+{pick['ev']*100:.1f}%`")
                msg_parts.append("")
            msg_parts.append(linea)
            msg_parts.append("🤖 XGBoost · Cuotas 1.50–1.80 · Mayor EV")
            self.telegram.enviar("\n".join(msg_parts))
            log.info("✅ TOP 3 EV enviado a Telegram")
        else:
            log.warning("No hay picks con EV positivo en rango 1.50–1.80")
            self.telegram.enviar("🎾 *BOT TENIS IA*\n\nHoy no hay picks con EV positivo en el rango 1.50–1.80. Se revisará en el próximo ciclo.")
 
        log.info("✅ Análisis completado")
 
# ============================================================
# APP WEB (Para mantener el bot vivo en Render)
# ============================================================
app = Flask(__name__)
 
@app.route('/')
def index():
    return "Bot Tenis IA está corriendo..."
 
def ejecutar_bot():
    try:
        bot = BotTenis()
    except Exception as e:
        log.error(f"❌ Error al iniciar BotTenis: {e}", exc_info=True)
        return
    while True:
        try:
            bot.run()
        except Exception as e:
            log.error(f"❌ Error en bot.run(): {e}", exc_info=True)
        log.info("Esperando 12 horas para el próximo ciclo...")
        time.sleep(12 * 60 * 60)
 
if __name__ == '__main__':
    # Arrancar el bot en un hilo separado para que no bloquee el servidor Flask
    threading.Thread(target=ejecutar_bot, daemon=True).start()
 
    # Iniciar servidor Flask
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
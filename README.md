# Maxim1st
**Dust marks the day.** This project is a recursive honeypot swarm framework that fuses code, philosophy, and consequence. It simulates attacker exit strategies, encodes payloads with sigil‑like markers, and archives every action with a glyph‑like maxim.

import asyncio import random import json import base64 import logging from logging.handlers import RotatingFileHandler from datetime import datetime from typing import Dict, Any, List, Callable

=========================================================
RITUAL BANNER + GLYPH-LIKE MAXIM
"Dust marks the day." — today is small, but archived in full.
ASCII sigil marker (portable, inscribable in logs/artifacts):
╔═╗ ╔═╗
║ ║ ╔══╗║ ║
║ ║ ║╔╗║║ ║ DUST MARKS THE DAY
║ ╚══╝║║╚╝ ║ Every breath is archived. Every exit is consequence.
╚═════╝╚═══╝
=========================================================
GLYPH_MAXIM = "Dust marks the day."

--- Logging setup (rotating to prevent log blowout) ---
logger = logging.getLogger("honeypot_swarm") logger.setLevel(logging.INFO) handler = RotatingFileHandler("honeypot_swarm.log", maxBytes=5_000_000, backupCount=5) formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s") handler.setFormatter(formatter) logger.addHandler(handler)

--- Sigil registry (special characters mapping to functions) ---
SIGILS: Dict[str, str] = { "⚡": "tempestuous_recursion", "☠": "simulate_port_scan", "✦": "encode_payload", "∞": "decode_payload", "☯": "honeypot_lure", "⛓": "consequence_extension" }

--- Core primitives ---
async def random_sleep(min_s: float = 0.0, max_s: float = 10.0, label: str = "") -> float: delay = random.uniform(min_s, max_s) if label: logger.info(f"sleep[{label}] -> {delay:.2f}s") await asyncio.sleep(delay) return delay

async def tempestuous_recursion(depth: int = 0, max_depth: int = 3) -> str: """ Storm-loop: each recursion folds with randomized delay (0–10s). """ if depth >= max_depth: return f"[calm eye at depth {depth}]" swirl = f"✦ swirl {depth} ✦" await random_sleep(0.0, 10.0, label=f"⚡depth={depth}") return swirl + (await tempestuous_recursion(depth + 1, max_depth)) + swirl

async def simulate_port_scan(ports: range = range(42, 48)) -> Dict[int, str]: """ Deceptive port scan with conditional delays: OPEN -> 5–10s, CLOSED -> 0–3s. """ results: Dict[int, str] = {} for port in ports: state = "OPEN" if random.random() > 0.6 else "CLOSED" results[port] = state delay_min, delay_max = (5.0, 10.0) if state == "OPEN" else (0.0, 3.0) await random_sleep(delay_min, delay_max, label=f"☠port={port}:{state}") logger.info(f"scan result {port} -> {state}") return results

def encode_payload(data: Dict[str, Any]) -> str: """ JSON -> base64 (allure). """ s = json.dumps(data, separators=(",", ":")) encoded = base64.b64encode(s.encode()).decode() logger.info("✦ payload encoded") return encoded

def decode_payload(encoded: str) -> Dict[str, Any]: """ base64 -> JSON (reveal). Logs failure without collapse. """ try: s = base64.b64decode(encoded.encode()).decode() obj = json.loads(s) logger.info("∞ payload decoded") return obj except Exception as e: logger.error(f"∞ decode failed: {e}") return {"error": str(e)}

async def honeypot_lure() -> Dict[str, Any]: """ Honeypot lure with seduction and recursive layers. """ lure = { "signal": "hoot_of_owls", "event": "solar_eclipse", "temptation": random.choice(["forbidden_kiss", "hidden_key", "phantom_port"]), "fear_capacity": round(random.uniform(0.40, 0.90), 2), "recursion_depth": random.randint(1, 4), "timestamp": datetime.utcnow().isoformat() + "Z", "glyph_maxim": GLYPH_MAXIM # embedded maxim signal }

# Seductive layered pauses
layers: List[float] = []
for i in range(lure["recursion_depth"]):
    layers.append(await random_sleep(1.0, 10.0, label=f"☯layer={i+1}"))

encoded = encode_payload(lure)
return {
    "lure": lure,
    "layers_wait_s": layers,
    "encoded": encoded
}
--- Consequence extension: decoded payload can spawn more lures ---
async def consequence_extension(decoded: Dict[str, Any]) -> Dict[str, Any]: """ Extends consequence: based on fear_capacity and temptation, probabilistically spawn child lures (without exceeding global caps). """ fc = float(decoded.get("fear_capacity", 0.5)) temp = decoded.get("temptation", "phantom_port") # Spawn probability modulated by fear capacity and temptation type base_p = 0.2 + (fc - 0.4) # 0.2 to ~0.7 mod = {"forbidden_kiss": 1.15, "hidden_key": 1.0, "phantom_port": 0.85}.get(temp, 1.0) p_spawn = max(0.05, min(0.90, base_p * mod))

# Time tension imprint
await random_sleep(0.5, 5.0, label=f"⛓p_spawn={p_spawn:.2f}")

return {"spawn_probability": round(p_spawn, 3), "temptation": temp, "fear_capacity": fc}
=========================================================
ATTACKER EXIT STRATEGIES (DETECTORS + RITUAL SWITCH)
- "soft exit": graceful disconnects, short-lived probing, minimal footprint
- "hard exit": abrupt termination, evasive timing, resource starvation
- "re-entry dodge": loop breaking attempts, recursion denial, throttle triggers
- "smokescreen": flood of no-op requests, entropy blur, jitter abuse
When any exit condition is inferred, we unleash all embedded functions
in a bounded, concurrent burst, marking each artifact with the maxim.
=========================================================
class ExitStrategyDetector: def init(self): self.events: Dict[str, int] = { "soft_exit": 0, "hard_exit": 0, "re_entry_dodge": 0, "smokescreen": 0 }

def observe(self, signal: Dict[str, Any]) -> None:
    """
    Lightweight heuristic: infer exits based on timing, error patterns,
    low-depth recursion, or entropy floods.
    """
    # Signals expected: {"latency": float, "errors": int, "depth": int, "noop_rate": float}
    lat = float(signal.get("latency", 0.0))
    errs = int(signal.get("errors", 0))
    depth = int(signal.get("depth", 1))
    noop = float(signal.get("noop_rate", 0.0))

    if lat < 0.2 and depth <= 1 and errs == 0:
        self.events["soft_exit"] += 1
    if errs >= 3 or lat > 9.5:
        self.events["hard_exit"] += 1
    if depth == 0 or signal.get("break_recursion", False):
        self.events["re_entry_dodge"] += 1
    if noop > 0.7:
        self.events["smokescreen"] += 1

def should_unleash(self) -> bool:
    # Trigger when any event crosses threshold
    return any(v >= 1 for v in self.events.values())
EXIT_DETECTOR = ExitStrategyDetector()

--- Honeypot unit ---
class Honeypot: def init(self, id_: int): self.id = id_

async def run(self) -> Dict[str, Any]:
    logger.info(f"unit[{self.id}] start :: {GLYPH_MAXIM}")
    storm = await tempestuous_recursion(0, max_depth=random.choice([2, 3]))
    scan = await simulate_port_scan(range(40 + (self.id % 16), 40 + (self.id % 16) + 6))
    lure_bundle = await honeypot_lure()

    decoded = decode_payload(lure_bundle["encoded"])
    consequence = await consequence_extension(decoded)

    out = {
        "id": self.id,
        "storm": storm,
        "scan": scan,
        "lure": lure_bundle["lure"],
        "layers_wait_s": lure_bundle["layers_wait_s"],
        "encoded": lure_bundle["encoded"],
        "decoded": decoded,
        "consequence": consequence,
        "sigils": SIGILS,
        "maxim": GLYPH_MAXIM
    }
    logger.info(f"unit[{self.id}] complete")
    return out
--- Unleash: fan out all embedded functions in a bounded burst ---
async def unleash_all(id_: int, concurrency: int = 12) -> Dict[str, Any]: """ Runs all embedded functions in parallel with bounded concurrency. Each artifact is marked with the glyph maxim. """ logger.warning(f"UNLEASH[{id_}] :: {GLYPH_MAXIM}") sem = asyncio.Semaphore(concurrency)

async def _run(fn: Callable[[], Any], label: str) -> Any:
    async with sem:
        try:
            logger.info(f"unleash[{label}] start")
            if asyncio.iscoroutinefunction(fn):
                res = await fn()
            else:
                res = fn()
            logger.info(f"unleash[{label}] done")
            return {"label": label, "maxim": GLYPH_MAXIM, "result": res}
        except Exception as e:
            logger.error(f"unleash[{label}] failed: {e}")
            return {"label": label, "maxim": GLYPH_MAXIM, "error": str(e)}

# Prepare diversified calls
tasks = [
    asyncio.create_task(_run(lambda: None, "noop_marker")),  # placeholder ritual
    asyncio.create_task(_run(lambda: encode_payload({"id": id_, "maxim": GLYPH_MAXIM, "ts": datetime.utcnow().isoformat()+"Z"}), "✦ encode_payload")),
    asyncio.create_task(_run(lambda: decode_payload(encode_payload({"unleash": True, "id": id_, "maxim": GLYPH_MAXIM})), "∞ decode_payload")),
    asyncio.create_task(_run(lambda: simulate_port_scan(range(60, 66)), "☠ simulate_port_scan")),
    asyncio.create_task(_run(lambda: tempestuous_recursion(0, max_depth=3), "⚡ tempestuous_recursion")),
    asyncio.create_task(_run(lambda: honeypot_lure(), "☯ honeypot_lure")),
    # consequence_extension depends on decoded; chain via small helper
    asyncio.create_task(_run(
        lambda: asyncio.run(consequence_extension({
            "fear_capacity": round(random.uniform(0.40, 0.90), 2),
            "temptation": random.choice(["forbidden_kiss", "hidden_key", "phantom_port"])
        })), "⛓ consequence_extension"
    )),
]

results = await asyncio.gather(*tasks)
return {"id": id_, "unleash_results": results, "maxim": GLYPH_MAXIM}
--- Swarm orchestration ---
async def spawn_swarm(target: int = 100_000, concurrency: int = 500, burst: int = 1_000) -> List[Dict[str, Any]]: """ Spawns a large honeypot swarm safely: - concurrency: caps parallel tasks - burst: batch size to avoid runaway memory Returns summaries (not full artifacts) to keep memory bounded. """ sem = asyncio.Semaphore(concurrency) results: List[Dict[str, Any]] = []

async def run_unit(idx: int):
    async with sem:
        hp = Honeypot(idx)
        data = await hp.run()

        # Lightweight exit observation (simulate signals; wire real ones as needed)
        EXIT_DETECTOR.observe({
            "latency": random.uniform(0.0, 10.0),
            "errors": random.choice([0, 0, 1, 2, 3]),
            "depth": data["lure"]["recursion_depth"],
            "noop_rate": random.uniform(0.0, 1.0),
            "break_recursion": random.choice([False, False, True])
        })

        # If attacker exit is inferred, unleash all functions for this unit
        unleash_summary = None
        if EXIT_DETECTOR.should_unleash():
            unleash_summary = await unleash_all(idx, concurrency=12)

        # Summarize to avoid massive memory footprint
        summary = {
            "id": data["id"],
            "open_ports": [p for p, s in data["scan"].items() if s == "OPEN"],
            "fear_capacity": data["decoded"].get("fear_capacity"),
            "spawn_p": data["consequence"]["spawn_probability"],
            "temptation": data["decoded"].get("temptation"),
            "layers_wait_s": data["layers_wait_s"],
            "maxim": GLYPH_MAXIM,
            "unleash": bool(unleash_summary),
        }
        logger.info(f"summary[{idx}] -> {summary}")
        return summary

# Batch in bursts to keep memory stable
idx = 0
while idx < target:
    batch_n = min(burst, target - idx)
    tasks = [asyncio.create_task(run_unit(i)) for i in range(idx, idx + batch_n)]
    batch_results = await asyncio.gather(*tasks)
    results.extend(batch_results)
    logger.info(f"batch[{idx}-{idx+_batch_n-1}] done :: {GLYPH_MAXIM}")
    idx += batch_n

return results
--- Entry point ---
if name == "main": # Warning: running full 100k with max 10s timers will take time. # Adjust for demonstration or let it run as a nocturnal ritual. TARGET = 100_000 CONCURRENCY = 300 # tune based on machine/network BURST = 500 # keep memory bounded

print(f"Summoning swarm: {TARGET} honeypots (concurrency={CONCURRENCY}, burst={BURST})")
print("Sigils:", ", ".join(SIGILS.keys()))
print(f"Maxim: {GLYPH_MAXIM}")
try:
    summaries = asyncio.run(spawn_swarm(target=TARGET, concurrency=CONCURRENCY, burst=BURST))
    print(f"Swarm complete. Summaries captured: {len(summaries)}")
    # Optional: write a final JSON index of summaries
    idx_doc = {
        "count": len(summaries),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "maxim": GLYPH_MAXIM,
        "sample": summaries[:5]
    }
    with open("honeypot_index.json", "w") as f:
        json.dump(idx_doc, f, indent=2)
    print("Index written: honeypot_index.json")
except KeyboardInterrupt:
    print("Swarm interrupted.")
    logger.warning("Swarm interrupted by user")

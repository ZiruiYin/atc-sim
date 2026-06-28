import os
import threading
import time
from collections import OrderedDict

from flask import Flask, jsonify, request, send_from_directory, Response
from werkzeug.serving import WSGIRequestHandler

from environment import SimulationEnv

# Repo root is also the GitHub Pages root. We serve index.html and the
# environment/ tree from here so the same paths work both ways:
#   - Local Flask:   http://127.0.0.1:5000/{environment/foo, env_manifest.json}
#   - GitHub Pages:  https://.../{environment/foo, env_manifest.json}
# No static_folder= argument; everything is served explicitly below.
ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

# Disable Nagle's algorithm on the Werkzeug dev server. Without TCP_NODELAY, the
# browser's tiny /step POSTs hit a Nagle + TCP delayed-ACK stall (~300ms on every
# other request) that makes the radar tick limp one-fast-one-slow. Use this
# handler for EVERY launch path (app.py AND main.py) so the fix can't be missed.
class NoNagleRequestHandler(WSGIRequestHandler):
    def setup(self):
        super().setup()
        try:
            import socket as _socket
            self.connection.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
        except OSError:
            pass

RADAR_SIDE = 800

# Defaults for a new session's first sim (overridden by main.py's CLI flags).
DEFAULT_AIRPORT = "test"
DEFAULT_STAR_MODE = True

# --- Per-session (multi-tenant) state ----------------------------------------
# The Flask backend serves many browsers at once. Each one gets its OWN
# simulation (and, lazily, its own AUTO planner), keyed by the X-Session-Id
# header the client sends with every request. Without this, all visitors share
# one global sim and see each other's aircraft. Sessions are dropped when idle
# or when the cap is exceeded (least-recently-used first).
MAX_SESSIONS = 80
SESSION_TTL = 1800.0          # seconds of inactivity before eviction
_sessions = OrderedDict()      # sid -> Session (most-recently-used at the end)
_sessions_lock = threading.Lock()


class Session:
    """One browser's world: its simulation plus optional AUTO planner."""

    def __init__(self, airport, star_mode):
        self.sim = SimulationEnv(
            radar_side=RADAR_SIDE,
            airport_name=airport,
            star_mode=star_mode,
        )
        # AUTO planner is created lazily (and torch loaded) only if this session
        # actually engages AUTO — most sessions never pay for it.
        self.auto_planner = None
        self.auto_on = False
        self.last_seen = time.monotonic()

    def is_simulated(self):
        # The SIMULATED airport (STAR-following) maps to the 'test' data; AUTO
        # and the strict 2 NM rule are SIMULATED-only.
        return self.sim.airport_name == "test"

    def get_planner(self):
        if self.auto_planner is None:
            from auto_plan.planner import AutoPlanner
            self.auto_planner = AutoPlanner(airport="test", runway="27", plan_steps=400)
        return self.auto_planner

    def auto_status(self):
        st = {"on": self.auto_on, "available": self.is_simulated()}
        if self.auto_planner is not None:
            st.update(self.auto_planner.status())
            st["on"] = self.auto_on
        return st

    def state_payload(self):
        """sim state + AUTO overlay (planning flag, flight-plan lines) when on."""
        s = self.sim.get_state()
        if self.auto_on and self.is_simulated() and self.auto_planner is not None:
            s.update(self.auto_planner.overlay())
        return s


def _evict_locked():
    """Drop idle sessions and enforce the cap. Call with _sessions_lock held."""
    now = time.monotonic()
    for sid in [sid for sid, s in _sessions.items() if now - s.last_seen > SESSION_TTL]:
        _sessions.pop(sid, None)
    while len(_sessions) > MAX_SESSIONS:
        _sessions.popitem(last=False)   # evict least-recently-used


def get_session(create=True):
    """Return this request's Session (keyed by the X-Session-Id header)."""
    sid = request.headers.get('X-Session-Id') or request.args.get('sid') or 'anon'
    with _sessions_lock:
        sess = _sessions.get(sid)
        if sess is None:
            if not create:
                return None
            sess = Session(DEFAULT_AIRPORT, DEFAULT_STAR_MODE)
            _sessions[sid] = sess
        sess.last_seen = time.monotonic()
        _sessions.move_to_end(sid)
        _evict_locked()
        return sess


def init_simulation(airport="test", star_mode=True):
    """Set the defaults new sessions start from (used by main.py's CLI)."""
    global DEFAULT_AIRPORT, DEFAULT_STAR_MODE
    DEFAULT_AIRPORT = airport
    DEFAULT_STAR_MODE = star_mode


@app.route('/')
def index():
    return send_from_directory(ROOT, 'index.html')


@app.route('/favicon.ico')
def favicon():
    # Browsers auto-request /favicon.ico; we have none, so answer 204 (no content)
    # instead of a noisy 404 in the console.
    return ('', 204)


@app.route('/env_manifest.json')
def env_manifest():
    return send_from_directory(ROOT, 'env_manifest.json')


@app.route('/environment/<path:filename>')
def environment_files(filename):
    return send_from_directory(os.path.join(ROOT, 'environment'), filename)


@app.route('/web/<path:filename>')
def web_files(filename):
    # Accounts/leaderboard client (config.js, supabaseClient.js, api.js). On
    # GitHub Pages these are served statically; under Flask we must serve them
    # explicitly, or the Supabase client never loads and login silently fails.
    return send_from_directory(os.path.join(ROOT, 'web'), filename)


@app.route('/state', methods=['GET'])
def state():
    return jsonify(get_session().state_payload())


@app.route('/step', methods=['POST'])
def step():
    sess = get_session()
    if sess.auto_on and sess.is_simulated() and sess.auto_planner is not None:
        # The planner runs its own fast_forward loop and HOLDS the sim while a
        # background replan is in flight (never blocks this request).
        if not sess.sim.crash_occurred:
            sess.auto_planner.step(sess.sim)
    else:
        for _ in range(sess.sim.fast_forward):
            sess.sim.step(1.0)
    return jsonify(sess.state_payload())


@app.route('/command', methods=['POST'])
def command():
    sess = get_session()
    data = request.get_json(force=True) or {}
    callsign = data.get('callsign', '')
    cmd = data.get('command', '')
    result = sess.sim.command(callsign, cmd)
    return jsonify({
        "ok": result.get("ok", False),
        "category": result.get("category"),
        "atc": result.get("atc"),
        "pilot": result.get("pilot"),
        "message": result.get("message"),
        "callsign": result.get("callsign", callsign),
    })


@app.route('/tts', methods=['POST'])
def tts():
    """Synthesize a controller/pilot line to WAV for the browser to play.

    Body: {text, is_controller (callsign placement), controller_voice (which
    checkpoint)}. Returns audio/wav bytes. Only reachable under the Flask backend
    (the static/Pyodide build simply gets no audio)."""
    data = request.get_json(force=True) or {}
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({"ok": False, "message": "no text"}), 400
    is_controller = bool(data.get('is_controller', True))
    controller_voice = bool(data.get('controller_voice', is_controller))
    try:
        from tts.synth import render
        wav = render(text, is_controller=is_controller, controller_voice=controller_voice)
    except Exception as e:                       # piper missing / synth failure
        return jsonify({"ok": False, "message": str(e)}), 500
    return Response(wav, mimetype='audio/wav')


@app.route('/speed', methods=['POST'])
def speed():
    sess = get_session()
    data = request.get_json(force=True) or {}
    multiplier = int(data.get('multiplier', 1))
    sess.sim.set_speed(multiplier)
    return jsonify({"ok": True, "fast_forward": sess.sim.fast_forward})


@app.route('/spawn_rate', methods=['POST'])
def spawn_rate():
    sess = get_session()
    data = request.get_json(force=True) or {}
    rate = int(data.get('rate', 90))
    sess.sim.set_spawn_rate(rate)
    return jsonify({"ok": True, "spawn_rate": sess.sim.spawn_rate})


@app.route('/spawn_directions', methods=['POST'])
def spawn_directions():
    sess = get_session()
    data = request.get_json(force=True) or {}
    dirs = data.get('directions', [])
    sess.sim.set_spawn_directions(dirs)
    return jsonify({"ok": True, "spawn_directions": sess.sim.spawn_directions})


@app.route('/restart', methods=['POST'])
def restart():
    sess = get_session()
    data = request.get_json(force=True) or {}
    sess.sim.restart(
        spawn_single=data.get('spawn_single'),
        star_mode=data.get('star_mode'),
        airport_name=data.get('airport_name'),
    )
    # The old aircraft are gone. If AUTO stays on (still SIMULATED) re-init the
    # planner for the fresh sim; otherwise clear it and turn AUTO off.
    if not sess.is_simulated():
        sess.auto_on = False
    if sess.auto_planner is not None:
        if sess.auto_on and sess.is_simulated():
            sess.auto_planner.enable(sess.sim)
        else:
            sess.auto_planner.reset(sess.sim)
    return jsonify(sess.state_payload())


@app.route('/auto', methods=['GET', 'POST'])
def auto():
    sess = get_session()
    if request.method == 'GET':
        return jsonify(sess.auto_status())
    data = request.get_json(force=True) or {}
    want = bool(data.get('on'))
    if want:
        if not sess.is_simulated():
            return jsonify({"ok": False, "on": False,
                            "message": "AUTO is only available for the SIMULATED airport"}), 400
        sess.get_planner().start()       # one-time policy load + pool warmup
        sess.auto_planner.enable(sess.sim)   # clear stale state; arming starts next step
        sess.auto_on = True
    else:
        sess.auto_on = False
        if sess.auto_planner is not None:
            sess.auto_planner.disable(sess.sim)   # restore physics, leave planes cleared to land
    return jsonify({"ok": True, **sess.auto_status()})


@app.route('/recording/start', methods=['POST'])
def recording_start():
    sess = get_session()
    ok = sess.sim.start_recording()
    return jsonify({"ok": bool(ok), "recording": sess.sim.is_recording()})


@app.route('/recording/stop', methods=['POST'])
def recording_stop():
    sess = get_session()
    csv_text, filename = sess.sim.stop_recording()
    return jsonify({
        "ok": csv_text is not None,
        "csv": csv_text or "",
        "filename": filename or "",
        "recording": sess.sim.is_recording(),
    })


def _warm_tts():
    """Pre-load both Piper voices at boot so the first manual command speaks
    instantly (cold load is ~2.5s/voice; warm synth is ~0.3s)."""
    try:
        from tts.synth import warmup
        warmup()
        print("[tts] voices warmed (controller + pilot)")
    except Exception as e:                       # piper missing / no voices
        print(f"[tts] warmup skipped: {e}")


# Held for the life of the process so the warmed policy + worker pool aren't torn
# down. Real sessions build their own planner but reuse the global worker pool
# (keyed by checkpoint/runway, not by planner instance), plus a torch already
# imported and a checkpoint already in the OS file cache -- so their first AUTO
# engage skips the heavy one-time costs.
_warm_auto_planner = None


def _warm_auto():
    """Pre-load the AUTO policy and spawn+warm its worker pool at boot, so the
    first time a user switches AUTO on it engages without paying the one-time
    cost of importing torch, loading the checkpoint, and (in pool mode) spawning
    worker processes. Off by default; the Hugging Face image sets ATC_WARM_AUTO=1."""
    global _warm_auto_planner
    try:
        from auto_plan.planner import AutoPlanner
        planner = AutoPlanner(airport="test", runway="27", plan_steps=400)
        st = planner.start()             # torch load + pool spawn/warm (idempotent)
        _warm_auto_planner = planner
        print(f"[auto] policy + pool warmed ({st.get('mode')}, "
              f"{st.get('n_workers')} worker(s))")
    except Exception as e:
        print(f"[auto] warmup skipped: {e}")


def _truthy(v):
    return str(v).strip().lower() in ('1', 'true', 'yes', 'on')


# Kick the warmups as soon as the backend is imported (i.e. when the server
# starts), in daemon threads so they never delay startup or request handling.
# TTS always warms (cheap, no subprocesses). AUTO warms only when opted in (it
# spawns worker processes) -- the Hugging Face deployment sets ATC_WARM_AUTO=1.
# The MainProcess guard keeps a multiprocessing spawn-child (should it ever
# import this module) from re-launching the warmups.
import multiprocessing as _mp
if _mp.current_process().name == 'MainProcess':
    threading.Thread(target=_warm_tts, daemon=True).start()
    if _truthy(os.environ.get('ATC_WARM_AUTO', '')):
        threading.Thread(target=_warm_auto, daemon=True).start()


if __name__ == '__main__':
    init_simulation()
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True,
            request_handler=NoNagleRequestHandler)

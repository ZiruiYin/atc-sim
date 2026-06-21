import os

from flask import Flask, jsonify, request, send_from_directory

from environment import SimulationEnv

# Repo root is also the GitHub Pages root. We serve index.html and the
# environment/ tree from here so the same paths work both ways:
#   - Local Flask:   http://127.0.0.1:5000/{environment/foo, env_manifest.json}
#   - GitHub Pages:  https://.../{environment/foo, env_manifest.json}
# No static_folder= argument; everything is served explicitly below.
ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

RADAR_SIDE = 800
sim = None

# AUTO planner (Flask-backend only — cannot run in the Pyodide build). The
# planner + torch are imported lazily so the base sim never pays for them.
_auto_planner = None
auto_on = False


def init_simulation(airport="test", star_mode=True):
    global sim
    sim = SimulationEnv(
        radar_side=RADAR_SIDE,
        airport_name=airport,
        star_mode=star_mode,
    )


def _is_simulated():
    # The SIMULATED airport (STAR-following) maps to the 'test' data; AUTO and
    # the strict 2 NM rule are SIMULATED-only.
    return sim is not None and sim.airport_name == "test"


def _get_planner():
    global _auto_planner
    if _auto_planner is None:
        from auto_plan import get_planner
        _auto_planner = get_planner(airport="test", runway="27", plan_steps=400)
    return _auto_planner


def _auto_status():
    st = {"on": auto_on, "available": _is_simulated()}
    if _auto_planner is not None:
        st.update(_auto_planner.status())
        st["on"] = auto_on
    return st


def _state_payload():
    """sim state + AUTO overlay (planning flag, flight-plan lines) when on."""
    s = sim.get_state()
    if auto_on and _is_simulated() and _auto_planner is not None:
        s.update(_auto_planner.overlay())
    return s


@app.route('/')
def index():
    return send_from_directory(ROOT, 'index.html')


@app.route('/env_manifest.json')
def env_manifest():
    return send_from_directory(ROOT, 'env_manifest.json')


@app.route('/environment/<path:filename>')
def environment_files(filename):
    return send_from_directory(os.path.join(ROOT, 'environment'), filename)


@app.route('/state', methods=['GET'])
def state():
    return jsonify(_state_payload())


@app.route('/step', methods=['POST'])
def step():
    if auto_on and _is_simulated() and _auto_planner is not None:
        # The planner runs its own fast_forward loop and HOLDS the sim while a
        # background replan is in flight (never blocks this request).
        if not sim.crash_occurred:
            _auto_planner.step(sim)
    else:
        for _ in range(sim.fast_forward):
            sim.step(1.0)
    return jsonify(_state_payload())


@app.route('/command', methods=['POST'])
def command():
    data = request.get_json(force=True) or {}
    callsign = data.get('callsign', '')
    cmd = data.get('command', '')
    result = sim.command(callsign, cmd)
    return jsonify({
        "ok": result.get("ok", False),
        "category": result.get("category"),
        "atc": result.get("atc"),
        "pilot": result.get("pilot"),
        "message": result.get("message"),
        "callsign": result.get("callsign", callsign),
    })


@app.route('/speed', methods=['POST'])
def speed():
    data = request.get_json(force=True) or {}
    multiplier = int(data.get('multiplier', 1))
    sim.set_speed(multiplier)
    return jsonify({"ok": True, "fast_forward": sim.fast_forward})


@app.route('/spawn_rate', methods=['POST'])
def spawn_rate():
    data = request.get_json(force=True) or {}
    rate = int(data.get('rate', 90))
    sim.set_spawn_rate(rate)
    return jsonify({"ok": True, "spawn_rate": sim.spawn_rate})


@app.route('/spawn_directions', methods=['POST'])
def spawn_directions():
    data = request.get_json(force=True) or {}
    dirs = data.get('directions', [])
    sim.set_spawn_directions(dirs)
    return jsonify({"ok": True, "spawn_directions": sim.spawn_directions})


@app.route('/restart', methods=['POST'])
def restart():
    global auto_on
    data = request.get_json(force=True) or {}
    sim.restart(
        spawn_single=data.get('spawn_single'),
        star_mode=data.get('star_mode'),
        airport_name=data.get('airport_name'),
    )
    # The old aircraft are gone. If AUTO stays on (still SIMULATED) re-init the
    # planner for the fresh sim; otherwise clear it and turn AUTO off.
    if not _is_simulated():
        auto_on = False
    if _auto_planner is not None:
        if auto_on and _is_simulated():
            _auto_planner.enable(sim)
        else:
            _auto_planner.reset(sim)
    return jsonify(_state_payload())


@app.route('/auto', methods=['GET', 'POST'])
def auto():
    global auto_on
    if request.method == 'GET':
        return jsonify(_auto_status())
    data = request.get_json(force=True) or {}
    want = bool(data.get('on'))
    if want:
        if not _is_simulated():
            return jsonify({"ok": False, "on": False,
                            "message": "AUTO is only available for the SIMULATED airport"}), 400
        _get_planner().start()      # one-time policy load + pool warmup
        _auto_planner.enable(sim)   # clear stale state; arming starts next step
        auto_on = True
    else:
        auto_on = False
        if _auto_planner is not None:
            _auto_planner.disable(sim)   # restore physics, leave planes cleared to land
    return jsonify({"ok": True, **_auto_status()})


@app.route('/recording/start', methods=['POST'])
def recording_start():
    ok = sim.start_recording()
    return jsonify({"ok": bool(ok), "recording": sim.is_recording()})


@app.route('/recording/stop', methods=['POST'])
def recording_stop():
    csv_text, filename = sim.stop_recording()
    return jsonify({
        "ok": csv_text is not None,
        "csv": csv_text or "",
        "filename": filename or "",
        "recording": sim.is_recording(),
    })


if __name__ == '__main__':
    init_simulation()
    app.run(host='127.0.0.1', port=5000, debug=False)

from flask import Flask, jsonify, request, send_from_directory

from environment import SimulationEnv

app = Flask(__name__, static_folder='static')

RADAR_SIDE = 800
sim = None


def init_simulation(airport="test", star_mode=False):
    global sim
    sim = SimulationEnv(
        radar_side=RADAR_SIDE,
        airport_name=airport,
        star_mode=star_mode,
    )


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/state', methods=['GET'])
def state():
    return jsonify(sim.get_state())


@app.route('/step', methods=['POST'])
def step():
    n = sim.fast_forward
    for _ in range(n):
        sim.step(1.0)
    return jsonify(sim.get_state())


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
    rate = int(data.get('rate', 60))
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
    data = request.get_json(force=True) or {}
    sim.restart(
        spawn_single=data.get('spawn_single'),
        star_mode=data.get('star_mode'),
        airport_name=data.get('airport_name'),
    )
    return jsonify(sim.get_state())


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

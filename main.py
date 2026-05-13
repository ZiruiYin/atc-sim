import argparse
import signal
import sys

import app as app_module
from app import app, init_simulation


def main():
    parser = argparse.ArgumentParser(description='ATC Radar web server')
    parser.add_argument(
        '--spawn_single',
        action='store_true',
        help='Spawn one aircraft at a time (next spawn after landed or improper exit)',
    )
    parser.add_argument(
        '--record',
        action='store_true',
        help='Write human RL training CSV under human_data/',
    )
    parser.add_argument(
        '--airport',
        default='egll',
        help='Airport ICAO (data/<icao>.json + data/<icao>_navigation.json must exist)',
    )
    parser.add_argument(
        '--star',
        action='store_true',
        help='Spawn aircraft on a random STAR procedure (edge-direction selector disabled)',
    )
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)

    args = parser.parse_args()
    init_simulation(spawn_single=args.spawn_single, record=args.record,
                    airport=args.airport, star_mode=args.star)

    def _shutdown_record(signum, frame):
        if app_module.recorder:
            app_module.recorder.close()
        sys.exit(0)

    if args.record:
        signal.signal(signal.SIGINT, _shutdown_record)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, _shutdown_record)

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()

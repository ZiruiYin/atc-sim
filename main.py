import argparse

from app import app, init_simulation


def main():
    parser = argparse.ArgumentParser(description='ATC Radar web server')
    parser.add_argument(
        '--airport',
        default='test',
        help='Airport to load (default: test = SIMULATED, STAR-following). '
             'Use egll for the EGLL airport (direction-based spawning).',
    )
    parser.add_argument(
        '--free_mode',
        action='store_true',
        help='Disable STAR procedures; spawn from radar edges (free vectoring). Default is STAR.',
    )
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)

    args = parser.parse_args()
    init_simulation(airport=args.airport, star_mode=not args.free_mode)
    # threaded so a /state poll isn't blocked while a /step (or a background
    # AUTO replan) is in flight — matters when the Hugging Face Space serves a
    # couple of tabs at once.
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()

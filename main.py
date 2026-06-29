import argparse

from app import app, init_simulation, NoNagleRequestHandler


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
    parser.add_argument('--threads', type=int, default=8,
                        help='waitress worker threads (concurrent requests)')
    parser.add_argument('--dev', action='store_true',
                        help='force the Werkzeug dev server instead of waitress')

    args = parser.parse_args()
    init_simulation(airport=args.airport, star_mode=not args.free_mode)

    # Serve via waitress (a real WSGI server) for both local and the HF deploy --
    # the Werkzeug dev server is "development only" and has socket quirks. Fall
    # back to the dev server (with NoNagle) if waitress isn't installed or --dev.
    if not args.dev:
        try:
            from waitress import serve
            print(f"[serve] waitress on {args.host}:{args.port} (threads={args.threads})")
            serve(app, host=args.host, port=args.port, threads=args.threads)
            return
        except ImportError:
            print("[serve] waitress not installed; falling back to dev server")
    # threaded so a /state poll isn't blocked while a /step (or a background AUTO
    # replan) is in flight. NoNagleRequestHandler sets TCP_NODELAY on responses.
    print(f"[serve] Werkzeug dev server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True,
            request_handler=NoNagleRequestHandler)


if __name__ == '__main__':
    main()

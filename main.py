import argparse

from app import app, init_simulation


def main():
    parser = argparse.ArgumentParser(description='ATC Radar web server')
    parser.add_argument(
        '--airport',
        default='test',
        help='Airport ICAO (default: test). EGLL is legacy and not deployed.',
    )
    parser.add_argument(
        '--star',
        action='store_true',
        help='Spawn aircraft on a random STAR procedure (edge-direction selector disabled)',
    )
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)

    args = parser.parse_args()
    init_simulation(airport=args.airport, star_mode=args.star)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()

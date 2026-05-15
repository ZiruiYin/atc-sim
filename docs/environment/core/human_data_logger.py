"""CSV recorder for RL training — one row per aircraft per simulated second."""

import csv
import io
import os
from datetime import datetime


FIELDNAMES = [
    'sim_time',
    'callsign',
    'x_nm',
    'y_nm',
    'altitude',
    'heading',
    'airspeed',
    'target_altitude',
    'target_heading',
    'target_airspeed',
    'loc',
    'gs',
    'on_ground',
    'star',
    'target_wpt',
    'terminal',
    'cmd_raw',
    'cmd_abort',
    'cmd_heading',
    'cmd_altitude_ft',
    'cmd_speed',
    'cmd_waypoint',
    'cmd_holding_wpt',
    'cmd_holding_turn',
    'cmd_landing_runway',
    'cmd_expedite_alt',
    'cmd_expedite_speed',
]


def _blank_action_cells():
    return {
        'cmd_raw': '',
        'cmd_abort': '',
        'cmd_heading': '',
        'cmd_altitude_ft': '',
        'cmd_speed': '',
        'cmd_waypoint': '',
        'cmd_holding_wpt': '',
        'cmd_holding_turn': '',
        'cmd_landing_runway': '',
        'cmd_expedite_alt': '',
        'cmd_expedite_speed': '',
    }


def _parse_command_for_log(cmd_string):
    """
    Parse command string into sparse action fields (mirrors Aircraft.process_command token rules).
    On grammar failure, returns blanks with only cmd_raw set.
    """
    blanks = _blank_action_cells()
    if cmd_string is None:
        return blanks

    stripped = cmd_string.strip()
    if not stripped:
        return blanks

    commands = stripped.upper().split()
    if commands[0] == 'A':
        commands.insert(1, None)

    if len(commands) % 2 != 0:
        blanks['cmd_raw'] = stripped
        return blanks

    command_pairs = []
    idx = 0
    while idx < len(commands):
        command_pairs.append((commands[idx], commands[idx + 1]))
        idx += 2

    command_types = [pair[0] for pair in command_pairs]

    if 'H' in command_types and ('C' in command_types or 'L' in command_types):
        blanks['cmd_raw'] = stripped
        return blanks

    a_index = -1
    if 'A' in command_types:
        a_index = command_types.index('A')

    if 'L' in command_types:
        l_index = command_types.index('L')
    else:
        l_index = len(command_types)

    for j, cmd_type in enumerate(command_types):
        if cmd_type in ('C', 'S', 'H'):
            if a_index != -1 and j < a_index:
                blanks['cmd_raw'] = stripped
                return blanks
            if j >= l_index:
                blanks['cmd_raw'] = stripped
                return blanks

    out = _blank_action_cells()

    for cmd_type, param in command_pairs:
        if cmd_type == 'A':
            out['cmd_abort'] = 'Y'

        elif cmd_type == 'C':
            if param is None:
                continue
            parts = param.split(';')
            first = parts[0]
            if len(first) == 3 and first.isdigit():
                out['cmd_heading'] = first
                if len(parts) > 1 and parts[1] in ('L', 'R'):
                    out['cmd_heading'] = first + ';' + parts[1]
            elif len(first) <= 2 and first.isdigit():
                out['cmd_altitude_ft'] = str(int(first) * 1000)
                if len(parts) > 1 and parts[1] == 'X':
                    out['cmd_expedite_alt'] = 'Y'
            else:
                out['cmd_waypoint'] = parts[0]
                if len(parts) > 1 and parts[1] in ('L', 'R'):
                    out['cmd_waypoint'] = parts[0] + ';' + parts[1]

        elif cmd_type == 'S':
            if param is None:
                continue
            parts = param.split(';')
            out['cmd_speed'] = parts[0]
            if len(parts) > 1 and parts[1] == 'X':
                out['cmd_expedite_speed'] = 'Y'

        elif cmd_type == 'H':
            if param is None:
                continue
            parts = param.split(';')
            out['cmd_holding_wpt'] = parts[0]
            if len(parts) > 1 and parts[1] in ('L', 'R'):
                out['cmd_holding_turn'] = parts[1]

        elif cmd_type == 'L':
            if param:
                out['cmd_landing_runway'] = param

    out['cmd_raw'] = stripped
    return out


class HumanDataRecorder:
    """Append-only CSV logger. Two sinks:

    - File (default): writes to human_data/{single_plane,multiple_planes}/{ts}.csv,
      flushed every timestep. Used by long-running CLI rollouts.
    - In-memory (in_memory=True): buffers rows in a StringIO; to_csv() returns
      the text. Used by the browser Record button so the CSV can be downloaded
      to the user's machine (works under both Flask and Pyodide).
    """

    def __init__(self, spawn_single=False, in_memory=False):
        self.spawn_single = spawn_single
        self.in_memory = in_memory
        # __file__ is at environment/core/human_data_logger.py; project root is 3 dirs up.
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sub = 'single_plane' if spawn_single else 'multiple_planes'
        self._human_dir = os.path.join(project_root, 'human_data', sub)
        self._filepath = None
        self._file = None
        self._writer = None
        self._pending_actions = {}

    def start(self):
        if self.in_memory:
            self._file = io.StringIO()
        else:
            os.makedirs(self._human_dir, exist_ok=True)
            name = datetime.now().strftime('%Y%m%d_%H%M%S.csv')
            self._filepath = os.path.join(self._human_dir, name)
            self._file = open(self._filepath, 'w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES, extrasaction='raise')
        self._writer.writeheader()
        if not self.in_memory:
            self._file.flush()

    def to_csv(self):
        """Return the buffered CSV text. Only meaningful in in-memory mode."""
        if self._file is None:
            return ''
        return self._file.getvalue() if self.in_memory else ''

    def suggested_filename(self):
        """A reasonable filename for downloads: <ts>_<mode>.csv."""
        mode = 'single' if self.spawn_single else 'multiple'
        return datetime.now().strftime(f'%Y%m%d_%H%M%S_{mode}.csv')

    def enqueue_command_action(self, callsign, cmd_raw):
        """Queue parsed sparse action fields for callsign until the next logged timestep."""
        cs = callsign.upper().strip()
        self._pending_actions[cs] = _parse_command_for_log(cmd_raw)

    @staticmethod
    def _state_row(sim_time, ac, nmpp, ax, ay, on_ground_str, terminal):
        x_nm = (ac.x - ax) * nmpp
        y_nm = -(ac.y - ay) * nmpp
        star_name = ac.star_name if ac.star else ''
        target_wpt = ac.target_wpt or ''
        return {
            'sim_time': sim_time,
            'callsign': ac.callsign,
            'x_nm': round(x_nm, 6),
            'y_nm': round(y_nm, 6),
            'altitude': ac.altitude,
            'heading': ac.heading,
            'airspeed': ac.airspeed,
            'target_altitude': ac.target_altitude,
            'target_heading': ac.target_heading,
            'target_airspeed': ac.target_airspeed,
            'loc': ac.loc_intercepted,
            'gs': ac.gs_intercepted,
            'on_ground': on_ground_str,
            'star': star_name,
            'target_wpt': target_wpt,
            'terminal': terminal,
        }

    def log_timestep(self, env, removal_terminal):
        """
        removal_terminal: map callsign -> 'LANDED' | 'IMPROPER_EXIT' for aircraft
        leaving this tick (still present in aircraft_list when called).
        """
        pending_snap = dict(self._pending_actions)
        self._pending_actions = {}

        nmpp = env.nm_per_pixel
        ax = env.airport_x
        ay = env.airport_y
        t = env.sim_time

        for ac in env.aircraft_list.values():
            cs = ac.callsign
            terminal = ''
            if cs in removal_terminal:
                terminal = removal_terminal[cs]

            on_ground_str = ''
            if ac.on_ground:
                on_ground_str = str(ac.on_ground)

            base = HumanDataRecorder._state_row(t, ac, nmpp, ax, ay, on_ground_str, terminal)
            if cs in pending_snap:
                action = pending_snap.pop(cs)
            else:
                action = _blank_action_cells()

            row = {**base, **action}
            self._writer.writerow(row)

        if not self.in_memory:
            self._file.flush()

    def close(self):
        if self._file is not None:
            if not self.in_memory:
                self._file.flush()
                self._file.close()
        # In memory mode we keep _file around so to_csv() still works after close().
        self._writer = None

import socket
import sys
import getopt
import os
import time
import math

PI = 3.14159265359
data_size = 2**17

# =========================
# COMMAND LINE HELP
# =========================
ophelp = 'Options:\n'
ophelp += ' --host, -H <host>    TORCS server host. [localhost]\n'
ophelp += ' --port, -p <port>    TORCS port. [3001]\n'
ophelp += ' --id, -i <id>        ID for server. [SCR]\n'
ophelp += ' --steps, -m <#>      Maximum simulation steps. 1 sec ~ 50 steps. [100000]\n'
ophelp += ' --episodes, -e <#>   Maximum learning episodes. [1]\n'
ophelp += ' --track, -t <track>  Your name for this track. Used for learning. [unknown]\n'
ophelp += ' --stage, -s <#>      0=warm up, 1=qualifying, 2=race, 3=unknown. [3]\n'
ophelp += ' --debug, -d          Output full telemetry.\n'
ophelp += ' --help, -h           Show this help.\n'
ophelp += ' --version, -v        Show current version.'
usage = 'Usage: %s [ophelp [optargs]] \n' % sys.argv[0]
usage = usage + ophelp
version = "20130505-2"

# =========================
# UTILITY FUNCTIONS
# =========================
def clip(v, lo, hi):
    return max(lo, min(hi, v))

def destringify(v):
    """Convert token list to float / list of floats when possible."""
    if not v:
        return v
    if len(v) == 1:
        try:
            return float(v[0])
        except Exception:
            return v[0]
    out = []
    for x in v:
        try:
            out.append(float(x))
        except Exception:
            out.append(x)
    return out

# =========================
# CLIENT
# =========================
class Client():
    def __init__(self):
        self.host = 'localhost'
        self.port = 3001
        self.sid = 'SCR'
        self.maxEpisodes = 1
        self.trackname = 'unknown'
        self.stage = 3
        self.debug = False
        self.maxSteps = 100000  # ~50 steps/sec

        self.parse_the_command_line()

        self.S = ServerState()
        self.R = DriverAction()
        self.setup_connection()

    def setup_connection(self):
        self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.so.settimeout(1)

        n_fail = 5
        while True:
            angles = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
            initmsg = f"{self.sid}(init {angles})"

            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error:
                sys.exit(-1)

            try:
                data, _ = self.so.recvfrom(data_size)
                sockdata = data.decode('utf-8')
            except socket.timeout:
                print("Waiting for server on %d............" % self.port)
                print("Count Down : " + str(n_fail))
                if n_fail < 0:
                    print("relaunch torcs")
                    os.system('pkill torcs')
                    time.sleep(1.0)
                    os.system('torcs -nofuel -nodamage -nolaptime &')
                    time.sleep(1.0)
                    os.system('sh autostart.sh')
                    n_fail = 5
                n_fail -= 1
                continue

            if '***identified***' in sockdata:
                print("Client connected on %d.............." % self.port)
                break

    def parse_the_command_line(self):
        try:
            opts, args = getopt.getopt(
                sys.argv[1:], 'H:p:i:m:e:t:s:dhv',
                ['host=', 'port=', 'id=', 'steps=', 'episodes=', 'track=', 'stage=', 'debug', 'help', 'version']
            )
        except getopt.error as why:
            print('getopt error: %s\n%s' % (why, usage))
            sys.exit(-1)

        for opt, arg in opts:
            if opt in ('-h', '--help'):
                print(usage)
                sys.exit(0)
            if opt in ('-d', '--debug'):
                self.debug = True
            if opt in ('-H', '--host'):
                self.host = arg
            if opt in ('-i', '--id'):
                self.sid = arg
            if opt in ('-t', '--track'):
                self.trackname = arg
            if opt in ('-s', '--stage'):
                self.stage = int(arg)
            if opt in ('-p', '--port'):
                self.port = int(arg)
            if opt in ('-e', '--episodes'):
                self.maxEpisodes = int(arg)
            if opt in ('-m', '--steps'):
                self.maxSteps = int(arg)
            if opt in ('-v', '--version'):
                print('%s %s' % (sys.argv[0], version))
                sys.exit(0)

        if len(args) > 0:
            print('Superflous input? %s\n%s' % (', '.join(args), usage))
            sys.exit(-1)

    def get_servers_input(self):
        if not self.so:
            return
        while True:
            try:
                data, _ = self.so.recvfrom(data_size)
                sockdata = data.decode('utf-8')
            except socket.timeout:
                continue

            if '***identified***' in sockdata:
                continue
            if '***shutdown***' in sockdata:
                print("Server has stopped the race.")
                self.shutdown()
                return
            if '***restart***' in sockdata:
                print("Server has restarted the race.")
                self.shutdown()
                return
            if not sockdata:
                continue

            self.S.parse_server_str(sockdata)
            break

    def respond_to_server(self):
        if not self.so:
            return
        try:
            self.so.sendto(repr(self.R).encode(), (self.host, self.port))
        except socket.error as emsg:
            print("Error sending to server: %s" % str(emsg))
            sys.exit(-1)

    def shutdown(self):
        if not self.so:
            return
        self.so.close()
        self.so = None

# =========================
# SERVER STATE
# =========================
class ServerState():
    def __init__(self):
        self.d = {}

    def parse_server_str(self, server_string):
        servstr = server_string.strip()[:-1]  # drop trailing ')'
        sslisted = servstr.strip().lstrip('(').rstrip(')').split(')(')
        for i in sslisted:
            w = i.split(' ')
            self.d[w[0]] = destringify(w[1:])

# =========================
# DRIVER ACTION
# =========================
class DriverAction():
    """
    TORCS expects something like:
    (accel 1)(brake 0)(gear 1)(steer 0)(clutch 0)(focus -90 -45 0 45 90)(meta 0)
    Including focus is safer (some setups assume it's always present).
    """
    def __init__(self):
        self.d = {
            'accel': 0.2,
            'brake': 0.0,
            'clutch': 0.0,
            'gear': 1,
            'steer': 0.0,
            'focus': [-90, -45, 0, 45, 90],
            'meta': 0
        }

    def clip_to_limits(self):
        self.d['steer'] = clip(self.d['steer'], -1, 1)
        self.d['brake'] = clip(self.d['brake'], 0, 1)
        self.d['accel'] = clip(self.d['accel'], 0, 1)
        self.d['clutch'] = clip(self.d['clutch'], 0, 1)

        if self.d['gear'] not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            self.d['gear'] = 1
        if self.d['meta'] not in [0, 1]:
            self.d['meta'] = 0

        f = self.d.get('focus')
        if type(f) is not list or min(f) < -180 or max(f) > 180:
            self.d['focus'] = [-90, -45, 0, 45, 90]

    def __repr__(self):
        self.clip_to_limits()
        out = ""
        for k in self.d:
            v = self.d[k]
            if isinstance(v, list):
                out += f"({k} " + " ".join(str(int(x)) for x in v) + ")"
            elif k in ('gear', 'meta'):
                out += f"({k} {int(v)})"
            else:
                out += f"({k} {float(v):.3f})"
        return out

# =========================
# USER CONFIGURABLE PARAMETERS
# =========================
TARGET_SPEED = 80
STEER_GAIN = 40
CENTERING_GAIN = 0.40
BRAKE_THRESHOLD = 0.39
GEAR_SPEEDS = [0, 50, 80, 120, 150, 200]
ENABLE_TRACTION_CONTROL = True

# Straight-line accel
STRAIGHT_STEER_THRESHOLD = 0.03
STRAIGHT_ACCEL_GAIN = 0.6

# Launch help
LAUNCH_SPEED = 5.0
LAUNCH_ACCEL = 1.0

# ===== Steering smoothing + corner anticipation (LIDAR/track sensors) =====
# Smoother steering: lower alpha + lower rate limit
STEER_SMOOTH_ALPHA = 0.10      # was 0.18 (lower = smoother)
STEER_RATE_LIMIT = 0.035       # was 0.06  (lower = smoother)

# Look much further ahead with LIDAR
LOOKAHEAD_TRIGGER_DIST = 120.0 # was 55.0 (bigger = earlier anticipation)
CORNER_LOOKAHEAD_GAIN = 0.90   # stronger pre-steer when corner is far ahead

# Corner speed reduction
CORNER_SPEED_FRACTION = 0.50   # slow to half of TARGET_SPEED in corners

# =========================
# INTERNAL STATE (for smoothing)
# =========================
_prev_steer = 0.0

# =========================
# LIDAR HELPERS
# =========================
def _lidar_bias_and_centre(track):
    """
    Uses TORCS 'track' sensors (LIDAR-like distances).
    Returns:
      bias in [-1, 1]  (signed: + left, - right)
      centre distance (front sensor)
    """
    if not isinstance(track, list) or len(track) < 19:
        return 0.0, 999.0

    centre = float(track[9])
    left = sum(float(x) for x in track[0:9]) / 9.0
    right = sum(float(x) for x in track[10:19]) / 9.0

    denom = (left + right) + 1e-6
    bias = (right - left) / denom
    return clip(bias, -1.0, 1.0), centre

def _corner_strength_from_lidar(track):
    """
    Returns a corner strength 0..1 based on how close the forward LIDAR is.
    0 => far ahead (straight), 1 => very near (corner imminent).
    """
    _, centre = _lidar_bias_and_centre(track)
    if centre >= LOOKAHEAD_TRIGGER_DIST:
        return 0.0
    strength = (LOOKAHEAD_TRIGGER_DIST - centre) / LOOKAHEAD_TRIGGER_DIST
    return clip(strength, 0.0, 1.0)

# =========================
# CONTROL LOGIC
# =========================
def calculate_steering(S):
    global _prev_steer

    angle = S.get('angle', 0.0)
    trackPos = S.get('trackPos', 0.0)

    # Base: align with track + recentre
    raw = (angle * STEER_GAIN / math.pi) - (trackPos * CENTERING_GAIN)

    # LIDAR: anticipate corners far ahead
    track = S.get('track', None)
    bias, centre = _lidar_bias_and_centre(track)

    if centre < LOOKAHEAD_TRIGGER_DIST:
        strength = (LOOKAHEAD_TRIGGER_DIST - centre) / LOOKAHEAD_TRIGGER_DIST  # 0..1
        raw += (bias * CORNER_LOOKAHEAD_GAIN * strength)

    desired = clip(raw, -1.0, 1.0)

    # Low-pass smooth
    filtered = _prev_steer + STEER_SMOOTH_ALPHA * (desired - _prev_steer)

    # Rate limit (prevents sudden yanks)
    delta = clip(filtered - _prev_steer, -STEER_RATE_LIMIT, STEER_RATE_LIMIT)
    new_steer = clip(_prev_steer + delta, -1.0, 1.0)

    _prev_steer = new_steer
    return new_steer

def calculate_throttle(S, R):
    speed = S.get('speedX', 0.0)

    # HARD launch
    if speed < LAUNCH_SPEED:
        return LAUNCH_ACCEL

    # Corner anticipation from LIDAR: reduce effective target speed down to half
    corner_strength = _corner_strength_from_lidar(S.get('track', None))  # 0..1
    effective_target = TARGET_SPEED * (1.0 - (1.0 - CORNER_SPEED_FRACTION) * corner_strength)
    # effective_target ranges from TARGET_SPEED (straight) down to TARGET_SPEED*0.5 (corner)

    # Straight-line boost only when truly straight AND under effective target
    if abs(R['steer']) < STRAIGHT_STEER_THRESHOLD and speed < effective_target:
        return min(1.0, R['accel'] + STRAIGHT_ACCEL_GAIN)

    # Normal speed control relative to effective target (also respects steering)
    steer_penalty = abs(R['steer']) * 2.5
    if speed < effective_target - steer_penalty:
        accel = R['accel'] + 0.35
    else:
        accel = R['accel'] - 0.45

    # Low-speed recovery
    if speed < 10:
        accel += 1 / (speed + 0.1)

    return clip(accel, 0.0, 1.0)

def apply_brakes(S):
    speed = S.get('speedX', 0.0)
    angle = S.get('angle', 0.0)

    # If we’re stopped, never brake
    if speed < 2.0:
        return 0.0

    # Use effective target (so corners naturally encourage braking too)
    corner_strength = _corner_strength_from_lidar(S.get('track', None))
    effective_target = TARGET_SPEED * (1.0 - (1.0 - CORNER_SPEED_FRACTION) * corner_strength)

    if speed > effective_target + 2:
        return 0.25
    if abs(angle) > BRAKE_THRESHOLD:
        return 0.4
    return 0.0

def shift_gears(S):
    speed = S.get('speedX', 0.0)
    gear = 1
    for i, sp in enumerate(GEAR_SPEEDS):
        if speed > sp:
            gear = i + 1
    return min(gear, 6)

def traction_control(S, accel):
    if not ENABLE_TRACTION_CONTROL:
        return accel

    w = S.get('wheelSpinVel', [0, 0, 0, 0])
    if not isinstance(w, list) or len(w) < 4:
        return accel

    slip = (w[2] + w[3]) - (w[0] + w[1])
    if slip > 2:
        accel -= 0.1
    return max(0.0, accel)

# =========================
# MAIN DRIVE FUNCTION
# =========================
def drive(c):
    S, R = c.S.d, c.R.d

    R['steer'] = calculate_steering(S)
    R['accel'] = calculate_throttle(S, R)
    R['brake'] = apply_brakes(S)
    R['accel'] = traction_control(S, R['accel'])
    R['gear'] = shift_gears(S)

    # Safety: never press brake and throttle together at low speed
    if S.get('speedX', 0.0) < 5.0 and R['brake'] > 0.0:
        R['brake'] = 0.0

# =========================
# MAIN LOOP
# =========================
if __name__ == "__main__":
    C = Client()
    for _ in range(C.maxSteps):
        C.get_servers_input()
        if not C.so:
            break
        drive(C)
        C.respond_to_server()
    C.shutdown()

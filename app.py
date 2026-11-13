"""Codezilla Kinetic Vision: pose analysis and form feedback for cricket technique.

Three input sources:
  Simulation   animated figure, needs no camera or video (default, safest for a live demo)
  Upload a clip  any short mp4/mov/avi
  Webcam       the camera of the machine that runs this app

Pose estimation uses MediaPipe when it is installed. Without it, uploads and webcam
fall back to motion tracking only. The simulation always works.
"""

import math
import os
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

try:
    import mediapipe as mp

    _POSE = mp.solutions.pose
    POSE_OK = True
except Exception as _err:
    _POSE = None
    POSE_OK = False
    POSE_ERR = f"{type(_err).__name__}: {_err}"
else:
    POSE_ERR = ""


# ----------------------------------------------------------------------------
# Page setup and brand styling
# ----------------------------------------------------------------------------


def _favicon():
    img = Image.new("RGB", (64, 64), (10, 10, 10))
    ImageDraw.Draw(img).rectangle((24, 10, 40, 54), fill=(255, 85, 0))
    return img


st.set_page_config(
    page_title="Codezilla Kinetic Vision",
    layout="wide",
    page_icon=_favicon(),
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap');
:root{--ink:#0A0A0A;--panel:#121212;--raise:#171717;--line:#2A2A2A;--text:#F2F2F2;--mute:#9C9C9C;--orange:#FF5500;}
.stApp{background:var(--ink);color:var(--text);}
.stApp, .stApp *:not([data-testid="stIconMaterial"]){font-family:'Barlow','Segoe UI',Arial,sans-serif;}
.stApp .kv-display{font-family:'Barlow Condensed','Arial Narrow',Arial,sans-serif !important;}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"]{display:none !important;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:2.2rem;max-width:1500px;}
[data-testid="stSidebar"]{background:var(--panel);border-right:1px solid var(--line);}
.stApp :focus-visible{outline:2px solid var(--orange);outline-offset:2px;}

.stButton>button{border-radius:3px;font-weight:600;font-size:1.05rem;height:3rem;width:100%;}
.stButton>button p{color:inherit;}
.stButton>button[kind="primary"]{background:var(--orange);color:#0A0A0A;border:1px solid var(--orange);}
.stButton>button[kind="primary"]:hover{background:#FF7033;border-color:#FF7033;color:#0A0A0A;}
.stButton>button[kind="secondary"]{background:transparent;color:var(--text);border:1px solid #444;}
.stButton>button[kind="secondary"]:hover{border-color:var(--orange);color:var(--orange);}

button[data-baseweb="tab"]{font-size:1.15rem;font-weight:600;padding:.6rem 1.4rem;}
[data-baseweb="tab-highlight"]{background:var(--orange) !important;height:3px;}
[data-baseweb="tab-border"]{background:var(--line);}
[data-testid="stImage"] img{border:1px solid var(--line);border-radius:3px;}

.kv-brand{display:flex;align-items:stretch;gap:16px;padding-bottom:18px;margin-bottom:6px;border-bottom:1px solid var(--line);}
.kv-mark{width:12px;background:var(--orange);}
.kv-title{font-size:2.5rem;font-weight:700;line-height:1;letter-spacing:.5px;}
.kv-sub{color:var(--mute);font-size:1.05rem;margin-top:6px;}

.kv-status{display:flex;flex-wrap:wrap;gap:6px 26px;color:var(--mute);font-size:.95rem;margin:4px 0 10px;}
.kv-status b{color:var(--text);font-weight:600;}

.kv-score{border:1px solid var(--line);background:var(--panel);padding:16px 18px 18px;}
.kv-num{font-size:6.2rem;font-weight:700;line-height:.85;color:var(--orange);}
.kv-numlab{color:var(--mute);margin:8px 0 12px;font-size:.95rem;}
.kv-bar{height:6px;background:#262626;}
.kv-bar span{display:block;height:100%;background:var(--orange);}

.kv-metric{border:1px solid var(--line);border-left:3px solid #666;background:var(--panel);padding:11px 14px 10px;margin-top:10px;}
.kv-metric.ok{border-left-color:var(--orange);}
.kv-metric.off{border-left-color:var(--text);}
.kv-head{display:flex;justify-content:space-between;align-items:baseline;}
.kv-name{font-size:1rem;color:var(--text);}
.kv-val{font-size:1.9rem;font-weight:600;line-height:1;}
.ok .kv-val{color:var(--orange);}
.kv-track{position:relative;height:8px;background:#1F1F1F;margin:10px 0 7px;}
.kv-zone{position:absolute;top:0;bottom:0;background:rgba(255,85,0,.38);}
.kv-mark2{position:absolute;top:-4px;width:3px;height:16px;background:var(--text);margin-left:-1px;}
.ok .kv-mark2{background:var(--orange);}
.kv-foot{display:flex;justify-content:space-between;color:var(--mute);font-size:.85rem;}
.kv-note{margin-top:12px;padding:13px 15px;background:var(--raise);border-left:3px solid var(--orange);font-size:1rem;line-height:1.4;}

.kv-step{display:grid;grid-template-columns:56px 1fr;gap:18px;padding:16px 0;border-top:1px solid var(--line);}
.kv-stepnum{font-size:2.6rem;font-weight:700;color:var(--orange);line-height:1;}
.kv-steptitle{font-size:1.5rem;font-weight:600;line-height:1.1;margin-bottom:4px;}
.kv-steptext{color:#CFCFCF;max-width:68ch;}

.kv-rowh, .kv-row{display:grid;grid-template-columns:minmax(170px,1.1fr) 2.2fr 1.5fr;gap:28px;padding:15px 0;border-top:1px solid var(--line);}
.kv-rowh{color:var(--mute);font-size:.9rem;border-top:none;padding-bottom:8px;}
.kv-role{font-size:1.45rem;font-weight:600;line-height:1.1;}
.kv-row div:nth-child(3){color:var(--orange);}
@media (max-width:800px){.kv-rowh{display:none;}.kv-row{grid-template-columns:1fr;gap:6px;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def _h(s):
    """Collapse indented HTML so Markdown never mistakes it for a code block."""
    return "".join(line.strip() for line in s.splitlines())


# ----------------------------------------------------------------------------
# Geometry and metric definitions
# ----------------------------------------------------------------------------

W, H = 960, 540
ORANGE = (0, 85, 255)  # BGR
WHITE = (242, 242, 242)

BONES = [
    ("ls", "rs"), ("lh", "rh"), ("ls", "lh"), ("rs", "rh"),
    ("ls", "le"), ("le", "lw"), ("rs", "re"), ("re", "rw"),
    ("lh", "lk"), ("lk", "la"), ("rh", "rk"), ("rk", "ra"),
]
JOINTS = ["ls", "rs", "le", "re", "lw", "rw", "lh", "rh", "lk", "rk", "la", "ra"]


def angle(a, b, c):
    """Interior angle at b, in degrees, between segments b-a and b-c."""
    ba = np.subtract(a, b)
    bc = np.subtract(c, b)
    n = np.linalg.norm(ba) * np.linalg.norm(bc)
    if n < 1e-6:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(ba, bc) / n, -1.0, 1.0))))


def mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def lean_from_vertical(top, bottom):
    return abs(math.degrees(math.atan2(top[0] - bottom[0], bottom[1] - top[1])))


MODES = {
    "Batting stance": [
        dict(key="knee", label="Knee flexion", unit="deg", target=(130, 165), scale=(90, 180),
             low="Knees are bent too deeply. Stand a little taller so you can move out of the stance.",
             high="Legs are too straight. Soften the knees to stay loaded and ready."),
        dict(key="lean", label="Back lean", unit="deg", target=(8, 25), scale=(0, 45),
             low="Torso is too upright. Tilt slightly forward from the hips.",
             high="Leaning too far forward. Lift the chest to keep your balance."),
        dict(key="base", label="Base width", unit="%", target=(40, 75), scale=(0, 120),
             low="Feet are too close together. Widen the stance to about shoulder width.",
             high="Feet are too far apart. Bring them in so you can move quickly."),
        dict(key="head", label="Head over base", unit="%", target=(0, 35), scale=(0, 100),
             low="", high="Head has drifted ahead of the base. Keep it above the middle of your feet."),
    ],
    "Bowling action": [
        dict(key="knee", label="Braced front knee", unit="deg", target=(160, 180), scale=(110, 180),
             low="Front knee is collapsing on landing. Brace the leg and push through it.", high=""),
        dict(key="elbow", label="Bowling elbow", unit="deg", target=(165, 180), scale=(120, 180),
             low="Bowling arm is bending. Keep it straight through release.", high=""),
        dict(key="lean", label="Back lean", unit="deg", target=(10, 30), scale=(0, 50),
             low="Torso is too upright. Lean into the delivery.",
             high="Falling away too far. Stay tall through the crease."),
        dict(key="front", label="Front arm height", unit="%", target=(50, 110), scale=(0, 150),
             low="Front arm is dropping. Reach up and out to guide the delivery.",
             high="Front arm is too high. Bring it down toward the target."),
    ],
}


class ArmTracker:
    """Works out which arm is the bowling arm: the wrist that travels the most."""

    def __init__(self):
        self.travel = {"l": 0.0, "r": 0.0}
        self.prev = None
        self.side = "r"

    def update(self, sk):
        cur = {"l": sk["lw"], "r": sk["rw"]}
        if self.prev:
            for s in "lr":
                self.travel[s] = 0.95 * self.travel[s] + dist(cur[s], self.prev[s])
            self.side = "l" if self.travel["l"] > self.travel["r"] else "r"
        self.prev = cur


def measure(sk, mode, tracker):
    shoulders, hips = mid(sk["ls"], sk["rs"]), mid(sk["lh"], sk["rh"])
    torso = max(dist(shoulders, hips), 1.0)
    knees = (angle(sk["lh"], sk["lk"], sk["la"]), angle(sk["rh"], sk["rk"], sk["ra"]))
    out = {"lean": lean_from_vertical(shoulders, hips)}
    if mode == "Batting stance":
        leg = max((dist(sk["lh"], sk["la"]) + dist(sk["rh"], sk["ra"])) / 2, 1.0)
        out["knee"] = sum(knees) / 2
        out["base"] = abs(sk["la"][0] - sk["ra"][0]) / leg * 100
        out["head"] = abs(sk["nose"][0] - mid(sk["la"], sk["ra"])[0]) / torso * 100
    else:
        tracker.update(sk)
        b = tracker.side
        f = "l" if b == "r" else "r"
        out["knee"] = max(knees)
        out["elbow"] = angle(sk[b + "s"], sk[b + "e"], sk[b + "w"])
        out["front"] = (sk[f + "s"][1] - sk[f + "w"][1]) / torso * 100
    return out


def metric_score(v, lo, hi):
    if lo <= v <= hi:
        return 1.0
    d = lo - v if v < lo else v - hi
    return max(0.0, 1.0 - d / ((hi - lo) * 0.8 + 1e-6))


def overall_score(vals, mode):
    spec = MODES[mode]
    return round(100 * sum(metric_score(vals[m["key"]], *m["target"]) for m in spec) / len(spec))


def coaching(vals, mode):
    worst, worst_s = None, 1.0
    for m in MODES[mode]:
        s = metric_score(vals[m["key"]], *m["target"])
        if s < worst_s:
            worst, worst_s = m, s
    if worst is None:
        return "All measurements are in range. Hold this position."
    v, (lo, hi) = vals[worst["key"]], worst["target"]
    text = worst["low"] if v < lo else worst["high"]
    return text or f"Adjust {worst['label'].lower()}."


# ----------------------------------------------------------------------------
# Simulation: a procedural figure, so the demo needs no equipment
# ----------------------------------------------------------------------------


def _ik(hip, ankle, l1, l2):
    """Two-bone solver. Returns the knee position, bending toward +x."""
    dx, dy = ankle[0] - hip[0], ankle[1] - hip[1]
    full = math.hypot(dx, dy) or 1e-3
    ux, uy = dx / full, dy / full
    d = min(full, l1 + l2 - 1e-3)
    a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    px, py = hip[0] + ux * a, hip[1] + uy * a
    k1, k2 = (px - uy * h, py + ux * h), (px + uy * h, py - ux * h)
    return k1 if k1[0] > k2[0] else k2


def _arm(shoulder, a1, a2, l1, l2):
    e = (shoulder[0] + l1 * math.sin(a1), shoulder[1] + l1 * math.cos(a1))
    w = (e[0] + l2 * math.sin(a2), e[1] + l2 * math.cos(a2))
    return e, w


def synth_pose(mode, t, w=W, h=H):
    u = h / 720.0
    ground, leg = 0.88 * h, 120 * u
    tau = 2 * math.pi
    if mode == "Batting stance":
        cx = 0.45 * w
        hip_y = ground - (218 + 12 * math.sin(tau * t / 2.4)) * u
        lean = math.radians(14 + 8 * math.sin(tau * t / 3.1))
        front, back = (cx + 55 * u, ground), (cx - 70 * u, ground)
    else:
        cx = 0.40 * w
        hip_y = ground - (219 + 7 * math.sin(tau * t / 3.3 + 1)) * u
        lean = math.radians(20 + 12 * math.sin(tau * t / 2.4 - 1))
        front, back = (cx + 90 * u, ground), (cx - 130 * u, ground - 40 * u)

    lh, rh = (cx, hip_y), (cx - 6 * u, hip_y)
    ls = (cx + 150 * u * math.sin(lean), hip_y - 150 * u * math.cos(lean))
    rs = (ls[0] - 10 * u, ls[1] + 4 * u)
    nose = (ls[0] + 45 * u * math.sin(lean + 0.1), ls[1] - 45 * u * math.cos(lean + 0.1))

    if mode == "Batting stance":
        p = (t % 4.0) / 4.0
        if p < 0.55:
            s = 0.0
        elif p < 0.75:
            s = (p - 0.55) / 0.2
        elif p < 0.9:
            s = 1 - (p - 0.75) / 0.15
        else:
            s = 0.0
        s = s * s * (3 - 2 * s)
        a1 = math.radians(25 + 115 * s)
        a2 = a1 + math.radians(35)
        le, lw = _arm(ls, a1, a2, 80 * u, 75 * u)
        re, rw = _arm(rs, a1 - 0.05, a2 - 0.05, 80 * u, 75 * u)
    else:
        phi = tau * ((t % 2.4) / 2.4)
        bend = math.radians(14 + 10 * math.sin(tau * t / 3.3))
        re, rw = _arm(rs, phi, phi - bend, 85 * u, 80 * u)
        fa = 2.3 + 0.5 * math.sin(tau * t / 2.9)
        le, lw = _arm(ls, fa, fa + 0.15, 85 * u, 80 * u)

    return {
        "nose": nose, "ls": ls, "rs": rs, "le": le, "re": re, "lw": lw, "rw": rw,
        "lh": lh, "rh": rh, "lk": _ik(lh, front, leg, leg), "rk": _ik(rh, back, leg, leg),
        "la": front, "ra": back,
    }


_BG = {}


def _background(w, h):
    if (w, h) not in _BG:
        bg = np.full((h, w, 3), (10, 10, 10), np.uint8)
        for x in range(0, w, 60):
            cv2.line(bg, (x, 0), (x, h), (17, 17, 17), 1)
        for y in range(0, h, 60):
            cv2.line(bg, (0, y), (w, y), (17, 17, 17), 1)
        g = int(0.88 * h)
        cv2.line(bg, (0, g), (w, g), (48, 48, 48), 2)
        for x in (0.18, 0.82):
            cv2.line(bg, (int(x * w), g), (int(x * w), g + 14), (48, 48, 48), 2)
        _BG[(w, h)] = bg
    return _BG[(w, h)].copy()


def synth_scene(sk, mode, w=W, h=H):
    u = h / 720.0
    img = _background(w, h)
    body, thick = (46, 46, 46), max(2, int(18 * u))
    for a, b in BONES:
        cv2.line(img, tuple(map(int, sk[a])), tuple(map(int, sk[b])), body, thick, cv2.LINE_AA)
    cv2.circle(img, tuple(map(int, sk["nose"])), int(24 * u), body, -1, cv2.LINE_AA)
    if mode == "Batting stance":
        e, wr = np.array(sk["le"]), np.array(sk["lw"])
        d = (wr - e) / (np.linalg.norm(wr - e) + 1e-6)
        tip = wr + d * 170 * u
        cv2.line(img, tuple(map(int, wr)), tuple(map(int, tip)), (52, 74, 100), max(2, int(13 * u)), cv2.LINE_AA)
    else:
        cv2.circle(img, tuple(map(int, sk["rw"])), max(3, int(9 * u)), (35, 35, 200), -1, cv2.LINE_AA)
    return img


# ----------------------------------------------------------------------------
# Pose engines for real video
# ----------------------------------------------------------------------------


class PoseEngine:
    IDX = dict(nose=0, ls=11, rs=12, le=13, re=14, lw=15, rw=16,
               lh=23, rh=24, lk=25, rk=26, la=27, ra=28)

    def __init__(self):
        self.model = _POSE.Pose(model_complexity=1, smooth_landmarks=True,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def process(self, bgr):
        res = self.model.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return None
        lm, (h, w) = res.pose_landmarks.landmark, bgr.shape[:2]
        core = np.mean([lm[self.IDX[k]].visibility for k in ("ls", "rs", "lh", "rh")])
        if core < 0.4:
            return None
        sk = {k: (lm[i].x * w, lm[i].y * h) for k, i in self.IDX.items()}
        legs = np.mean([lm[self.IDX[k]].visibility for k in ("lk", "rk", "la", "ra")])
        sk["full_body"] = bool(legs >= 0.4)
        return sk

    def close(self):
        self.model.close()


class MotionTracker:
    """Fallback when MediaPipe is missing: follows the largest moving object."""

    def __init__(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    def box(self, frame):
        cnts, _ = cv2.findContours(self.bg.apply(frame), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        return cv2.boundingRect(c) if cv2.contourArea(c) > 2000 * (frame.shape[1] / W) ** 2 else None


def fit(frame, width=W):
    h, w = frame.shape[:2]
    if w == width:
        return frame
    return cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------


def put_text(img, text, org, scale=0.6, color=WHITE):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def brackets(img, u):
    h, w = img.shape[:2]
    m, n, t = int(18 * u), int(30 * u), max(2, int(3 * u))
    for x, dx in ((m, 1), (w - m, -1)):
        for y, dy in ((m, 1), (h - m, -1)):
            cv2.line(img, (x, y), (x + dx * n, y), ORANGE, t)
            cv2.line(img, (x, y), (x, y + dy * n), ORANGE, t)


def dashed_v(img, x, y0, y1, color, step=10):
    for y in range(int(y0), int(y1), step * 2):
        cv2.line(img, (int(x), y), (int(x), min(y + step, int(y1))), color, 1, cv2.LINE_AA)


def draw_overlay(frame, sk, opts, mode, arm_side="r", note=None, box=None):
    u = frame.shape[0] / 540.0
    brackets(frame, u)
    if box is not None:
        x, y, bw, bh = box
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), ORANGE, 2)
        put_text(frame, "Pose model not installed: motion box only", (int(34 * u), int(48 * u)), 0.7 * u)
    if note:
        put_text(frame, note, (int(34 * u), int(48 * u)), 0.7 * u)
    if sk is None:
        if box is None and not note:
            put_text(frame, "No player in view", (int(34 * u), int(48 * u)), 0.7 * u)
        return
    if not note:
        put_text(frame, "Tracking", (int(34 * u), int(48 * u)), 0.7 * u, (0, 140, 255))
    if opts["balance"]:
        foot_y = max(sk["la"][1], sk["ra"][1])
        dashed_v(frame, sk["nose"][0], sk["nose"][1], foot_y, WHITE)
        mx = int(mid(sk["la"], sk["ra"])[0])
        cv2.line(frame, (mx, int(foot_y) - int(8 * u)), (mx, int(foot_y) + int(8 * u)), ORANGE, max(2, int(3 * u)))
    if opts["skeleton"]:
        for a, b in BONES:
            cv2.line(frame, tuple(map(int, sk[a])), tuple(map(int, sk[b])), ORANGE, max(2, int(3 * u)), cv2.LINE_AA)
        for j in JOINTS:
            p = tuple(map(int, sk[j]))
            cv2.circle(frame, p, int(6 * u), ORANGE, -1, cv2.LINE_AA)
            cv2.circle(frame, p, int(6 * u), WHITE, 1, cv2.LINE_AA)
        cv2.circle(frame, tuple(map(int, sk["nose"])), int(7 * u), WHITE, -1, cv2.LINE_AA)
    if opts["angles"]:
        trio = [("lh", "lk", "la"), ("rh", "rk", "ra")]
        if mode == "Bowling action":
            trio.append((arm_side + "s", arm_side + "e", arm_side + "w"))
        for a, b, c in trio:
            p = sk[b]
            put_text(frame, f"{angle(sk[a], p, sk[c]):.0f}", (int(p[0] + 12 * u), int(p[1] + 4 * u)), 0.6 * u)


# ----------------------------------------------------------------------------
# HTML builders
# ----------------------------------------------------------------------------


def _val(v, unit):
    return f"{v:.0f}&deg;" if unit == "deg" else f"{v:.0f}%"


def panel_html(vals, mode, note):
    spec = MODES[mode]
    score = overall_score(vals, mode) if vals else None
    parts = [f'<div class="kv-score"><div class="kv-display kv-num">{"--" if score is None else score}</div>'
             f'<div class="kv-numlab">Form score out of 100</div>'
             f'<div class="kv-bar"><span style="width:{score or 0}%"></span></div></div>']
    for m in spec:
        lo, hi = m["target"]
        s0, s1 = m["scale"]
        zl = max(0.0, (lo - s0) / (s1 - s0) * 100)
        zw = min(100 - zl, (hi - lo) / (s1 - s0) * 100)
        if vals is None:
            state, vtxt, tag, mark = "idle", "--", "Waiting", ""
        else:
            v = vals[m["key"]]
            state = "ok" if lo <= v <= hi else "off"
            vtxt, tag = _val(v, m["unit"]), ("In range" if state == "ok" else "Adjust")
            pos = min(100.0, max(0.0, (v - s0) / (s1 - s0) * 100))
            mark = f'<i class="kv-mark2" style="left:{pos:.1f}%"></i>'
        parts.append(
            f'<div class="kv-metric {state}"><div class="kv-head"><span class="kv-name">{m["label"]}</span>'
            f'<span class="kv-display kv-val">{vtxt}</span></div>'
            f'<div class="kv-track"><i class="kv-zone" style="left:{zl:.1f}%;width:{zw:.1f}%"></i>{mark}</div>'
            f'<div class="kv-foot"><span>Target {lo} to {_val(hi, m["unit"])}</span><span>{tag}</span></div></div>'
        )
    parts.append(f'<div class="kv-note">{note}</div>')
    return _h("".join(parts))


def status_html(source, mode, engine, extra=""):
    return _h(f'<div class="kv-status"><span><b>{source}</b></span><span>{mode}</span>'
              f'<span>{engine}</span>{extra}</div>')


# ----------------------------------------------------------------------------
# Header and sidebar
# ----------------------------------------------------------------------------

st.markdown(_h("""
<div class="kv-brand"><div class="kv-mark"></div><div>
<div class="kv-display kv-title">Codezilla Kinetic Vision</div>
<div class="kv-sub">Measure a cricketer's joint angles from video and get coaching feedback in real time.</div>
</div></div>
"""), unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="kv-display" style="font-size:1.6rem;font-weight:600;">Setup</div>', unsafe_allow_html=True)
    mode = st.selectbox("Technique", list(MODES.keys()))
    source = st.radio(
        "Video source", ["Simulation", "Upload a clip", "Webcam"],
        help="Simulation is an animated figure and needs no camera or video file.",
    )
    upload = None
    if source == "Upload a clip":
        upload = st.file_uploader("Cricket clip", type=["mp4", "mov", "avi"],
                                  help="A side-on view of one player works best.")
    st.divider()
    st.markdown('<div class="kv-display" style="font-size:1.3rem;font-weight:600;">Overlay</div>', unsafe_allow_html=True)
    opts = {
        "skeleton": st.toggle("Skeleton", True),
        "angles": st.toggle("Angle readouts", True),
        "balance": st.toggle("Balance line", True, help="Dashed line from the head straight down to the feet."),
        "loop": st.toggle("Loop uploaded clip", True),
        "mirror": st.toggle("Mirror webcam", True),
    }
    st.divider()
    if POSE_OK:
        st.caption("Pose model ready. Uploads and webcam use full joint tracking.")
    else:
        st.caption("Pose model not installed. Uploads and webcam show motion tracking only. "
                   "Simulation is not affected.")
        if POSE_ERR:
            st.code(POSE_ERR, language=None)

# ----------------------------------------------------------------------------
# Analysis loop
# ----------------------------------------------------------------------------

IDLE_NOTE = "Press Start analysis to begin live measurements."


def engine_name(src):
    if src == "Simulation":
        return "Built-in motion model"
    return "MediaPipe Pose" if POSE_OK else "Motion tracking"


def run_analysis(view, panel, status):
    spec = MODES[mode]
    tracker, ema = ArmTracker(), {}
    scores, misses, n = [], {m["key"]: 0 for m in spec}, 0
    cap = engine = motion = tmp = None

    try:
        if source != "Simulation":
            if source == "Upload a clip":
                if upload is None:
                    st.error("No clip selected. Upload a clip in the sidebar, or switch the source to Simulation.")
                    return
                suffix = os.path.splitext(upload.name)[1] or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                    f.write(upload.getvalue())
                    tmp = f.name
                cap = cv2.VideoCapture(tmp)
            else:
                cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open the video. For a webcam, close other apps that use the camera "
                         "and allow camera access. Otherwise, switch the source to Simulation.")
                return
            engine, motion = (PoseEngine(), None) if POSE_OK else (None, MotionTracker())

        fps = 24.0 if cap is None else min(30.0, max(10.0, cap.get(cv2.CAP_PROP_FPS) or 24.0))
        t, rewound = 0.0, False

        while True:
            started = time.perf_counter()
            box = None
            if cap is None:
                sk = synth_pose(mode, t)
                frame = synth_scene(sk, mode)
                t += 1.0 / fps
            else:
                ok, frame = cap.read()
                if not ok:
                    if source == "Upload a clip" and opts["loop"] and not rewound:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        rewound = True
                        continue
                    break
                rewound = False
                if source == "Webcam" and opts["mirror"]:
                    frame = cv2.flip(frame, 1)
                frame = fit(frame)
                sk = engine.process(frame) if engine else None
                box = motion.box(frame) if motion else None

            vals = None
            legs_missing = sk is not None and not sk.get("full_body", True)
            if sk is not None and not legs_missing:
                raw = measure(sk, mode, tracker)
                for k, v in raw.items():
                    ema[k] = v if k not in ema else ema[k] + 0.35 * (v - ema[k])
                vals = dict(ema)
                scores.append(overall_score(vals, mode))
                n += 1
                for m in spec:
                    if not m["target"][0] <= vals[m["key"]] <= m["target"][1]:
                        misses[m["key"]] += 1

            draw_overlay(frame, sk, opts, mode, tracker.side, box=box)
            view.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB",
                       output_format="JPEG", use_container_width=True)

            if n % 3 == 0 or vals is None:  # refresh the side panel every third frame
                if vals is not None:
                    note = coaching(vals, mode)
                elif legs_missing:
                    note = "Step back or tilt the camera down so your knees and feet are in frame."
                elif POSE_OK or source == "Simulation":
                    note = "No player detected. Step into the frame so the full body is visible."
                else:
                    note = "Install the pose model to measure joint angles. See the README."
                panel.markdown(panel_html(vals, mode, note), unsafe_allow_html=True)
                if n:
                    worst = max(misses, key=misses.get)
                    label = next(m["label"] for m in spec if m["key"] == worst)
                    extra = (f'<span>Frames analysed <b>{n}</b></span>'
                             f'<span>Average score <b>{sum(scores) / len(scores):.0f}</b></span>'
                             + (f'<span>Most frequent issue <b>{label}</b></span>' if misses[worst] else ""))
                    status.markdown(status_html(source, mode, engine_name(source), extra), unsafe_allow_html=True)

            time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - started)))
    finally:
        if cap is not None:
            cap.release()
        if engine is not None:
            engine.close()
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------

tab_live, tab_how, tab_careers = st.tabs(["Live analysis", "How it works", "Careers"])

with tab_live:
    left, right = st.columns([5, 2], gap="large")
    with left:
        status = st.empty()
        view = st.empty()
        b_start, b_stop, _ = st.columns([1, 1, 2])
        start = b_start.button("Start analysis", type="primary", key="start")
        b_stop.button("Stop", key="stop")
    with right:
        panel = st.empty()

    status.markdown(status_html(source, mode, engine_name(source)), unsafe_allow_html=True)
    idle_sk = synth_pose(mode, 1.2)
    idle = synth_scene(idle_sk, mode)
    draw_overlay(idle, idle_sk, opts, mode, note="Ready")
    view.image(cv2.cvtColor(idle, cv2.COLOR_BGR2RGB), channels="RGB", output_format="JPEG", use_container_width=True)
    panel.markdown(panel_html(None, mode, IDLE_NOTE), unsafe_allow_html=True)

    if start:
        run_analysis(view, panel, status)

with tab_how:
    st.markdown(_h("""
    <div style="margin-top:14px;">
    <div class="kv-step"><div class="kv-display kv-stepnum">1</div><div>
      <div class="kv-display kv-steptitle">Capture frames</div>
      <div class="kv-steptext">A video is a stack of still images, played about 24 to 30 times a second. The app reads one frame at a time.</div></div></div>
    <div class="kv-step"><div class="kv-display kv-stepnum">2</div><div>
      <div class="kv-display kv-steptitle">Find the joints</div>
      <div class="kv-steptext">A neural network trained on many thousands of labelled images locates the shoulders, elbows, wrists, hips, knees and ankles in each frame as x and y positions.</div></div></div>
    <div class="kv-step"><div class="kv-display kv-stepnum">3</div><div>
      <div class="kv-display kv-steptitle">Measure the angles</div>
      <div class="kv-steptext">The bend at a joint is the angle between two bones. This is school trigonometry: the dot product of the two bone vectors, divided by their lengths, gives the cosine of the angle.</div></div></div>
    <div class="kv-step"><div class="kv-display kv-stepnum">4</div><div>
      <div class="kv-display kv-steptitle">Compare with a target and coach</div>
      <div class="kv-steptext">Each measurement has a target range. The form score shows how many are in range, and the coaching note names the one furthest out. These ranges are rules written by people, so coaches and sports scientists shape what the software says.</div></div></div>
    </div>
    """), unsafe_allow_html=True)
    st.latex(r"\theta = \arccos\left(\frac{\vec{BA}\cdot\vec{BC}}{|\vec{BA}|\,|\vec{BC}|}\right)")
    with st.expander("Talking points for the presenter"):
        st.markdown(
            "- Start on Simulation to show the whole interface in thirty seconds, then bring a volunteer up to the Webcam.\n"
            "- Ask the room which joint angle matters most for a batter, then switch the technique and check it live.\n"
            "- The model only sees pixels. The meaning of a good stance comes from people who know the sport.\n"
            "- Everything here runs on one laptop, written in Python, with no cloud service."
        )

with tab_careers:
    roles = [
        ("Computer vision engineer", "Builds the models that find people and joints in video, like the one in this demo.",
         "Python, NumPy, OpenCV, linear algebra"),
        ("Sports scientist", "Measures how athletes move to improve technique and lower injury risk.",
         "Physics, anatomy, statistics"),
        ("Performance analyst", "Turns match footage and tracking data into decisions for coaches and selectors.",
         "Spreadsheets, SQL, data visualisation"),
        ("Physiotherapist", "Checks rehabilitation exercises and tracks recovery with movement measurements.",
         "Biology, anatomy, communication"),
        ("Game and animation developer", "Uses motion capture so digital characters move like real people.",
         "Python or C#, geometry, game engines"),
        ("Robotics engineer", "Teaches machines to sense and copy human movement.",
         "Programming, trigonometry, electronics"),
    ]
    rows = "".join(
        f'<div class="kv-row"><div class="kv-display kv-role">{r}</div><div>{d}</div><div>{s}</div></div>'
        for r, d, s in roles
    )
    st.markdown(_h(
        '<div style="margin-top:14px;"><div class="kv-rowh"><div>Role</div><div>How it connects to this demo</div>'
        f'<div>Start with</div></div>{rows}</div>'
    ), unsafe_allow_html=True)
    st.markdown("")
    st.markdown("This whole app is a few hundred lines of Python, and the angle formula is school maths. "
                "You can start building things like it with a laptop and free tutorials.")

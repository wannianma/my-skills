#!/usr/bin/env python3
# 参考示例：只针对某一段具体录屏。SEGS、KEYS、TAPS、NARR、CAPS 都是按该视频实测的，换一段必须重新测量。
# 运行前，工作目录里要有 edge-tts 生成的讲解 WAV，以及 lines.json。
"""Build polished product demo from /workspace/videos/demo-recording.mp4.
Python renders zoom/crossfade/ripple/captions frame-by-frame (cv2 warpAffine, sub-pixel smooth),
audio is assembled in numpy (segment crossfades, narration ducking), then muxed with ffmpeg."""
import json, subprocess, numpy as np, cv2, math, sys
from PIL import Image, ImageDraw, ImageFont

SRC = '/workspace/videos/demo-recording.mp4'
OUT = '/workspace/videos/demo-edited.mp4'
W, H, FPS, SR = 870, 1392, 60, 48000
XF = 0.35                     # crossfade at cuts (s)
WD = '/workspace/demo-edit'

# ---------------- edit decision list (source seconds) ----------------
SEGS = [(0.0, 31.7),          # intro + vocab card + some keynote
        (36.6, 45.5),         # "It was really different ... It was a revolution."
        (50.65, 70.5),        # "For 2010..." + 复读 on, sentence loops
        (74.3, 114.5)]        # loop end, 复读 off, 跟读, 挖空, to "an all new design."
HOLD = 1.4                    # freeze last frame + fade out
starts = []; t = 0.0
for i, (a, b) in enumerate(SEGS):
    starts.append(t); t += (b - a) - XF
BODY_END = starts[-1] + (SEGS[-1][1] - SEGS[-1][0])
TOTAL = BODY_END + HOLD
def s2o(src):
    for (a, b), o in zip(SEGS, starts):
        if a <= src <= b: return o + src - a
    raise ValueError(src)

# ---------------- zoom windows (x0, y0, w) ; h = w*1.6 ----------------
FULL   = (0, 0, 870)
INTRO  = (175, 92, 520)
CARD   = (175, 555, 520)
REP    = (175, 540, 520)
FOLLOW = (150, 98, 570)
CLOZE1 = (175, 100, 520)
CLOZE2 = (175, 495, 520)
END    = (160, 125, 550)
# (time, window): eased transitions between consecutive keys
KEYS = [(0, FULL), (1.7, FULL), (2.25, INTRO), (12.25, INTRO), (12.85, CARD), (18.6, CARD), (19.2, FULL),
        (25.5, FULL), (26.1, INTRO), (38.2, INTRO), (38.8, FULL),
        (48.0, FULL), (48.6, REP), (62.0, REP), (62.6, FULL),
        (67.05, FULL), (67.6, FOLLOW), (81.2, FOLLOW), (81.7, CLOZE1), (84.6, CLOZE1), (85.4, CLOZE2),
        (89.6, CLOZE2), (90.3, FULL), (94.6, FULL), (95.3, END), (TOTAL + 1, END)]
def ease(u): return 4*u**3 if u < .5 else 1 - (-2*u + 2)**3/2
def window_at(t):
    for (t0, w0), (t1, w1) in zip(KEYS, KEYS[1:]):
        if t0 <= t <= t1:
            u = ease((t - t0)/(t1 - t0)) if t1 > t0 else 1
            # interpolate in log-scale for width (perceptually even zoom), linear for centre
            c0 = (w0[0] + w0[2]/2, w0[1] + w0[2]*0.8); c1 = (w1[0] + w1[2]/2, w1[1] + w1[2]*0.8)
            ww = math.exp(math.log(w0[2])*(1-u) + math.log(w1[2])*u)
            cx = c0[0] + (c1[0]-c0[0])*u; cy = c0[1] + (c1[1]-c0[1])*u
            return cx - ww/2, cy - ww*0.8, ww
    return FULL

# ---------------- taps (source coords, output time) ----------------
TAPS = [((385, 648), s2o(12.75)),     # word 'reinvented'
        ((537, 1298), s2o(59.4)),     # 复读 on
        ((537, 1298), s2o(75.8)),     # 复读 off
        ((364, 145), s2o(82.6)),      # 跟读 tab
        ((488, 145), s2o(96.8))]      # 挖空 tab

# ---------------- narration (output time) ----------------
NARR = [('n1', 2.2), ('n2', 13.2), ('n3', s2o(59.6)), ('n4', s2o(82.8)), ('n5', s2o(97.1)), ('n6', s2o(106.65))]
LINES = dict(json.load(open(f'{WD}/lines.json')))
# ---------------- captions: (text, t_in, t_out, y_centre in output px) ----------------
CAPS = [('双语字幕 · 逐词高亮', 2.4, 8.0, 1290),
        ('词汇卡片 · 点词即查', 13.0, 18.4, 520),
        ('复读 · 单句循环', s2o(59.7), s2o(59.7)+5.5, 1040),
        ('跟读 · 先听原音，再开口', s2o(82.8), s2o(82.8)+5.5, 540),
        ('挖空 · 遮住关键词，靠耳朵听', s2o(97.0), 85.2, 540),
        ('看演讲，学英语', s2o(106.6), BODY_END + HOLD - 0.2, 1320)]
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 40, index=2)  # SC
def make_cap(text):
    tw = FONT.getbbox(text); w = tw[2] - tw[0] + 64; h = 76
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w-1, h-1), radius=h//2, fill=(20, 24, 38, 205))
    d.ellipse((22, h//2-6, 34, h//2+6), fill=(64, 140, 255, 255))
    d.text((44 - tw[0], (h - (tw[3]-tw[1]))//2 - tw[1]), text, font=FONT, fill=(255, 255, 255, 255))
    a = np.asarray(im).astype(np.float32)
    return a[..., :3][..., ::-1].copy(), a[..., 3:4]/255.0   # BGR, alpha
CAPIMG = [make_cap(c[0]) for c in CAPS]

def fix_artifact(f):   # cover static pink extension bubble at right edge
    f[414:468, 822:870] = f[356:410, 822:870]
    return f

def draw_ripples(f, t):
    for (x, y), tt in TAPS:
        u = (t - tt)/0.7
        if 0 <= u <= 1:
            ov = f.copy()
            r = int(14 + 46*ease(u)); a = (1 - u)*0.55
            cv2.circle(ov, (x, y), r, (255, 170, 60), -1, cv2.LINE_AA)
            cv2.addWeighted(ov, a*0.5, f, 1 - a*0.5, 0, f)
            cv2.circle(f, (x, y), r, (255, 140, 30), 3, cv2.LINE_AA) if u < .85 else None
    return f

class Reader:
    def __init__(self, a, b):
        self.a = a; self.idx = round(a*FPS) - 1; self.cur = None
        self.p = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', f'{a:.4f}', '-i', SRC, '-t', f'{b-a+0.5:.3f}',
                                   '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE)
        self.first = round(a*FPS)
    def get(self, src_t):
        want = max(round(src_t*FPS), self.first)
        while self.idx < want:
            buf = self.p.stdout.read(W*H*3)
            if len(buf) < W*H*3: break
            self.cur = np.frombuffer(buf, np.uint8).reshape(H, W, 3); self.idx += 1
        return self.cur

def render():
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS),
                            '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p',
                            '-profile:v', 'high', f'{WD}/video_only.mp4'], stdin=subprocess.PIPE)
    readers = {}
    n = int(round(TOTAL*FPS)); last = None
    for i in range(n):
        t = i/FPS
        layers = []
        for k, ((a, b), o) in enumerate(zip(SEGS, starts)):
            st = a + (t - o)
            if a <= st <= b + 1e-6:
                if k not in readers: readers[k] = Reader(a, b)
                fr = readers[k].get(st)
                # weight: fade in over first XF (except seg 0)
                w = 1.0 if k == 0 else min(1.0, (t - o)/XF)
                layers.append((fr, w))
        if layers:
            if len(layers) == 1: f = layers[0][0].astype(np.float32)
            else:
                (f0, _), (f1, w1) = layers[0], layers[1]
                f = f0.astype(np.float32)*(1 - w1) + f1.astype(np.float32)*w1
            f = np.clip(f, 0, 255).astype(np.uint8); last = f.copy()
        else:
            f = last.copy()
        f = fix_artifact(f)
        f = draw_ripples(f, t)
        x0, y0, ww = window_at(t)
        s = W/ww
        M = np.float32([[s, 0, -x0*s], [0, s, -y0*s]])
        out = cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_CUBIC if s > 1.01 else cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        for (text, ti, to, yc), (rgb, al) in zip(CAPS, CAPIMG):
            if ti <= t <= to:
                a = min(1, (t-ti)/0.3, (to-t)/0.3)
                dy = int(12*(1 - min(1, (t-ti)/0.3)))
                h, w = rgb.shape[:2]; x = (W - w)//2; y = yc - h//2 + dy
                reg = out[y:y+h, x:x+w]
                out[y:y+h, x:x+w] = reg*(1 - al*a) + rgb*al*a
        if t > TOTAL - 1.0: out *= max(0, (TOTAL - t)/1.0)
        if t < 0.4: out *= t/0.4 * 0.5 + 0.5
        enc.stdin.write(np.clip(out, 0, 255).astype(np.uint8).tobytes())
        if i % 600 == 0: print(f'frame {i}/{n}', flush=True)
    enc.stdin.close(); enc.wait()

def load_audio(path, ch, af='anull'):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-af', af, '-f', 'f32le', '-ac', str(ch), '-ar', str(SR), '-'],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, ch).copy()

def build_audio():
    src = load_audio(SRC, 2, 'dynaudnorm=f=250:g=9:p=0.9:m=10:s=0')  # even out uneven keynote levels
    N = int(round(TOTAL*SR)); mix = np.zeros((N, 2), np.float32)
    for k, ((a, b), o) in enumerate(zip(SEGS, starts)):
        seg = src[int(a*SR):int(b*SR)].copy(); L = len(seg); xf = int(XF*SR)
        if k > 0: seg[:xf] *= np.sin(np.linspace(0, np.pi/2, xf))[:, None]
        if k < len(SEGS)-1: seg[-xf:] *= np.cos(np.linspace(0, np.pi/2, xf))[:, None]
        else:  # fade the tail into the hold
            fl = int(1.2*SR); seg[-fl:] *= np.linspace(1, 0.0, fl)[:, None]**1.5
        s0 = int(round(o*SR)); mix[s0:s0+L] += seg[:N-s0]
    # narration + ducking envelope
    narr = np.zeros(N, np.float32); duck = np.ones(N, np.float32)
    for key, t0 in NARR:
        v = load_audio(f'{WD}/tts/{key}.wav', 1, 'highpass=f=80,acompressor=threshold=0.08:ratio=3:attack=4:release=90:makeup=1')[:, 0]
        v = v / (np.sqrt(np.mean(v**2)) + 1e-9) * 0.11        # ~ -19 dBFS RMS
        s0 = int(t0*SR); narr[s0:s0+len(v)] += v[:N-s0]
        a = max(0, s0 - int(.35*SR)); b = min(N, s0 + len(v) + int(.45*SR))
        duck[a:b] = np.minimum(duck[a:b], 0.25)
    k = int(0.3*SR); ker = np.ones(k)/k                       # smooth ramps (~0.3 s)
    duck = np.convolve(np.convolve(duck, ker, 'same'), ker, 'same').astype(np.float32)
    # bring keynote speech to a consistent level
    rms = np.sqrt(np.mean(mix[np.abs(mix).max(1) > 0.01]**2)); mix *= 0.10/rms
    out = mix*duck[:, None] + narr[:, None]*1.0
    print('peak before limiter', float(np.abs(out).max()))
    # brick-wall-ish limiter (ffmpeg alimiter) instead of global gain, so narration level is kept
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', '-',
                    '-af', 'alimiter=limit=0.89:attack=3:release=60:level=disabled', '-c:a', 'pcm_s16le',
                    f'{WD}/audio_mix.wav'], input=out.astype('<f4').tobytes(), check=True)
    np.save(f'{WD}/duck.npy', duck[::480])

if __name__ == '__main__':
    print('segments out starts', [round(s, 2) for s in starts], 'TOTAL', round(TOTAL, 2))
    print('narr', [(k, round(t, 2)) for k, t in NARR]); print('taps', [round(t, 2) for _, t in TAPS])
    if 'audio' in sys.argv or len(sys.argv) == 1: build_audio()
    if 'video' in sys.argv or len(sys.argv) == 1: render()
    if 'mux' in sys.argv or len(sys.argv) == 1:
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', f'{WD}/video_only.mp4', '-i', f'{WD}/audio_mix.wav',
                        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '160k', '-shortest', '-movflags', '+faststart', OUT], check=True)
        print('done', OUT)

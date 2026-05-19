"""
로컬 서버 — 큐 기반 팬아웃 아키텍처

POST /upload 에서 수신한 프레임을 2개의 독립 큐로 팬아웃:
  gui_queue  (maxsize=1,   drop_oldest=True)  → MJPEG 스트림 (항상 최신)
  db_queue   (maxsize=100, drop_oldest=False) → 디스크 저장

각 큐는 전용 워커 스레드만 소비 → 메모리 충돌·병목 방지
"""

from __future__ import annotations

import json
import queue
import time
import logging
import threading
import numpy as np
import cv2
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, Response, render_template_string, send_from_directory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── 설정 ──────────────────────────────────────────────────────────────────────
HOST        = "0.0.0.0"
PORT        = 5000
SAVE_DIR    = Path("received_images")
MAX_CONTENT = 10 * 1024 * 1024
# ─────────────────────────────────────────────────────────────────────────────


# ── 프레임 데이터 클래스 ──────────────────────────────────────────────────────
@dataclass
class Frame:
    data:      bytes                          # 원본 JPEG 바이트
    filename:  str                            # 저장 파일명
    timestamp: str                            # ISO 타임스탬프
    shape:     dict                           # {"h": H, "w": W, "channels": C}
    meta:      dict = field(default_factory=dict)   # 트리거 메타 (label 등)


# ── FrameQueue ────────────────────────────────────────────────────────────────
class FrameQueue:
    """
    크기 제한이 있는 스레드 안전 프레임 큐.

    drop_oldest=True  : 큐가 가득 차면 오래된 프레임을 버리고 새 프레임 삽입 (GUI)
    drop_oldest=False : 큐가 가득 차면 새 프레임을 버림 (AI/DB — 처리 속도에 따라감)

    통계(dropped, processed)는 모니터링 전용이며
    CPython GIL이 단순 정수 증가를 사실상 원자적으로 보호함.
    """

    def __init__(self, name: str, maxsize: int, drop_oldest: bool = False):
        self.name         = name
        self._q           = queue.Queue(maxsize=maxsize)
        self._drop_oldest = drop_oldest
        self.dropped      = 0
        self.processed    = 0

    def put(self, frame: Frame) -> bool:
        """Non-blocking put. 큐에 넣었으면 True, 드롭했으면 False."""
        if self._drop_oldest:
            while self._q.full():
                try:
                    self._q.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    break
            self._q.put_nowait(frame)
            return True
        try:
            self._q.put_nowait(frame)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def get(self, timeout: float = 1.0) -> Frame | None:
        try:
            frame = self._q.get(timeout=timeout)
            self.processed += 1
            return frame
        except queue.Empty:
            return None

    @property
    def qsize(self) -> int:
        return self._q.qsize()


# ── 큐 인스턴스 ───────────────────────────────────────────────────────────────
gui_queue = FrameQueue("gui", maxsize=1,   drop_oldest=True)   # 최신 프레임만 유지
db_queue  = FrameQueue("db",  maxsize=100, drop_oldest=False)  # 저장 버퍼

_ALL_QUEUES: list[FrameQueue] = [gui_queue, db_queue]


# ── 팬아웃 디스패처 ───────────────────────────────────────────────────────────
def dispatch(frame: Frame) -> None:
    """업로드된 프레임을 모든 소비자 큐에 동시 전달. 각 큐의 드롭 정책은 독립적."""
    for q in _ALL_QUEUES:
        if not q.put(frame):
            log.debug("[%s] 큐 포화 — 프레임 드롭 (누적: %d)", q.name, q.dropped)


# ── 워커 스레드 ───────────────────────────────────────────────────────────────

# GUI 워커 ──────────────────────────────────────────────────────────────────
_latest_frame: bytes = b""
_latest_lock  = threading.Lock()

def _gui_worker() -> None:
    """gui_queue 에서 프레임을 꺼내 MJPEG 스트리밍용 최신 프레임을 갱신."""
    global _latest_frame
    while True:
        frame = gui_queue.get()
        if frame is None:
            break
        with _latest_lock:
            _latest_frame = frame.data


# 인메모리 인덱스 — db_worker가 쓰고, /images/<filename>/info 가 읽음
_image_index: dict[str, dict] = {}
_image_index_lock = threading.Lock()


# DB 워커 ───────────────────────────────────────────────────────────────────
def _db_worker() -> None:
    """db_queue 에서 프레임을 꺼내 이미지 파일 저장 후 index.jsonl 에 경로 기록."""
    while True:
        frame = db_queue.get()
        if frame is None:
            break
        img_path   = SAVE_DIR / frame.filename
        index_path = SAVE_DIR / "index.jsonl"
        try:
            img_path.write_bytes(frame.data)
            record = {
                "path":      str(img_path),
                "timestamp": frame.timestamp,
                "shape":     frame.shape,
                "meta":      frame.meta,
            }
            with index_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            with _image_index_lock:
                _image_index[frame.filename] = record
            log.info("[db] 저장: %s  %s", frame.filename, frame.shape)
        except OSError as exc:
            log.error("[db] 저장 실패: %s", exc)


# ── 워커 시작 ─────────────────────────────────────────────────────────────────
def start_workers() -> list[threading.Thread]:
    """두 워커 스레드를 데몬으로 시작. 메인 프로세스 종료 시 자동 정리됨."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    workers = [
        threading.Thread(target=_gui_worker, name="gui-worker", daemon=True),
        threading.Thread(target=_db_worker,  name="db-worker",  daemon=True),
    ]
    for t in workers:
        t.start()
    log.info("워커 스레드 시작: %s", [t.name for t in workers])
    return workers


# ── 트리거 상태 ───────────────────────────────────────────────────────────────
_trigger_event = threading.Event()
_trigger_lock  = threading.Lock()
_trigger_meta: dict = {}


# ── 응답 빌더 ─────────────────────────────────────────────────────────────────
def build_response(status: str, **kwargs) -> dict:
    """
    공통 JSON 응답 팩토리. status / timestamp 는 항상 포함.
    **kwargs 로 필드를 자유롭게 추가 가능.
    """
    return {
        "status":    status,
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        **kwargs,
    }


# ── 이미지 검증 ───────────────────────────────────────────────────────────────
def validate_image(data: bytes) -> tuple[bool, dict]:
    """cv2.imdecode() 로 실제 디코딩해 유효성 검사. 성공 시 shape 반환."""
    nparr = np.frombuffer(data, dtype=np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return False, {"error": "cv2.imdecode() 실패 — 손상되거나 지원하지 않는 이미지 형식"}
    h, w, c = img.shape
    return True, {"h": h, "w": w, "channels": c}


# ── Flask 앱 ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT

# ── HTML 뷰어 ─────────────────────────────────────────────────────────────────
_VIEWER_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>JetRover 라이브 뷰</title>
  <style>
    body  { background:#111; color:#eee; font-family:sans-serif;
            display:flex; flex-direction:column; align-items:center; padding:20px; }
    img   { max-width:100%; border:2px solid #444; border-radius:6px; }
    h1    { margin-bottom:12px; }
    .bar  { margin-top:14px; display:flex; gap:10px; align-items:center; }
    input { padding:6px 10px; border-radius:4px; border:1px solid #555;
            background:#222; color:#eee; width:200px; }
    button{ padding:8px 18px; border-radius:4px; border:none;
            background:#2563eb; color:#fff; cursor:pointer; font-size:1em; }
    button:hover { background:#1d4ed8; }
    #msg  { margin-top:8px; font-size:0.85em; color:#6ee7b7; min-height:1.2em; }
    #stats{ margin-top:6px; font-size:0.8em; color:#888; }
  </style>
</head>
<body>
  <h1>JetRover 라이브 스트림</h1>
  <img id="feed" src="/stream" alt="스트림 대기 중...">
  <div class="bar">
    <input id="label" placeholder="촬영 레이블 (선택)" />
    <button onclick="sendTrigger()">촬영 신호 전송</button>
  </div>
  <div id="msg"></div>
  <div id="stats">서버: {{ host }}:{{ port }} | 저장: {{ save_dir }} | <a href="/gallery" style="color:#60a5fa">갤러리</a></div>
  <script>
    function sendTrigger() {
      const label = document.getElementById('label').value.trim();
      const body  = label ? { label } : {};
      fetch('/trigger', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      .then(r => r.json())
      .then(d => {
        document.getElementById('msg').textContent =
          d.message + (d.label ? ` [${d.label}]` : '') + '  ' + d.timestamp;
      })
      .catch(e => document.getElementById('msg').textContent = '오류: ' + e);
    }
  </script>
</body>
</html>"""


# ── 갤러리 HTML ───────────────────────────────────────────────────────────────
_GALLERY_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>JetRover 갤러리</title>
  <style>
    body      { background:#111; color:#eee; font-family:sans-serif; padding:20px; }
    h1        { text-align:center; margin-bottom:6px; }
    .bar      { display:flex; justify-content:center; gap:12px; align-items:center; margin-bottom:18px; }
    button    { padding:5px 12px; border-radius:4px; border:none; background:#2563eb; color:#fff; cursor:pointer; font-size:.82em; }
    button:hover  { background:#1d4ed8; }
    button.json-btn { background:#374151; }
    button.json-btn:hover { background:#4b5563; }
    label     { font-size:0.9em; cursor:pointer; }
    #count    { font-size:0.85em; color:#888; }
    .grid     { display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:14px; }
    .card     { background:#1e1e1e; border-radius:8px; overflow:hidden; border:1px solid #333; }
    .card a img { width:100%; display:block; aspect-ratio:4/3; object-fit:cover; transition:opacity .15s; }
    .card a img:hover { opacity:.85; }
    .info     { padding:8px 10px; font-size:0.78em; }
    .lbl      { color:#6ee7b7; font-weight:bold; margin-bottom:3px; }
    .ts       { color:#888; }
    .shape    { color:#555; margin-top:2px; }
    .actions  { margin-top:6px; display:flex; gap:6px; }
    .json-box { display:none; margin:0; padding:8px; background:#0d1117;
                border-top:1px solid #333; font-size:.72em; color:#a5f3fc;
                overflow-x:auto; white-space:pre; }
    .empty    { text-align:center; color:#555; margin-top:80px; font-size:1.1em; }
  </style>
</head>
<body>
  <h1>갤러리</h1>
  <div class="bar">
    <a href="/" style="color:#60a5fa;font-size:.9em">← 라이브 뷰</a>
    <button onclick="load()">새로고침</button>
    <label><input type="checkbox" id="auto" onchange="toggleAuto()"> 자동 새로고침 (5초)</label>
    <span id="count"></span>
  </div>
  <div class="grid" id="grid"></div>

  <script>
    let timer = null;

    function load() {
      fetch('/images')
        .then(r => r.json())
        .then(d => {
          const records = [...d.records].reverse();
          document.getElementById('count').textContent = '총 ' + d.total + '장';
          const grid = document.getElementById('grid');
          if (!records.length) {
            grid.innerHTML = '<div class="empty">저장된 이미지가 없습니다</div>';
            return;
          }
          grid.innerHTML = records.map(r => {
            const fname = r.filename;
            const sid   = fname.replace('.', '-');   // ID에 점 제거
            const label = r.meta && r.meta.label ? r.meta.label : '';
            const shape = r.shape ? r.shape.w + '×' + r.shape.h : '';
            return '<div class="card">'
              + '<a href="/images/' + fname + '" target="_blank">'
              + '<img src="/images/' + fname + '" loading="lazy" alt="' + fname + '">'
              + '</a>'
              + '<div class="info">'
              + (label ? '<div class="lbl">' + label + '</div>' : '')
              + '<div class="ts">'    + (r.timestamp || '') + '</div>'
              + '<div class="shape">' + shape + '</div>'
              + '<div class="actions">'
              + '<button class="json-btn" onclick="toggleJson(\'' + fname + '\',\'' + sid + '\')">JSON 보기</button>'
              + '</div>'
              + '</div>'
              + '<pre class="json-box" id="jb-' + sid + '"></pre>'
              + '</div>';
          }).join('');
        })
        .catch(e => console.error(e));
    }

    function toggleJson(fname, sid) {
      const box = document.getElementById('jb-' + sid);
      if (box.style.display === 'block') {
        box.style.display = 'none';
        return;
      }
      if (box.textContent) {
        box.style.display = 'block';
        return;
      }
      fetch('/images/' + fname + '/info')
        .then(r => r.json())
        .then(d => {
          box.textContent = JSON.stringify(d.record, null, 2);
          box.style.display = 'block';
        })
        .catch(e => { box.textContent = '오류: ' + e; box.style.display = 'block'; });
    }

    function toggleAuto() {
      if (document.getElementById('auto').checked) {
        timer = setInterval(load, 5000);
      } else {
        clearInterval(timer);
        timer = null;
      }
    }

    load();
  </script>
</body>
</html>"""


# ── 엔드포인트 ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(
        _VIEWER_HTML,
        host=request.host.split(":")[0],
        port=PORT,
        save_dir=str(SAVE_DIR.resolve()),
    )


@app.route("/upload", methods=["POST"])
def upload():
    """이미지를 수신 → 검증 → Frame 생성 → 모든 큐로 팬아웃."""
    if "image" not in request.files:
        return jsonify(build_response("error", error="image 필드가 없습니다")), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify(build_response("error", error="파일명이 비어 있습니다")), 400

    data = file.read()
    if not data:
        return jsonify(build_response("error", error="빈 파일")), 400

    valid, img_meta = validate_image(data)
    if not valid:
        log.warning("이미지 검증 실패: %s", img_meta["error"])
        return jsonify(build_response("error", **img_meta)), 422

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    with _trigger_lock:
        meta = dict(_trigger_meta)

    frame = Frame(
        data=data,
        filename=f"{ts}.jpg",
        timestamp=datetime.now().isoformat(timespec="milliseconds"),
        shape=img_meta,
        meta=meta,
    )
    dispatch(frame)

    return jsonify(
        build_response("ok",
            file=frame.filename,
            bytes=len(data),
            image=img_meta,
            queued={q.name: q.qsize for q in _ALL_QUEUES},  # 각 큐 현재 대기 수
        )
    ), 200


@app.route("/trigger", methods=["POST"])
def set_trigger():
    body = request.get_json(silent=True) or {}
    with _trigger_lock:
        _trigger_meta.clear()
        _trigger_meta.update(body)
    _trigger_event.set()
    label = body.get("label", "")
    log.info("촬영 트리거 설정 — label=%r", label)
    return jsonify(
        build_response("ok",
            message="촬영 신호가 JetRover로 전달됩니다",
            **({} if not label else {"label": label}),
        )
    ), 200


@app.route("/trigger", methods=["GET"])
def get_trigger():
    fired = _trigger_event.is_set()
    if fired:
        _trigger_event.clear()
        with _trigger_lock:
            meta = dict(_trigger_meta)
            _trigger_meta.clear()
        return jsonify(build_response("ok", triggered=True,  **meta)), 200
    return jsonify(build_response("ok",     triggered=False)), 200


def _mjpeg_generator():
    boundary = b"--frame\r\n"
    while True:
        with _latest_lock:
            frame = _latest_frame
        if frame:
            yield (
                boundary
                + b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(frame)}\r\n\r\n".encode()
                + frame
                + b"\r\n"
            )
        time.sleep(0.05)


@app.route("/stream")
def stream():
    return Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/gallery")
def gallery():
    """저장된 이미지를 격자로 보여주는 HTML 갤러리."""
    return render_template_string(_GALLERY_HTML)


@app.route("/images")
def list_images():
    """index.jsonl 의 모든 레코드를 JSON 배열로 반환. filename 필드를 함께 제공."""
    index_path = SAVE_DIR / "index.jsonl"
    if not index_path.exists():
        return jsonify(build_response("ok", records=[], total=0)), 200
    records = []
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rec["filename"] = Path(rec["path"]).name
                records.append(rec)
            except (json.JSONDecodeError, KeyError):
                pass
    return jsonify(build_response("ok", records=records, total=len(records))), 200


@app.route("/images/<filename>/info")
def image_info(filename: str):
    """촬영 직후 인메모리 인덱스에서 JSON 레코드를 즉시 반환."""
    with _image_index_lock:
        record = _image_index.get(filename)
    if record is None:
        return jsonify(build_response("error", error="레코드 없음")), 404
    return jsonify(build_response("ok", record=record)), 200


@app.route("/images/<filename>")
def serve_image(filename: str):
    """저장된 이미지 파일을 직접 제공."""
    return send_from_directory(SAVE_DIR.resolve(), filename)


@app.route("/status")
def status():
    """서버 상태 및 큐별 통계 반환."""
    with _latest_lock:
        has_frame = bool(_latest_frame)
    return jsonify(
        build_response("ok",
            streaming=has_frame,
            trigger_pending=_trigger_event.is_set(),
            save_dir=str(SAVE_DIR),
            queues={
                q.name: {
                    "qsize":     q.qsize,
                    "dropped":   q.dropped,
                    "processed": q.processed,
                }
                for q in _ALL_QUEUES
            },
        )
    ), 200


# ── 진입점 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JetRover 이미지 수신 서버")
    parser.add_argument("--host",     default=HOST,           help="바인딩할 호스트")
    parser.add_argument("--port",     type=int, default=PORT,  help="포트 번호")
    parser.add_argument("--save-dir", default=str(SAVE_DIR),   help="이미지 저장 폴더")
    parser.add_argument("--verbose",  action="store_true",     help="디버그 로그")
    args = parser.parse_args()

    SAVE_DIR = Path(args.save_dir)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    start_workers()

    log.info("서버 시작 → http://%s:%d", args.host, args.port)
    log.info("이미지 저장 경로: %s", SAVE_DIR.resolve())
    log.info("라이브 뷰어: http://localhost:%d/", args.port)

    app.run(host=args.host, port=args.port, threaded=True)

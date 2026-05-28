"""
로컬 서버 — 큐 기반 팬아웃 아키텍처

POST /upload 에서 수신한 프레임을 3개의 독립 큐로 팬아웃:
  gui_queue      (maxsize=1,   drop_oldest=True)  → MJPEG 스트림 (항상 최신)
  db_queue       (maxsize=100, drop_oldest=False) → 디스크 저장
  analysis_queue (maxsize=10,  drop_oldest=False) → YOLO 분석

각 큐는 전용 워커 스레드만 소비 → 메모리 충돌·병목 방지
"""

from __future__ import annotations

import json
import queue
import sqlite3
import time
import logging
import threading
import numpy as np
import cv2
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, Response, render_template_string

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── 설정 ──────────────────────────────────────────────────────────────────────
HOST               = "0.0.0.0"
PORT               = 5000
SAVE_DIR           = Path("received_images")
DB_PATH            = Path(__file__).parent / "autocam_rover.db"
MAX_CONTENT        = 10 * 1024 * 1024
DB_RECONNECT_DELAY = 2   # DB 재연결 대기(초)
DB_RECONNECT_TRIES = 3   # DB 재연결 최대 시도 횟수
# ─────────────────────────────────────────────────────────────────────────────


def _reconnect_db(old_conn: sqlite3.Connection, db_path: str, label: str) -> sqlite3.Connection:
    """기존 DB 연결을 닫고 재연결을 시도한다. 성공 여부와 무관하게 새 연결 객체 반환."""
    try:
        old_conn.close()
    except Exception:
        pass
    log.warning("[%s] DB 연결 끊김 — %ds 후 재연결 시도 (%s)", label, DB_RECONNECT_DELAY, db_path)
    time.sleep(DB_RECONNECT_DELAY)
    return sqlite3.connect(db_path)


# ── 데이터 클래스 ────────────────────────────────────────────────────────────
@dataclass
class Frame:
    data:           bytes
    event_id:       int
    filename:       str
    captured_at:    str    # JetRover 촬영 시각
    timestamp:      str    # 서버 수신 시각 (received_at)
    event_type:     str
    robot_location: dict   # {"x": float, "y": float}


@dataclass
class AnalysisRecord:
    event_id:               int
    person_detected:        bool
    fallen_object_detected: bool   # 현재는 항상 False (미구현)
    obstacle_detected:      bool   # 현재는 항상 False (미구현)
    motion_detected:        bool   # 움직임 예상 객체(사람 포함) 존재 시 True
    risk_level:             str    # 사람 감지 시 "high", 아니면 "low"
    result_summary:         str
    analyzed_at:            str


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
gui_queue      = FrameQueue("gui",      maxsize=1,   drop_oldest=True)   # 최신 프레임만 유지
db_queue       = FrameQueue("db",       maxsize=100, drop_oldest=False)  # 저장 버퍼
analysis_queue = FrameQueue("analysis", maxsize=10,  drop_oldest=False)  # YOLO 분석

_ALL_QUEUES: list[FrameQueue] = [gui_queue, db_queue, analysis_queue]


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
            continue
        with _latest_lock:
            _latest_frame = frame.data


# ── 순번 카운터 ───────────────────────────────────────────────────────────────
_event_counter      = 0
_event_counter_lock = threading.Lock()

def _init_event_counter() -> None:
    """서버 재시작 시 index.jsonl 의 마지막 event_id 를 읽어 이어받음."""
    global _event_counter
    index_path = SAVE_DIR / "index.jsonl"
    if not index_path.exists():
        return
    last_id = 0
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                eid = int(json.loads(line).get("event_id", 0))
                if eid > last_id:
                    last_id = eid
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
    _event_counter = last_id
    log.info("event_id 카운터 초기화: %d 에서 재개", _event_counter)

def _next_event_id() -> int:
    global _event_counter
    with _event_counter_lock:
        _event_counter += 1
        return _event_counter


# 가장 최근 저장 레코드 — 페이지 최초 로드 시 /latest 가 읽음
_latest_record: dict = {}
_latest_record_lock = threading.Lock()

# 가장 최근 분석 결과 — 페이지 최초 로드 시 /latest_analysis 가 읽음
_latest_analysis: dict = {}
_latest_analysis_lock = threading.Lock()

# SSE 클라이언트 큐 목록 — db_worker가 push, /events 가 consume
_sse_clients: list[queue.Queue] = []
_sse_lock = threading.Lock()

def _sse_push(record: dict, event_name: str | None = None) -> None:
    """저장 완료된 레코드를 연결된 모든 브라우저로 즉시 전송.
    event_name 지정 시 named SSE event로 전송."""
    msg = (record, event_name)
    with _sse_lock:
        for q in list(_sse_clients):
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass  # 느린 클라이언트는 건너뜀


# DB 워커 ───────────────────────────────────────────────────────────────────
def _db_worker() -> None:
    """db_queue 에서 프레임을 꺼내 이미지 저장 후 index.jsonl 기록 + autocam_rover.db 삽입."""
    db_path = str(DB_PATH)
    conn    = sqlite3.connect(db_path)
    while True:
        frame = db_queue.get()
        if frame is None:
            continue
        img_path   = SAVE_DIR / frame.filename
        index_path = SAVE_DIR / "index.jsonl"
        try:
            img_path.write_bytes(frame.data)
            loc = frame.robot_location
            record = {
                "event_id":       frame.event_id,
                "captured_at":    frame.captured_at,
                "received_at":    frame.timestamp,
                "robot_location": loc,
                "event_type":     frame.event_type,
                "image_path":     str(img_path),
            }
            with index_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            params = (
                frame.captured_at, frame.timestamp,
                json.dumps(loc, ensure_ascii=False),
                frame.event_type, str(img_path),
            )
            db_ok = False
            for attempt in range(DB_RECONNECT_TRIES):
                try:
                    conn.execute(
                        "INSERT INTO event_logs "
                        "(captured_at, received_at, robot_location, event_type, image_path) "
                        "VALUES (?, ?, ?, ?, ?)",
                        params,
                    )
                    conn.commit()
                    db_ok = True
                    break
                except sqlite3.Error as exc:
                    log.warning("[db] DB 오류 (시도 %d/%d): %s", attempt + 1, DB_RECONNECT_TRIES, exc)
                    conn = _reconnect_db(conn, db_path, "db")
            if not db_ok:
                log.error("[db] DB 삽입 %d회 모두 실패 — SSE/latest 업데이트 차단", DB_RECONNECT_TRIES)
                continue
            with _latest_record_lock:
                _latest_record.clear()
                _latest_record.update(record)
            _sse_push(record, event_name="record")
            log.info("[db] #%d 저장: %s", frame.event_id, frame.filename)
        except OSError as exc:
            log.error("[db] 저장 실패: %s", exc)


# ── YOLO 분석 ────────────────────────────────────────────────────────────────
_yolo = None

def _load_yolo() -> None:
    """YOLOv8n 모델을 최초 1회 로드. ultralytics 미설치 시 경고만 출력."""
    global _yolo
    try:
        from ultralytics import YOLO
        _yolo = YOLO("yolov8n.pt")
        log.info("YOLO 모델 로드 완료 (yolov8n.pt)")
    except ImportError:
        log.warning("ultralytics 패키지 없음 — YOLO 분석 비활성화 (pip install ultralytics)")
    except Exception as exc:
        log.warning("YOLO 모델 로드 실패: %s — 분석 비활성화", exc)


def _detect_person(img: np.ndarray) -> bool:
    """클래스 0(person)이 하나라도 감지되면 True."""
    if _yolo is None:
        return False
    results = _yolo(img, classes=[0], verbose=False)
    for r in results:
        if len(r.boxes) > 0:
            return True
    return False


def _detect_fallen_object(img: np.ndarray) -> bool:
    """낙하물 감지 — 미구현, 항상 False."""
    return False


def _detect_obstacle(img: np.ndarray) -> bool:
    """장애물 감지 — 미구현, 항상 False."""
    return False


def _detect_risk(person: bool, fallen_object: bool, obstacle: bool) -> str:
    """위험 수준 판단 — 사람 감지 시 'high', 아니면 'low'."""
    return "high" if person else "low"


def _run_analysis(event_id: int, img: np.ndarray) -> AnalysisRecord:
    person        = _detect_person(img)
    fallen_object = _detect_fallen_object(img)
    obstacle      = _detect_obstacle(img)
    motion        = person
    risk          = _detect_risk(person, fallen_object, obstacle)

    parts = []
    if person:
        parts.append("사람 감지됨")
    elif fallen_object:
        parts.append("낙하물 감지됨")
    elif obstacle:
        parts.append("장애물 감지됨")
    else:
        parts.append("감지 없음")
    parts.append("움직임O" if motion else "움직임X")

    return AnalysisRecord(
        event_id=event_id,
        person_detected=person,
        fallen_object_detected=fallen_object,
        obstacle_detected=obstacle,
        motion_detected=motion,
        risk_level=risk,
        result_summary=", ".join(parts),
        analyzed_at=datetime.now().isoformat(timespec="milliseconds"),
    )


def _save_analysis(record: AnalysisRecord) -> None:
    """analysis.jsonl 에 분석 결과 추가 저장."""
    path  = SAVE_DIR / "analysis.jsonl"
    entry = {
        "event_id":               record.event_id,
        "person_detected":        record.person_detected,
        "fallen_object_detected": record.fallen_object_detected,
        "obstacle_detected":      record.obstacle_detected,
        "motion_detected":        record.motion_detected,
        "risk_level":             record.risk_level,
        "result_summary":         record.result_summary,
        "analyzed_at":            record.analyzed_at,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# 분석 워커 ──────────────────────────────────────────────────────────────────
def _analysis_worker() -> None:
    """analysis_queue 에서 프레임을 꺼내 YOLO 추론 후 analysis.jsonl 기록 + autocam_rover.db 삽입."""
    db_path = str(DB_PATH)
    conn    = sqlite3.connect(db_path)
    while True:
        frame = analysis_queue.get()
        if frame is None:
            continue
        try:
            nparr = np.frombuffer(frame.data, dtype=np.uint8)
            img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                log.warning("[analysis] imdecode 실패 — event_id=%d", frame.event_id)
                continue
            record = _run_analysis(frame.event_id, img)
            _save_analysis(record)
            params = (
                int(record.person_detected), int(record.fallen_object_detected),
                int(record.obstacle_detected), int(record.motion_detected),
                record.risk_level, record.result_summary, record.analyzed_at,
            )
            db_ok = False
            for attempt in range(DB_RECONNECT_TRIES):
                try:
                    conn.execute(
                        "INSERT INTO analysis_results "
                        "(person_detected, fallen_object_detected, obstacle_detected, "
                        "motion_detected, risk_level, result_summary, analyzed_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        params,
                    )
                    conn.commit()
                    db_ok = True
                    break
                except sqlite3.Error as exc:
                    log.warning("[analysis] DB 오류 (시도 %d/%d): %s", attempt + 1, DB_RECONNECT_TRIES, exc)
                    conn = _reconnect_db(conn, db_path, "analysis")
            if not db_ok:
                log.error("[analysis] DB 삽입 %d회 모두 실패 — SSE/latest 업데이트 차단", DB_RECONNECT_TRIES)
                continue
            log.info("[analysis] #%d  %s", record.event_id, record.result_summary)
            entry = {
                "event_id":               record.event_id,
                "person_detected":        record.person_detected,
                "fallen_object_detected": record.fallen_object_detected,
                "obstacle_detected":      record.obstacle_detected,
                "motion_detected":        record.motion_detected,
                "risk_level":             record.risk_level,
                "result_summary":         record.result_summary,
                "analyzed_at":            record.analyzed_at,
            }
            with _latest_analysis_lock:
                _latest_analysis.clear()
                _latest_analysis.update(entry)
            _sse_push(entry, event_name="analysis")
        except Exception as exc:
            log.error("[analysis] 오류 (event_id=%d): %s", frame.event_id, exc)


# ── 워커 시작 ─────────────────────────────────────────────────────────────────
def start_workers() -> list[threading.Thread]:
    """워커 스레드를 데몬으로 시작. 메인 프로세스 종료 시 자동 정리됨."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    _init_event_counter()
    _load_yolo()
    workers = [
        threading.Thread(target=_gui_worker,      name="gui-worker",      daemon=True),
        threading.Thread(target=_db_worker,        name="db-worker",       daemon=True),
        threading.Thread(target=_analysis_worker,  name="analysis-worker", daemon=True),
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
def validate_image(data: bytes) -> tuple[bool, str]:
    """cv2.imdecode() 로 실제 디코딩해 유효성 검사."""
    nparr = np.frombuffer(data, dtype=np.uint8)
    if cv2.imdecode(nparr, cv2.IMREAD_COLOR) is None:
        return False, "cv2.imdecode() 실패 — 손상되거나 지원하지 않는 이미지 형식"
    return True, ""


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
    body     { background:#111; color:#eee; font-family:sans-serif;
               display:flex; flex-direction:column; align-items:center; padding:20px; }
    img      { max-width:100%; border:2px solid #444; border-radius:6px; }
    h1       { margin-bottom:12px; }
    .bar     { margin-top:14px; display:flex; gap:10px; align-items:center; }
    input    { padding:6px 10px; border-radius:4px; border:1px solid #555;
               background:#222; color:#eee; width:200px; }
    button   { padding:8px 18px; border-radius:4px; border:none;
               background:#2563eb; color:#fff; cursor:pointer; font-size:1em; }
    button:hover { background:#1d4ed8; }
    #msg     { margin-top:8px; font-size:0.85em; color:#6ee7b7; min-height:1.2em; }
    #stats   { margin-top:6px; font-size:0.8em; color:#888; }
    .json-wrap  { margin-top:16px; width:100%; max-width:640px; }
    .json-label { font-size:0.78em; color:#6b7280; margin-bottom:4px; }
    .json-out   { margin:0; padding:14px; background:#0d1117; border:1px solid #333;
                  border-radius:6px; font-size:0.8em; white-space:pre;
                  overflow-x:auto; min-height:48px; }
    #json-out      { color:#a5f3fc; }
    #analysis-out  { color:#86efac; }
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
  <div id="stats">서버: {{ host }}:{{ port }} | 저장: {{ save_dir }}</div>

  <div class="json-wrap">
    <div class="json-label">최근 촬영 JSON</div>
    <pre id="json-out" class="json-out">대기 중...</pre>
  </div>

  <div class="json-wrap">
    <div class="json-label">최근 분석 결과 (YOLO)</div>
    <pre id="analysis-out" class="json-out">대기 중...</pre>
  </div>

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
          d.message + (d.label ? ' [' + d.label + ']' : '') + '  ' + d.timestamp;
      })
      .catch(e => document.getElementById('msg').textContent = '오류: ' + e);
    }

    // 페이지 최초 로드 시 마지막 레코드/분석 결과 한 번씩 가져옴
    fetch('/latest')
      .then(r => r.json())
      .then(d => {
        if (d.record)
          document.getElementById('json-out').textContent =
            JSON.stringify(d.record, null, 2);
      })
      .catch(() => {});

    fetch('/latest_analysis')
      .then(r => r.json())
      .then(d => {
        if (d.record)
          document.getElementById('analysis-out').textContent =
            JSON.stringify(d.record, null, 2);
      })
      .catch(() => {});

    // 이후 갱신은 서버 푸시(SSE)로만 수행
    const es = new EventSource('/events');
    es.addEventListener('record', function(e) {
      document.getElementById('json-out').textContent =
        JSON.stringify(JSON.parse(e.data), null, 2);
    });
    es.addEventListener('analysis', function(e) {
      document.getElementById('analysis-out').textContent =
        JSON.stringify(JSON.parse(e.data), null, 2);
    });
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

    valid, err = validate_image(data)
    if not valid:
        log.warning("이미지 검증 실패: %s", err)
        return jsonify(build_response("error", error=err)), 422

    now = datetime.now()
    received_at = now.isoformat(timespec="milliseconds")
    ts          = now.strftime("%Y%m%d_%H%M%S_%f")

    captured_at = request.form.get("captured_at") or received_at
    event_type  = request.form.get("event_type") or "motion"   # 기본값: motion
    try:
        loc_x = float(request.form.get("location_x", 0))
        loc_y = float(request.form.get("location_y", 0))
    except ValueError:
        loc_x, loc_y = 0.0, 0.0                               # 기본값: 0, 0

    event_id=_next_event_id()
    frame = Frame(
        data=data,
        event_id=event_id,
        filename=f"{ts}_{event_id}.jpg",
        captured_at=captured_at,
        timestamp=received_at,
        event_type=event_type,
        robot_location={"x": loc_x, "y": loc_y},
    )
    dispatch(frame)

    return jsonify(
        build_response("ok",
            event_id=frame.event_id,
            image_path=str(SAVE_DIR / frame.filename),
            captured_at=frame.captured_at,
            received_at=received_at,
            event_type=frame.event_type,
            robot_location=frame.robot_location,
            queued={q.name: q.qsize for q in _ALL_QUEUES},
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


@app.route("/latest")
def latest():
    """페이지 최초 로드 시 사용 — 가장 마지막 저장 레코드를 반환."""
    with _latest_record_lock:
        record = dict(_latest_record)
    if not record:
        return jsonify(build_response("ok", record=None)), 200
    return jsonify(build_response("ok", record=record)), 200


@app.route("/latest_analysis")
def latest_analysis():
    """페이지 최초 로드 시 사용 — 가장 마지막 분석 결과를 반환."""
    with _latest_analysis_lock:
        record = dict(_latest_analysis)
    if not record:
        return jsonify(build_response("ok", record=None)), 200
    return jsonify(build_response("ok", record=record)), 200


def _sse_generator(q: queue.Queue):
    """이미지 저장/분석 완료 시 레코드를 SSE 형식으로 스트리밍."""
    try:
        while True:
            try:
                record, event_name = q.get(timeout=25)
                data_line = f"data: {json.dumps(record, ensure_ascii=False)}\n"
                if event_name:
                    yield f"event: {event_name}\n{data_line}\n"
                else:
                    yield f"{data_line}\n"
            except queue.Empty:
                yield ": heartbeat\n\n"
    except GeneratorExit:
        pass  # 클라이언트 정상 종료
    finally:
        with _sse_lock:
            try:
                _sse_clients.remove(q)
            except ValueError:
                pass


@app.route("/events")
def events():
    """브라우저와 SSE 연결을 유지하며 새 이미지 저장 시 즉시 푸시."""
    q: queue.Queue = queue.Queue(maxsize=10)
    with _sse_lock:
        _sse_clients.append(q)
    return Response(
        _sse_generator(q),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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

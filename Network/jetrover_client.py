import cv2
import time
import threading
import requests
import argparse
import logging
from datetime import datetime
from typing import Tuple, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── 설정 ──────────────────────────────────────────────────────────────────────
DEFAULT_SERVER     = "http://192.168.0.10:5000"   # 로컬 서버 IP로 변경
JPEG_QUALITY       = 80       # JPEG 압축 품질 (0-100)
STREAM_FPS         = 10       # 서버로 전송할 초당 프레임 수(상한)
CAPTURE_FPS        = 15       # 카메라 캡처 FPS (전송 FPS 이상이면 충분)
MOTION_THRESH      = 3.0      # 프레임 변화 임계값. 0 이면 게이팅 비활성(항상 전송)
KEEPALIVE_INTERVAL = 2.0      # 변화가 없어도 최소 이 간격(초)마다 1장은 전송
RECONNECT_DELAY    = 3        # 서버 연결 실패 시 재시도 간격(초)
REQUEST_TIMEOUT    = 5        # 요청 타임아웃(초)
# ─────────────────────────────────────────────────────────────────────────────

# 카메라 스레드가 최신 프레임을 여기 보관
_current_frame = None
_frame_lock    = threading.Lock()
_running       = True


# ── 카메라 ────────────────────────────────────────────────────────────────────

def gstreamer_pipeline(
    sensor_id: int = 0,
    width: int = 640,
    height: int = 480,
    framerate: int = CAPTURE_FPS,
    flip_method: int = 0,
) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! appsink"
    )


def open_camera(use_csi: bool, camera_index: int, width: int, height: int,
                capture_fps: int) -> cv2.VideoCapture:
    if use_csi:
        # CSI: 파이프라인에서 직접 framerate 를 낮춰 캡처/변환 부하를 줄인다.
        cap = cv2.VideoCapture(
            gstreamer_pipeline(sensor_id=camera_index, width=width,
                               height=height, framerate=capture_fps),
            cv2.CAP_GSTREAMER,
        )
    else:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # USB: 드라이버 캡처 FPS 를 낮춰 USB 대역폭·디코딩 부하를 줄인다.
        cap.set(cv2.CAP_PROP_FPS, capture_fps)

    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다. 연결 상태를 확인하세요.")
    return cap


def camera_reader(cap: cv2.VideoCapture):
    """
    백그라운드 스레드: 카메라를 계속 읽어 _current_frame 을 갱신.
    CSI 카메라는 버퍼 문제로 read()를 지속 호출해야 최신 프레임을 얻을 수 있음.
    """
    global _current_frame, _running
    while _running:
        ret, frame = cap.read()
        if ret:
            with _frame_lock:
                _current_frame = frame
        else:
            log.warning("카메라 read() 실패 — 재시도 중")
            time.sleep(0.1)


# ── 네트워크 ──────────────────────────────────────────────────────────────────

def poll_trigger(session: requests.Session, trigger_url: str) -> Tuple[bool, dict]:
    """GET /trigger 를 호출해 촬영 신호 여부 반환."""
    try:
        resp = session.get(trigger_url, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("triggered", False), data
        log.warning("트리거 폴링 응답 코드: %d", resp.status_code)
    except requests.exceptions.ConnectionError:
        log.warning("서버에 연결할 수 없습니다 (%s)", trigger_url)
        time.sleep(RECONNECT_DELAY)
    except requests.exceptions.Timeout:
        log.warning("트리거 폴링 타임아웃")
    except Exception as exc:
        log.error("트리거 폴링 오류: %s", exc)
    return False, {}


def encode_frame(frame, quality: int = JPEG_QUALITY) -> bytes:
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok, buf = cv2.imencode(".jpg", frame, encode_params)
    if not ok:
        raise RuntimeError("프레임 JPEG 인코딩 실패")
    return buf.tobytes()


def upload_frame(session: requests.Session, upload_url: str, jpeg_bytes: bytes,
                 meta: dict, verbose: bool = False) -> bool:
    """인코딩된 프레임을 서버에 업로드. meta 에 event_type·robot_location 등을 담는다."""
    now         = datetime.now()
    captured_at = now.isoformat(timespec="milliseconds")
    timestamp   = now.strftime("%Y%m%d_%H%M%S_%f")
    loc         = meta.get("robot_location", {})
    files     = {"image": (f"{timestamp}.jpg", jpeg_bytes, "image/jpeg")}
    form_data = {
        "captured_at": captured_at,
        "event_type":  meta.get("event_type") or "motion",
        "location_x":  str(loc.get("x", 0)),
        "location_y":  str(loc.get("y", 0)),
    }
    try:
        resp = session.post(upload_url, files=files, data=form_data, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            if verbose:
                r   = resp.json()
                loc = r.get("robot_location", {})
                log.info(
                    "[#%s] event_type=%s  location=(%s, %s)  captured=%s  path=%s",
                    r.get("event_id",    "?"),
                    r.get("event_type",  "?"),
                    loc.get("x", 0),
                    loc.get("y", 0),
                    r.get("captured_at", "?"),
                    r.get("image_path",  "?"),
                )
            return True
        log.warning("업로드 응답 코드: %d  body=%s", resp.status_code, resp.text[:200])
    except requests.exceptions.ConnectionError:
        log.warning("업로드 서버에 연결할 수 없습니다")
    except requests.exceptions.Timeout:
        log.warning("업로드 타임아웃")
    except Exception as exc:
        log.error("업로드 오류: %s", exc)
    return False


# ── 프레임 변화 감지(게이팅) ───────────────────────────────────────────────────

def _downscaled_gray(frame: np.ndarray) -> np.ndarray:
    """차분 비교용 저해상도 그레이스케일. resize/cvtColor 는 JPEG 인코딩보다 훨씬 싸다."""
    small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def _frame_changed(prev_small: Optional[np.ndarray], cur_small: np.ndarray,
                   thresh: float) -> bool:
    """이전 프레임 대비 평균 절대차가 임계값을 넘으면 변화로 판단."""
    if thresh <= 0 or prev_small is None:
        return True
    diff = float(cv2.absdiff(cur_small, prev_small).mean())
    return diff > thresh


# ── 스트리밍 워커 ─────────────────────────────────────────────────────────────

def _streaming_worker(upload_url: str, stream_fps: float, quality: int,
                      motion_thresh: float, keepalive: float) -> None:
    """별도 스레드: STREAM_FPS 상한 내에서 '변화가 있을 때만' 최신 프레임을 업로드."""
    session        = requests.Session()
    frame_interval = 1.0 / max(stream_fps, 0.1)
    sent_count     = 0
    skipped_count  = 0
    fail_streak    = 0
    prev_small     = None
    last_sent_t    = 0.0

    log.info("스트리밍 워커 시작 → 상한 %.1f FPS, motion_thresh=%.1f, keepalive=%.1fs",
             stream_fps, motion_thresh, keepalive)
    try:
        while _running:
            t0 = time.monotonic()

            with _frame_lock:
                frame = _current_frame.copy() if _current_frame is not None else None

            if frame is not None:
                now       = time.monotonic()
                cur_small = _downscaled_gray(frame)
                moved     = _frame_changed(prev_small, cur_small, motion_thresh)
                force     = (now - last_sent_t) >= keepalive   # 정지 장면 keepalive
                prev_small = cur_small

                if moved or force:
                    try:
                        jpeg = encode_frame(frame, quality)
                    except RuntimeError as e:
                        log.error(e)
                    else:
                        if upload_frame(session, upload_url, jpeg, {"event_type": "stream"}):
                            sent_count += 1
                            fail_streak = 0
                            last_sent_t = now
                            log.debug("[stream] #%d 전송 (skip 누적 %d)", sent_count, skipped_count)
                        else:
                            fail_streak += 1
                            if fail_streak >= 5:
                                log.warning("[stream] 업로드 연속 %d회 실패 — %ds 대기", fail_streak, RECONNECT_DELAY)
                                time.sleep(RECONNECT_DELAY)
                                fail_streak = 0
                else:
                    skipped_count += 1   # 변화 없음 → 인코딩·업로드 생략

            elapsed = time.monotonic() - t0
            wait    = frame_interval - elapsed
            if wait > 0:
                time.sleep(wait)
    finally:
        session.close()
        log.info("스트리밍 워커 종료 (전송 %d / 생략 %d 프레임)", sent_count, skipped_count)


# ── 메인 루프 ─────────────────────────────────────────────────────────────────

def run(args):
    global _running

    base_url    = args.server.rstrip("/")
    upload_url  = f"{base_url}/upload"
    trigger_url = f"{base_url}/trigger"

    log.info("카메라 초기화 중... (CSI=%s, index=%d, capture_fps=%d)",
             args.csi, args.camera, args.capture_fps)
    cap = open_camera(args.csi, args.camera, args.width, args.height, args.capture_fps)
    log.info("카메라 오픈 완료 (%dx%d @ %dfps)", args.width, args.height, args.capture_fps)

    # 카메라 버퍼를 계속 비워 최신 프레임 유지
    reader = threading.Thread(target=camera_reader, args=(cap,), daemon=True)
    reader.start()

    # 실시간 스트리밍 워커 (별도 세션, 별도 스레드)
    streamer = threading.Thread(
        target=_streaming_worker,
        args=(upload_url, args.stream_fps, args.quality, args.motion_thresh, args.keepalive),
        daemon=True,
    )
    streamer.start()

    # 트리거 폴링 전용 세션 (스트리밍 세션과 분리)
    session     = requests.Session()
    shot_count  = 0
    fail_streak = 0

    log.info("트리거 폴링 시작 → %s  (간격: %.1fs)", trigger_url, args.poll_interval)

    try:
        while True:
            triggered, trigger_data = poll_trigger(session, trigger_url)

            if triggered:
                log.info("촬영 신호 수신 — label=%r", trigger_data.get("label", ""))

                with _frame_lock:
                    frame = _current_frame.copy() if _current_frame is not None else None

                if frame is None:
                    log.warning("아직 프레임이 없습니다 — 건너뜀")
                else:
                    # 트리거 촬영은 게이팅과 무관하게 항상 전송
                    try:
                        jpeg = encode_frame(frame, args.quality)
                    except RuntimeError as e:
                        log.error(e)
                    else:
                        success = upload_frame(session, upload_url, jpeg,
                                               trigger_data, verbose=True)
                        if success:
                            shot_count += 1
                            fail_streak = 0
                        else:
                            fail_streak += 1
                            if fail_streak >= 5:
                                log.warning("업로드 연속 %d회 실패 — %ds 대기", fail_streak, RECONNECT_DELAY)
                                time.sleep(RECONNECT_DELAY)
                                fail_streak = 0

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        log.info("사용자 종료 (트리거 촬영 %d장)", shot_count)
    finally:
        _running = False
        cap.release()
        session.close()


# ── 인수 파싱 ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="JetRover 실시간 영상 스트리밍 + 트리거 촬영 클라이언트 (부하 절감)")
    parser.add_argument("--server",        default=DEFAULT_SERVER,            help="서버 베이스 URL (예: http://192.168.0.10:5000)")
    parser.add_argument("--quality",       type=int,   default=JPEG_QUALITY,  help="JPEG 압축 품질 (0-100)")
    parser.add_argument("--stream-fps",    type=float, default=STREAM_FPS,    help="실시간 스트리밍 초당 프레임 수(상한)")
    parser.add_argument("--capture-fps",   type=int,   default=CAPTURE_FPS,   help="카메라 캡처 FPS (전송 FPS 이상이면 충분)")
    parser.add_argument("--motion-thresh", type=float, default=MOTION_THRESH, help="프레임 변화 임계값. 0이면 게이팅 끄고 항상 전송")
    parser.add_argument("--keepalive",     type=float, default=KEEPALIVE_INTERVAL, help="정지 장면에서 최소 전송 간격(초)")
    parser.add_argument("--poll-interval", type=float, default=1.0,           help="트리거 폴링 간격(초)")
    parser.add_argument("--width",         type=int,   default=640,           help="캡처 해상도 가로")
    parser.add_argument("--height",        type=int,   default=480,           help="캡처 해상도 세로")
    parser.add_argument("--camera",        type=int,   default=0,             help="카메라 인덱스 또는 센서 ID")
    parser.add_argument("--csi",           action="store_true",               help="CSI 카메라 사용 (기본: USB)")
    parser.add_argument("--verbose",       action="store_true",               help="디버그 로그 출력")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    run(args)

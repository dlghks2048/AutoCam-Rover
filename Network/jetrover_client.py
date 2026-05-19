"""
JetRover Client
- 서버의 GET /trigger 를 폴링하다가 신호가 오면 카메라로 1장 캡처 후 업로드
- CSI 카메라(Jetson 기본) 또는 USB 카메라 모두 지원
"""

import cv2
import time
import threading
import requests
import argparse
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── 설정 ──────────────────────────────────────────────────────────────────────
DEFAULT_SERVER   = "http://192.168.0.10:5000"   # 로컬 서버 IP로 변경
JPEG_QUALITY     = 80       # JPEG 압축 품질 (0-100)
POLL_INTERVAL    = 0.5      # 트리거 폴링 간격(초)
RECONNECT_DELAY  = 3        # 서버 연결 실패 시 재시도 간격(초)
REQUEST_TIMEOUT  = 5        # 요청 타임아웃(초)
# ─────────────────────────────────────────────────────────────────────────────

# 카메라 스레드가 최신 프레임을 여기 보관 (트리거 시 즉시 사용)
_current_frame = None
_frame_lock    = threading.Lock()
_running       = True


# ── 카메라 ────────────────────────────────────────────────────────────────────

def gstreamer_pipeline(
    sensor_id: int = 0,
    width: int = 640,
    height: int = 480,
    framerate: int = 30,
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


def open_camera(use_csi: bool, camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    if use_csi:
        cap = cv2.VideoCapture(gstreamer_pipeline(sensor_id=camera_index, width=width, height=height),
                               cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

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

def poll_trigger(session: requests.Session, trigger_url: str) -> tuple[bool, dict]:
    """
    GET /trigger 를 호출해 촬영 신호 여부 반환.
    반환: (triggered 여부, 응답 JSON 전체)
    """
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


def upload_frame(session: requests.Session, upload_url: str, jpeg_bytes: bytes, meta: dict) -> bool:
    """
    인코딩된 프레임을 서버에 업로드.
    meta 에는 트리거 응답의 부가 정보(label 등)가 담겨 있음.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    files = {"image": (f"{timestamp}.jpg", jpeg_bytes, "image/jpeg")}
    try:
        resp = session.post(upload_url, files=files, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            result = resp.json()
            log.info(
                "업로드 완료 → %s  %s  label=%r",
                result.get("file", "?"),
                result.get("image", {}),
                meta.get("label", ""),
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


# ── 메인 루프 ─────────────────────────────────────────────────────────────────

def run(args):
    global _running

    base_url    = args.server.rstrip("/")
    upload_url  = f"{base_url}/upload"
    trigger_url = f"{base_url}/trigger"

    log.info("카메라 초기화 중... (CSI=%s, index=%d)", args.csi, args.camera)
    cap = open_camera(args.csi, args.camera, args.width, args.height)
    log.info("카메라 오픈 완료 (%dx%d)", args.width, args.height)

    # 카메라 버퍼를 계속 비워 최신 프레임 유지
    reader = threading.Thread(target=camera_reader, args=(cap,), daemon=True)
    reader.start()

    session      = requests.Session()
    shot_count   = 0
    fail_streak  = 0

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
                    try:
                        jpeg = encode_frame(frame, args.quality)
                    except RuntimeError as e:
                        log.error(e)
                    else:
                        success = upload_frame(session, upload_url, jpeg, trigger_data)
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
        log.info("사용자 종료 (총 %d장 촬영·전송)", shot_count)
    finally:
        _running = False
        cap.release()
        session.close()


# ── 인수 파싱 ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="JetRover 이벤트 기반 이미지 업로드 클라이언트")
    parser.add_argument("--server",        default=DEFAULT_SERVER,  help="서버 베이스 URL (예: http://192.168.0.10:5000)")
    parser.add_argument("--quality",       type=int, default=JPEG_QUALITY,   help="JPEG 압축 품질 (0-100)")
    parser.add_argument("--poll-interval", type=float, default=POLL_INTERVAL, help="트리거 폴링 간격(초)")
    parser.add_argument("--width",         type=int, default=640,            help="캡처 해상도 가로")
    parser.add_argument("--height",        type=int, default=480,            help="캡처 해상도 세로")
    parser.add_argument("--camera",        type=int, default=0,              help="카메라 인덱스 또는 센서 ID")
    parser.add_argument("--csi",           action="store_true",              help="CSI 카메라 사용 (기본: USB)")
    parser.add_argument("--verbose",       action="store_true",              help="디버그 로그 출력")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    run(args)

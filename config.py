from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
LOG_DIR = BASE_DIR / "logs"
DATABASE_PATH = BASE_DIR / "face_door.db"
MODEL_DIR = BASE_DIR / "models"

ESP32_DOOR_IP = "192.168.1.15"
ESP32_OPEN_PATH = "/open"
REQUEST_TIMEOUT = 3

HOST = "0.0.0.0"
PORT = 5000

FACE_TOLERANCE = 0.50
DETECTION_SCALE_FACTOR = 1.1
DETECTION_MIN_NEIGHBORS = 5
MIN_FACE_SIZE = (80, 80)

# =========================
# Tối ưu tốc độ (không đổi độ chính xác)
# =========================

# Haar cascade sẽ chạy trên bản resize xuống độ rộng này (px) thay vì
# full-res, giúp bước detect nhanh hơn đáng kể. Bbox trả về vẫn được
# quy đổi lại về toạ độ ảnh gốc nên các bước sau (recognize, anti-spoof,
# liveness) vẫn dùng crop full-res như cũ -> độ chính xác không đổi.
# Đặt = None hoặc >= độ rộng frame thực tế để tắt downscale.
DETECTION_DOWNSCALE_WIDTH = 480

# In ra thời gian (ms) từng bước xử lý của mỗi frame trong console,
# dùng để đo bottleneck thực tế trước khi tối ưu tiếp. Nên bật khi
# đang benchmark, tắt khi chạy thật để đỡ log rác.
ENABLE_TIMING_LOG = True

# Số worker thread chạy song song các model trong ensemble anti-spoof.
# None = tự động lấy theo số model đang tải.
ANTI_SPOOF_MAX_WORKERS = None

REQUIRED_STABLE_FRAMES = 3
RECOGNITION_RESET_SECONDS = 1.5

DOOR_OPEN_DURATION = 5.0
PERSON_COOLDOWN_SECONDS = 10.0

LIVENESS_ENABLED = True
LIVENESS_TIMEOUT_SECONDS = 8.0
LIVENESS_CLOSED_FRAMES = 2

JPEG_QUALITY = 85

ANTI_SPOOF_ENABLED = True

# Trỏ tới thư mục chứa các model .pth
ANTI_SPOOF_MODEL_PATH = MODEL_DIR

# Silent-Face: class 1 là khuôn mặt thật
ANTI_SPOOF_LIVE_CLASS_INDEX = 1

ANTI_SPOOF_THRESHOLD = 0.65

REQUIRE_ACTIVE_CHALLENGE = True

CENTER_FACE_MAX_OFFSET = 0.25

# =========================
# Presence Detection
# =========================

# Tỷ lệ diện tích tối thiểu của khuôn mặt chính
# so với toàn bộ khung hình.
# 0.08 = khuôn mặt chiếm khoảng 8% ảnh.
MIN_PRIMARY_FACE_RATIO = 0.08

# =========================
# Presence Detection
# =========================

CENTER_FACE_MAX_OFFSET = 0.25
MIN_PRIMARY_FACE_RATIO = 0.08
PRESENCE_LOST_SECONDS = 2.0

# Yêu cầu thử thách chủ động như chớp mắt
REQUIRE_ACTIVE_CHALLENGE = True

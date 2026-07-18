Đặt model ONNX chống giả tại: models/anti_spoof.onnx

Yêu cầu output: logits hoặc probability [1, C].
Sau khi đặt model, kiểm tra ANTI_SPOOF_LIVE_CLASS_INDEX trong config.py.
Hệ thống mặc định FAIL-CLOSED: thiếu/lỗi model thì cửa không mở.

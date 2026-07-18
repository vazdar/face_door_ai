# Face Door AI Anti-Spoofing

Pipeline an toàn:

1. Phát hiện khuôn mặt.
2. Chỉ chấp nhận **đúng một người**, mặt đủ lớn và gần giữa camera.
3. Nhận diện danh tính từ `dataset/`.
4. Model AI ONNX đánh giá `REAL/FAKE` và trả `live_score`.
5. Thử thách chủ động yêu cầu chớp mắt.
6. Chỉ mở cửa khi tất cả điều kiện đều đạt và nhận diện ổn định nhiều frame.

## Cấu trúc mới

- `anti_spoof.py`: inference model AI ONNX chống ảnh in, ảnh trên điện thoại/màn hình và replay cơ bản.
- `presence.py`: xác định người đang đứng trước màn hình; từ chối khi không có người, quá xa, lệch tâm hoặc có nhiều người.
- `liveness.py`: thử thách chớp mắt để bổ sung cho model AI thụ động.
- `recognizer.py`: xác định danh tính.

## Model AI

Đặt file model tại:

```text
models/anti_spoof.onnx
```

Model phải nhận tensor ảnh NCHW và trả logits/probability `[1, C]`. Kiểm tra metadata model để đặt đúng:

```python
ANTI_SPOOF_LIVE_CLASS_INDEX = 1
ANTI_SPOOF_THRESHOLD = 0.82
```

Nếu model dùng lớp `LIVE=0`, đổi index thành `0`. Hệ thống mặc định **fail-closed**: thiếu model hoặc inference lỗi thì không mở cửa.

Một lựa chọn tham khảo là MiniFASNet/Silent-Face-Anti-Spoofing. Không nên lấy ngẫu nhiên model không rõ preprocessing hoặc thứ tự class; cần thử bằng bộ ảnh thật/ảnh in/màn hình của chính camera trước khi sử dụng.

## Dataset

```text
dataset/
  Nguyen_Van_A/
    1.jpg
    2.jpg
```

## Cài đặt và chạy

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Mở `http://<IP-may-tinh>:5000/`.

## Hiệu chỉnh thực tế

- Thu ít nhất 50-100 mẫu thật và giả từ chính ESP32-CAM.
- Giả gồm: ảnh in, ảnh trên điện thoại, video replay, nhiều độ sáng và góc nghiêng.
- Tăng `ANTI_SPOOF_THRESHOLD` nếu còn lọt ảnh giả.
- Giảm nhẹ threshold nếu người thật bị từ chối quá nhiều.
- Không dùng một model RGB đơn lẻ như lớp bảo vệ duy nhất cho hệ thống có yêu cầu bảo mật cao; nên thêm IR/depth nếu có điều kiện.

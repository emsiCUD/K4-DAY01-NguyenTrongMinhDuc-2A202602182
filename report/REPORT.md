# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics: 8.4.145**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

* **Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):**
  `{"class_id": 468, "class_name": "cab", "rank": 1, "score": 0.510915, "taxonomy_name": "ImageNet-1K"}`.
* **Record này mô tả toàn ảnh như thế nào?**
  Nhãn "cab" (xe taxi) đại diện cho ngữ cảnh chung của toàn bộ khung hình ảnh `traffic`, thay vì chỉ định một đối tượng cụ thể hay một vùng điểm ảnh cụ thể nào trong hình.
* **Ai định nghĩa class list mà checkpoint có thể dự đoán?**
  Con người (những chuyên gia xây dựng tập dữ liệu) đã định nghĩa danh sách lớp này, cụ thể ở đây là hệ thống phân loại (taxonomy) ImageNet-1K được huấn luyện kèm với checkpoint.
* **Vì sao cần giữ cả ID, tên lớp và tên taxonomy?**
  Để đảm bảo tính nhất quán (consistency), giúp hệ thống truy xuất và ánh xạ chính xác. Một nhãn có thể có nhiều tên gọi (ví dụ "cab" hoặc "taxi"), việc giữ ID và taxonomy đóng vai trò như "căn cước" để tránh nhầm lẫn giữa các dự án khác nhau.
* **Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?**
  Guideline cần quy định rõ ràng cách ưu tiên khi gán nhãn: ví dụ ưu tiên chủ thể có kích thước lớn nhất, nằm ở trung tâm ảnh, hoặc chuyển bài toán sang dạng phân loại đa nhãn (multi-label classification).
* **Vì sao model score không phải ground truth?**
  Model score (ví dụ: 0.510915) chỉ là độ tin cậy (confidence) của mô hình dựa trên thuật toán. Nó có thể dự đoán sai. Ground truth phải là dữ liệu thực tế do con người xác nhận dựa trên guideline của dự án.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

* **Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):**
  `{"class_name": "person", "score": 0.912625, "bbox_xyxy": [385.33, 69.24, 498.92, 348.92], "bbox_width": 113.58, "bbox_height": 279.68}`.
* **Diễn giải vị trí box bằng lời:**
  Hộp giới hạn (bounding box) chứa vật thể "person" có góc trên cùng bên trái bắt đầu tại tọa độ x = 385.33, y = 69.24 pixel, và kéo dài đến góc dưới cùng bên phải tại tọa độ x = 498.92, y = 348.92 pixel^^. Kích thước của hộp có chiều rộng là 113.58 pixel và chiều cao là 279.68 pixel.
* **So sánh số prediction ở hai threshold:**
  Khi hạ threshold (ngưỡng điểm số tin cậy), số lượng prediction sẽ tăng lên do mô hình giữ lại cả những dự đoán có độ tin cậy thấp. Ngược lại, nếu nâng threshold lên cao, các dự đoán điểm thấp sẽ bị loại bỏ, làm giảm tổng số prediction.
* **Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?**
  Threshold thấp giúp độ bao phủ cao hơn (ít bỏ sót vật thể) nhưng lại tạo ra nhiều dự đoán sai (false positives/nhiễu rác). Điều này làm tăng khối lượng công việc của reviewer/QC vì họ phải tốn thời gian loại bỏ các box sai.
* **Đề xuất một quy tắc box chặt:**
  Bounding box phải ôm sát nhất có thể vào các mép ngoài cùng nhìn thấy được của vật thể, không được chứa quá nhiều khoảng không gian nền (background) thừa thãi bên trong hộp.
* **Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?**
  Guideline cần quy định rõ: nếu vật bị che khuất trên 50% thì có cần vẽ box nữa không? Và box chỉ nên bao quanh phần đang hiển thị, hay phải ước lượng vẽ bao quanh cả phần bị che khuất (amodal bounding box)? Nếu điểm mờ nhòe khó phân định, annotator cần escalation để xin ý kiến.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

* **Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):**
  `{"instance_id": "kitchen-001", "class_name": "person", "score": 0.899318, "polygon_point_count": 348, "polygon_xy": [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], ...]}`.
* **Polygon bổ sung chi tiết gì so với box?**
  Polygon (đa giác) mô tả chính xác đường viền hình dáng thực tế của vật thể, giúp loại bỏ hoàn toàn các điểm ảnh (pixel) thuộc phông nền (background) vốn bị dính vào khi dùng hình hộp chữ nhật (bounding box).
* **`instance_id` dùng để làm gì và không phải loại ID nào?**
  `instance_id` (như "kitchen-001") dùng để phân biệt các cá thể (object) độc lập trong cùng một bức ảnh (ví dụ phân biệt cái bát A và cái bát B). Nó không phải là class ID (mã lớp) và cũng không phải tracking ID (định danh theo dõi qua nhiều khung hình video).
* **Đề xuất một quy tắc biên mask:**
  Đường viền của mask (polygon) phải bám khít vào contour (đường biên) thực tế của vật thể, không lẹm vào vật thể lân cận và không được lấn ra nền.
* **Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?**
  Guideline cần quy định có được nội suy ranh giới giữa hai vật thể đặt sát nhau hay không. Đối với các vùng bóng râm hoặc mờ viền không thể phân định bằng mắt thường, annotator cần escalation (báo cáo lên cấp trên/domain expert) để thống nhất luật.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ              | Đơn vị/định dạng ground truth                                    | Lỗi hoặc điểm mơ hồ quan sát được                                                     | Annotator làm gì?                                                                                    | Reviewer xem gì?                                                                                        |
| --------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| Phân loại ảnh      | 1 nhãn (Label) cho toàn bộ bức ảnh.                               | Ảnh có nhiều chủ thể (vd: có cả ô tô và người) nhưng model chỉ dự đoán 1 lớp. | Đọc kỹ guideline để quyết định chọn nhãn chủ thể chính xác nhất.                        | Kiểm tra nhãn có mô tả đúng ngữ cảnh/chủ thể lớn nhất theo luật hay không.                |
| Phát hiện vật thể | Nhãn (Label) + Bounding Box (4 tọa độ).                            | Bounding box ôm quá lỏng (chứa nhiều nền) hoặc bỏ sót các vật thể nhỏ.             | Chỉnh sửa các mép của bounding box để ôm sát nhất phần nhìn thấy của vật thể.          | Kiểm tra sai số của các đường biên hộp và xác nhận không có vật thể nào bị bỏ sót.   |
| Instance segmentation | Nhãn (Label) + Polygon (Danh sách các điểm tọa độ bám biên). | Các điểm đa giác bị lẹm góc, không uốn cong sát viền thực tế của vật thể.      | Chấm thêm điểm tọa độ để nắn đường đa giác bám sát vào đường cong của vật thể. | Zoom phóng to ảnh để kiểm tra độ chính xác của đường polygon tại các mép, các góc mờ. |

## 5. An toàn dữ liệu

* **Một quy tắc bảo vệ dữ liệu:**
  Tuyệt đối không tải các dữ liệu nhạy cảm của khách hàng, ảnh chụp khuôn mặt, biển số xe hoặc tài liệu nội bộ công ty lên các hệ thống đám mây công cộng (như Google Colab, GitHub) khi chưa được phép.
* **Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:**
  Người quản lý trực tiếp (Manager), Data Steward hoặc Lab Coach của dự án.

## 6. Danh sách bằng chứng

- [X] `classification_predictions.json`
- [X] `detection_predictions.json`
- [X] `segmentation_predictions.json`
- [X] `IMAGE_ATTRIBUTION.md`
- [X] `visuals/classification_top5.png`
- [X] `visuals/detection_predictions.png`
- [X] `visuals/segmentation_prediction.png`
- [X] Ô validation cuối notebook báo `PASS`.
- [X] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.

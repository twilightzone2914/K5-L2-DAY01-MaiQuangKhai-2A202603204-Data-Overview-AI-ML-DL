# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** GPU (`cuda`)

**Python / PyTorch / Ultralytics:** Python `3.13.16` / PyTorch `2.11.0+cu130` / Ultralytics `8.4.145`

**Checkpoint (kèm SHA-256):**
- `yolo11n-cls.pt` (`sha256: c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7`)
- `yolo11n.pt` (`sha256: 0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`)
- `yolo11n-seg.pt` (`sha256: 55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`)

**Mẫu ảnh COCO công khai (`sample_id` / `coco_image_id` / `sha256`):**
- `traffic` (`COCO 210273`, `640×428`, `sha256: 3ec23de63592c1eef86740fd46fed2cb66170fca02b695d37e6ae379dcee4355`)
- `kitchen` (`COCO 397133`, `640×427`, `sha256: 09e1d25c75f7879bdaa69c327fece5cabacd53939c8c2ef9e87f1c97a2e478c4`)
- `dining` (`COCO 166918`, `480×640`, `sha256: a7f8457580a2bb8635ca7acd1b2061ea7977a84418e5e8441a77ba1f861d7e22`)

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `sample_id`=`traffic` ; `coco_image_id`=`210273` ; `class_id`=`468` ; `class_name`=`cab` ; `rank`=`1` ; `score`=`0.510915` ; `taxonomy_name`=`ImageNet-1K`
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   - **Trả lời:** Record này nói về **cả bức ảnh** (`task = "image_classification"`). Vì trong JSON chỉ có thông tin kích thước toàn ảnh (`image_width = 640`, `image_height = 428`), `class_id`, `class_name`, `rank` và `score` cho toàn cảnh chứ không hề có tọa độ hộp (`bbox_xyxy`) hay đa giác (`polygon_xy`) để định vị bất kỳ vật thể riêng lẻ nào trong ảnh.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   - **Trả lời:** Danh sách tên lớp **đi kèm cố định với taxonomy của dữ liệu huấn luyện và lưu trong checkpoint** (`yolo11n-cls.pt` gắn với bộ nhãn `ImageNet-1K` gồm 1000 lớp). Model không thể tự nghĩ ra tên lớp mới ngoài danh sách này.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
   - **Trả lời:** Vì các bộ dữ liệu/taxonomy khác nhau có thể dùng chung một tên lớp hoặc từ đồng nghĩa nhưng mang mã số (`class_id`) và định nghĩa khác nhau (ví dụ `car`/`cab` trong `ImageNet-1K` khác với `car` có `class_id = 2` trong `COCO-80`). Ghi đủ cả 3 trường giúp tránh nhầm lẫn ngữ nghĩa, đảm bảo truy xuất chính xác phiên bản bộ nhãn và tái lập kết quả giữa các hệ thống.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
   - **Trả lời:** Với bài toán phân loại đơn nhãn (single-label), khi ảnh có nhiều đối tượng (như ảnh `traffic` có cả xe buýt, ô tô con, xe tải nhỏ, người đi bộ), guideline cần quy định rõ **quy tắc ưu tiên chọn lớp chính** (ví dụ: ưu tiên vật thể chiếm diện tích lớn nhất/ở trung tâm ảnh, ưu tiên theo mức độ quan trọng nghiệp vụ, hoặc gán nhãn theo bối cảnh tổng thể của cảnh) và cách xử lý khi không có đối tượng nào chiếm ưu thế.
6. Vì sao không được lấy `score` của model làm ground truth?
   - **Trả lời:** `score` (`0.510915`) chỉ là độ tự tin nội tại (xác suất dự đoán) của model trên bộ lớp `ImageNet-1K`, không phải sự thật được con người kiểm chứng theo guideline. Thực tế trên ảnh `traffic`, cảnh đường phố hỗn hợp có nhiều xe buýt và ô tô con nhưng model lại gán cả bức ảnh là `cab` (xe taxi) với score `0.510915` do bị ép chọn trong taxonomy `ImageNet-1K`.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `sample_id`=`kitchen` ; `class_id`=`0` ; `class_name`=`person` ; `score`=`0.912625` ; `bbox_xyxy`=`[385.33, 69.24, 498.92, 348.92]` ; `bbox_width`=`113.58` ; `bbox_height`=`279.68`
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
   - **Trả lời:** Tính theo hệ tọa độ pixel với gốc `(0, 0)` ở góc trên-trái của bức ảnh `kitchen` (`640 × 427` pixel): hộp bao quanh người đầu bếp (`person`) bắt đầu từ góc trên-trái tại tọa độ `(x_min = 385.33, y_min = 69.24)` pixel, kéo tới góc dưới-phải tại tọa độ `(x_max = 498.92, y_max = 348.92)` pixel; chiều rộng của hộp là `113.58` pixel (`498.92 - 385.33`) và chiều cao là `279.68` pixel (`348.92 - 69.24`).
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
   - **Trả lời:** Trên ảnh `sample_id = kitchen`:
     - Ở ngưỡng `0.20`: có **17 prediction** (`['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl', 'spoon', 'potted plant', 'spoon', 'dining table', 'spoon', 'bottle']`).
     - Ở ngưỡng `0.60`: có **6 prediction** (`['person', 'bowl', 'bowl', 'oven', 'oven', 'person']`).
     - (Ở ngưỡng mặc định `0.35`: có **11 prediction**).
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
   - **Trả lời:** Khi hạ ngưỡng từ `0.60` xuống `0.20`, số lượng hộp tăng từ 6 lên 17 (thêm 11 hộp có score thấp). Nếu dùng prediction để hỗ trợ/đối chiếu, **reviewer** phải kiểm tra kỹ để lọc bỏ các **dương tính giả (false positives / hộp nhận diện nhầm lớp)**, các hộp bị trùng lặp/chồng lấn sai hoặc hộp bao quanh biên chưa chuẩn, thay vì chỉ kiểm tra các vật thể rõ ràng.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
   - **Trả lời:** **Quy tắc vẽ Bounding Box:** *"Vẽ một hộp chữ nhật bao sát nhất (tight box) toàn bộ phần nhìn thấy được của mỗi vật thể thuộc danh sách lớp mục tiêu; cạnh hộp phải chạm sát các điểm cực trị (trên, dưới, trái, phải) của phần vật thể hiển thị trên ảnh, không chừa khoảng trống thừa và không tự suy đoán vẽ mở rộng ra phần bị vật khác che khuất hoàn toàn."*
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   - **Trả lời:**
     - **Quan sát trên ảnh `kitchen` (`detection_predictions.png`):** Đối tượng `person` ở mép trái dưới ảnh (`score = 0.61`) bị cắt bởi khung hình và chỉ lộ một phần cánh tay/bàn tay đang làm bánh; ngoài ra các cái tô (`bowl` ở góc dưới trái, như `bowl 0.38`, `bowl 0.50`) bị chồng lấp và cắt một phần ở mép trái ảnh.
     - **Cách xử lý:**
       - Với trường hợp phổ biến (vật thể bị cắt ở mép ảnh hoặc chỉ lộ một bộ phận nhỏ như cánh tay của `person 0.61`): cần **viết thành quy tắc rõ ràng trong guideline** (ví dụ: quy định ngưỡng tỷ lệ hiển thị tối thiểu hoặc quy tắc có gán nhãn `person` khi chỉ thấy bàn tay/cánh tay ở rìa ảnh hay không) vì tình huống này lặp lại thường xuyên trên nhiều ảnh.
       - Nếu guideline hiện tại chưa quy định ngưỡng che khuất/cắt mép cho trường hợp chỉ lộ cánh tay như vậy, annotator không được tự đoán mà phải **escalation (đưa lên hỏi Lab Coach)** để thống nhất và cập nhật bổ sung vào guideline.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `sample_id`=`kitchen` ; `instance_id`=`kitchen-001` ; `class_id`=`0` ; `class_name`=`person` ; `score`=`0.899318` ; số điểm polygon=`polygon_point_count`=`348` ; 5 điểm đầu của `polygon_xy`=`[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]`
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
   - **Trả lời:** Trong khi `bbox_xyxy` chỉ là hình chữ nhật 4 cạnh nên luôn bao gồm cả nhiều pixel nền (background) hoặc phần chồng lấn của vật thể khác ở các góc, thì `polygon_xy` cung cấp chuỗi tọa độ các đỉnh `[x, y]` bám sát đường viền thực tế của vật thể. Nhờ đó, polygon mô tả chính xác hình dáng hình học, đường cong và diện tích pixel thực sự thuộc về đối tượng (ví dụ dáng người đầu bếp `person 0.90`, hình dáng chiếc muôi `spoon 0.49` hay mặt bàn `dining table 0.58`).
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
   - **Trả lời:**
     - `instance_id` (ví dụ `kitchen-001`, `kitchen-002`) dùng để **phân biệt từng cá thể (instance) riêng biệt** được phát hiện trong phạm vi một bức ảnh/bộ output của bài lab (ví dụ trong ảnh `kitchen` có nhiều cái tô cùng lớp `bowl`, nhưng mỗi cái tô là một instance riêng có `instance_id` và đa giác riêng).
     - Nó **không phải `class_id`**: vì `class_id` định danh loại vật thể trong taxonomy COCO-80 (ví dụ `person` là `0`, `bowl` là `45`), nhiều instance khác nhau có thể có cùng `class_id`.
     - Nó **không phải `tracking_id`**: vì `tracking_id` dùng để theo dõi cùng một đối tượng xuyên suốt nhiều khung hình (frames) theo thời gian trong video, còn `instance_id` ở đây chỉ đánh số thứ tự cục bộ trong từng ảnh tĩnh.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
   - **Trả lời:** **Quy tắc vẽ biên Polygon/Mask:** *"Đặt các đỉnh đa giác bám sát đường biên nhìn thấy được của từng đối tượng riêng lẻ (không gộp hai vật cùng lớp chạm nhau thành một mask); đặt mật độ điểm dày hơn ở các đoạn cong/góc gấp và thưa hơn ở cạnh thẳng để không ăn lẹm vào vật thể cũng không tràn ra pixel nền."*
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
   - **Trả lời:**
     - **Quan sát trên ảnh `kitchen` (`segmentation_prediction.png`):** Ở góc dưới bên trái trên mặt bàn (`dining table 0.58`), cụm các cái tô (`bowl 0.74`, `bowl 0.70`, `bowl 0.53`) đặt sát/chồng lên nhau và chạm vào cánh tay người (`person 0.43`); ngoài ra mask của `dining table 0.58` bị các vật dụng trên bàn che khuất nhiều chỗ và tràn cả lên các tô/bột bánh trên bàn.
     - **Cách xử lý:**
       - Áp dụng **guideline** cho quy tắc tách hai vật cùng lớp chạm nhau (mỗi cái `bowl` phải được vẽ một polygon `instance` độc lập theo đúng phần biên nhìn thấy, và mặt bàn không được vẽ đè lên phần diện tích bị các vật đặt trên bàn che khuất).
       - Thực hiện **escalation (hỏi Lab Coach)** ở những vùng biên bị nhòe tối hoặc lẫn vào bột/khăn trải bàn khó phân định đâu là mép tô, đâu là bột/mặt bàn, hoặc khi chưa rõ có phải trừ lỗ (hole/multipolygon) cho mặt bàn quanh từng đồ vật nhỏ hay không, nhằm tránh mỗi annotator tự suy diễn một kiểu.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn cấp toàn ảnh (image-level): mỗi ảnh gắn `class_id` + `class_name` thuộc một taxonomy xác định (single-label hoặc multi-label tùy bài toán). | Ảnh `traffic` (`classification_top5.png`) có nhiều xe buýt, ô tô con, người đi bộ nhưng model (`yolo11n-cls.pt`) đoán toàn ảnh là `cab` (`class_id=468`, `score=0.510915`) và `minibus` (`score=0.164284`) do lệch pha giữa cảnh đa vật thể COCO và taxonomy `ImageNet-1K`. | Đọc kỹ định nghĩa lớp và quy tắc ưu tiên chủ đề chính trong guideline để chọn đúng lớp cho toàn ảnh; báo escalation nếu ảnh chứa nhiều chủ đề ngang nhau chưa có trong rule. | Kiểm tra nhãn có đúng quy tắc chọn chủ đề chính của guideline và đúng taxonomy dự án không; đảm bảo không chép mù quáng top-1 prediction của model. |
| Phát hiện vật thể | Nhãn cấp vật thể (object-level): mỗi vật thể có một hộp chữ nhật `[x_min, y_min, x_max, y_max]` và `class_id`/`class_name`. | Trên ảnh `kitchen` (`detection_predictions.png`), `person 0.61` ở mép trái chỉ lộ cánh tay; nhiều nồi/chảo treo trên tường không được phát hiện hoặc bị nhầm; khi đổi ngưỡng `0.20` (17 hộp) sang `0.60` (6 hộp) thì mất các vật thật như `cup`, `spoon`, `potted plant`, `dining table`. | Vẽ đầy đủ tight box cho tất cả vật thể mục tiêu xuất hiện trong ảnh (kể cả vật mà model bỏ sót ở ngưỡng cao), xử lý vật cắt mép/che khuất theo guideline hoặc hỏi Lab Coach. | Kiểm tra độ sát của 4 cạnh hộp (tightness), tính đúng đắn của `class_name`, rà soát các vật thể bị bỏ sót (false negatives) và yêu cầu xóa/sửa các hộp dư hoặc vẽ lệch (rework). |
| Instance segmentation | Nhãn cấp cá thể (instance-level): mỗi vật thể riêng biệt có một đa giác `polygon_xy` bám sát biên kèm `class_id`/`class_name` và định danh instance riêng. | Trên ảnh `kitchen` (`segmentation_prediction.png`), các tô `bowl` (`0.74`, `0.70`, `0.53`) chạm/chồng nhau ở góc dưới trái; mask của `dining table 0.58` tràn đè lên cả các vật dụng nằm trên mặt bàn; `oven 0.63` bên phải (thấy ở detection) bị bỏ sót ở segmentation ngưỡng `0.35`. | Vẽ đa giác tách biệt cho từng instance (không gộp 2 `bowl` chạm nhau), đi điểm bám sát biên thật của phần nhìn thấy và loại trừ phần bị vật khác che khuất; escalate các ca biên mờ. | Kiểm tra biên polygon có bám sát mép vật thể không (không lẹm/tràn nền), các vật cùng lớp chạm nhau đã tách riêng instance chưa, và xử lý vùng giao/che khuất giữa mặt bàn với đồ vật đã chuẩn chưa. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   - **Trả lời:** Chỉ sử dụng ba ảnh COCO công khai đi kèm bài lab, tuyệt đối không tải ảnh cá nhân, khuôn mặt, biển số hay dữ liệu nội bộ lên Colab/GitHub và không ghi họ tên, MSSV, email, số điện thoại vào `REPORT.md` hoặc bất kỳ tệp output nào (định danh chỉ đặt ở tên repository).
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   - **Trả lời:** Dừng ngay thao tác tải lên/gán nhãn/commit, xóa dữ liệu đó khỏi phiên làm việc cục bộ/Colab và báo ngay cho **Lab Coach / Giảng viên (GV)** để được hướng dẫn xử lý.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS

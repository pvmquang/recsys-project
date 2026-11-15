# Số liệu đợt 2: toàn danh mục, AUC, cold-start

Checklist 2.4, 2.5, 2.8. Ba mục này **không** đổi giao thức chung: bảng chính vẫn đo bằng
1 phim đúng + 99 phim âm. Chúng đo thêm để biết đọc bảng chính đến đâu thì dừng.

Phần **"Kết luận"** do người viết. Phần **"Số liệu chi tiết"** là output nguyên văn của
`python scripts/exp_giao_thuc.py`. Lấy số từ phần đó. CSV kèm seed và commit nằm ở
`reports/exp_2_*.csv` + `.meta.json`.

> **Quy tắc:** chênh lệch nào có CI 95% chứa 0 thì **không** viết "mô hình X thắng".

---

## Kết luận

### 2.4: Bỏ lấy mẫu thì lợi thế của KNN so với Popularity không còn phân biệt được với nhiễu

| | lấy mẫu 99 phim âm | toàn danh mục (~9.700 phim) |
|---|---|---|
| ItemKNN − Popularity, nDCG@10 | +0.114, **có ý nghĩa** | +0.007, nhiễu |
| UserKNN − Popularity, nDCG@10 | +0.064, **có ý nghĩa** | +0.005, nhiễu |
| ItemKNN − UserKNN, nDCG@10 | +0.051, **có ý nghĩa** | +0.002, nhiễu |

- Thứ tự *trung bình* vẫn giữ: ItemKNN > UserKNN > Popularity > Random. Không có cặp nào
  **đảo chiều** có ý nghĩa.
- Nhưng trên toàn danh mục, phim đúng lọt top-10 chỉ ở 25-35 trên 610 user. Với ngần ấy
  lượt trúng, CI quá rộng để tách ba mô hình. Câu "+32% nDCG" vì vậy chỉ đúng **với giao
  thức lấy mẫu**, và báo cáo phải ghi kèm giao thức đó.
- Một cặp vẫn có ý nghĩa trên toàn danh mục: UserKNN − Popularity về recall@10 (+0.016).
- Muốn tách mô hình trên toàn danh mục thì phải có nhiều user hơn: **ml-1m (6.040 user)**.

### 2.5: Chênh AUC giữa ItemKNN và UserKNN không phải do thế hòa

- Đã tính AUC từ điểm, cặp bằng điểm tính 0.5. AUC của mọi mô hình chỉ đổi ở chữ số thứ ba,
  và ItemKNN − UserKNN vẫn có ý nghĩa (−0.018, CI [−0.034, −0.002]).
- Lý do thật nằm ở bảng "Tách AUC": ItemKNN không chấm được **29%** phim đúng (UserKNN
  17,5%). Khi đã chấm được thì ItemKNN xếp **tốt hơn**: AUC 0.974 so với 0.936.
- Vậy AUC ở đây trộn hai câu hỏi: *"có ý kiến về phim đúng không"* và *"có ý kiến rồi thì
  xếp đúng không"*. Báo cáo nên đưa cả hai số thay vì một AUC gộp.

### 2.8: Cold-start: thứ hạng ItemKNN và UserKNN đảo chiều theo độ dài lịch sử

| lịch sử train | mô hình tốt nhất (nDCG@10) | UserKNN − ItemKNN |
|---|---|---|
| 1 tương tác | **Popularity** | −0.112, ItemKNN hơn |
| 3 tương tác | **UserKNN ≈ Popularity** | +0.099, UserKNN hơn |
| 5 tương tác | **UserKNN** | +0.096, UserKNN hơn |
| 10 tương tác | **UserKNN** | +0.040, UserKNN hơn |
| đầy đủ (19-2.697) | **ItemKNN** | −0.051, ItemKNN hơn |

- ItemKNN cần lịch sử. Với 1-5 tương tác, ItemKNN không có ý kiến về 98,5-99,6% ứng
  viên, nên thứ tự gần như do xáo ngẫu nhiên quyết định. Nó thua Popularity cho tới khoảng
  10 tương tác (ở 10 thì hòa).
- UserKNN với 1 tương tác không có láng giềng nào (100% ứng viên −inf): Pearson của một
  user chỉ có một rating là không xác định. Đúng như ghi chú ở `knn.UserKNN`.
- Đây là căn cứ cho B7, thay cho bảng theo nhóm bị bỏ ở đợt 1: lai ghép kiểu chuyển
  (Popularity → UserKNN → ItemKNN theo số tương tác) có lý do đo được ở từng ngưỡng.
- Hạn chế: cắt 1/4 số user làm ~10% phim mất khỏi train (phim chỉ những user đó xem). nDCG
  của Popularity gần như không đổi (0.3603 → 0.3606), nên ảnh hưởng nhỏ, nhưng có tồn tại.
- Giữ n tương tác **gần nhất** trước thời điểm test, không phải n tương tác đầu tiên. Lý do
  ghi ở `split.truncate_history`.

---

## Số liệu chi tiết

ml-latest-small · seed 42 · k láng giềng = 40 · 610 user kiểm tra

### 2.4 + 2.5: lấy mẫu 99 item âm so với toàn danh mục


| Mô hình | ndcg@10 lấy mẫu | ndcg@10 toàn danh mục | recall@10 toàn danh mục | AUC lấy mẫu (hòa = 0.5) | ứng viên −inf |
|---|---|---|---|---|---|
| B0-Random | 0.0538 [0.0415, 0.0673] | 0.0013 [0.0000, 0.0034] | 0.0033 [0.0000, 0.0082] | 0.4956 [0.4729, 0.5187] | 0.0% |
| B0-Popularity | 0.3603 [0.3317, 0.3891] | 0.0190 [0.0114, 0.0272] | 0.0410 [0.0262, 0.0574] | 0.8307 [0.8100, 0.8492] | 0.3% |
| B1-ItemKNN | 0.4744 [0.4422, 0.5053] | 0.0259 [0.0176, 0.0349] | 0.0574 [0.0393, 0.0754] | 0.8182 [0.7980, 0.8369] | 87.9% |
| B2-UserKNN | 0.4238 [0.3938, 0.4528] | 0.0244 [0.0164, 0.0333] | 0.0574 [0.0393, 0.0754] | 0.8366 [0.8174, 0.8535] | 71.9% |

#### 2.5: tách AUC (lấy mẫu): chấm được phim đúng không, và chấm rồi thì xếp đúng không

| Mô hình | phim đúng có điểm −inf | AUC khi phim đúng có điểm | AUC khi phim đúng −inf |
|---|---|---|---|
| B0-Popularity | 3.8% (23/610) | 0.8632 | 0.0011 |
| B1-ItemKNN | 29.0% (177/610) | 0.9736 | 0.4382 |
| B2-UserKNN | 17.5% (107/610) | 0.9364 | 0.3679 |

#### Chênh lệch ghép cặp: thứ tự có giữ khi bỏ lấy mẫu không?

| So sánh | độ đo | lấy mẫu | toàn danh mục |
|---|---|---|---|
| B1-ItemKNN − B0-Popularity | ndcg@10 | +0.1141 [+0.0914, +0.1348] **có ý nghĩa** | +0.0069 [-0.0033, +0.0171] nhiễu |
| B2-UserKNN − B0-Popularity | ndcg@10 | +0.0635 [+0.0452, +0.0826] **có ý nghĩa** | +0.0054 [-0.0011, +0.0122] nhiễu |
| B1-ItemKNN − B2-UserKNN | ndcg@10 | +0.0506 [+0.0283, +0.0720] **có ý nghĩa** | +0.0015 [-0.0082, +0.0109] nhiễu |
| B1-ItemKNN − B0-Popularity | recall@10 | +0.0721 [+0.0410, +0.1033] **có ý nghĩa** | +0.0164 [-0.0033, +0.0361] nhiễu |
| B2-UserKNN − B0-Popularity | recall@10 | +0.0475 [+0.0213, +0.0770] **có ý nghĩa** | +0.0164 [+0.0016, +0.0311] **có ý nghĩa** |
| B1-ItemKNN − B2-UserKNN | recall@10 | +0.0246 [-0.0049, +0.0557] nhiễu | +0.0000 [-0.0197, +0.0197] nhiễu |
| B1-ItemKNN − B0-Popularity | auc | -0.0124 [-0.0290, +0.0045] nhiễu | -0.0136 [-0.0300, +0.0033] nhiễu |
| B2-UserKNN − B0-Popularity | auc | +0.0060 [-0.0075, +0.0206] nhiễu | +0.0054 [-0.0077, +0.0196] nhiễu |
| B1-ItemKNN − B2-UserKNN | auc | -0.0184 [-0.0344, -0.0024] **có ý nghĩa** | -0.0190 [-0.0350, -0.0027] **có ý nghĩa** |

### 2.8: cold-start nhân tạo (cắt lịch sử train còn n tương tác gần nhất)


ndcg@10, CI 95%. Mỗi dòng đo trên đủ 610 user (4 phần, mỗi lần chỉ cắt một phần).

| giữ lại | B0-Popularity | B1-ItemKNN | B2-UserKNN | ItemKNN − Popularity | UserKNN − Popularity | UserKNN − ItemKNN |
|---|---|---|---|---|---|---|
| 1 tương tác | 0.3606 [0.3323, 0.3882] | 0.1563 [0.1305, 0.1846] | 0.0442 [0.0328, 0.0569] | -0.2043 [-0.2409, -0.1686] **có ý nghĩa** | -0.3164 [-0.3475, -0.2867] **có ý nghĩa** | -0.1121 [-0.1388, -0.0879] **có ý nghĩa** |
| 3 tương tác | 0.3603 [0.3320, 0.3881] | 0.2525 [0.2207, 0.2839] | 0.3518 [0.3226, 0.3796] | -0.1077 [-0.1446, -0.0724] **có ý nghĩa** | -0.0085 [-0.0350, +0.0172] nhiễu | +0.0992 [+0.0625, +0.1352] **có ý nghĩa** |
| 5 tương tác | 0.3602 [0.3318, 0.3880] | 0.3045 [0.2720, 0.3364] | 0.4005 [0.3709, 0.4291] | -0.0557 [-0.0912, -0.0219] **có ý nghĩa** | +0.0403 [+0.0186, +0.0622] **có ý nghĩa** | +0.0960 [+0.0634, +0.1302] **có ý nghĩa** |
| 10 tương tác | 0.3618 [0.3333, 0.3901] | 0.3714 [0.3360, 0.4041] | 0.4113 [0.3822, 0.4424] | +0.0096 [-0.0233, +0.0428] nhiễu | +0.0495 [+0.0309, +0.0693] **có ý nghĩa** | +0.0399 [+0.0099, +0.0713] **có ý nghĩa** |
| đầy đủ | 0.3603 [0.3317, 0.3891] | 0.4744 [0.4422, 0.5053] | 0.4238 [0.3938, 0.4528] | +0.1141 [+0.0914, +0.1348] **có ý nghĩa** | +0.0635 [+0.0452, +0.0826] **có ý nghĩa** | -0.0506 [-0.0720, -0.0283] **có ý nghĩa** |

Tỉ lệ ứng viên −inf (mô hình không có ý kiến):

| giữ lại | B0-Popularity | B1-ItemKNN | B2-UserKNN |
|---|---|---|---|
| 1 tương tác | 10.2% | 99.6% | 100.0% |
| 3 tương tác | 10.0% | 99.0% | 67.3% |
| 5 tương tác | 9.9% | 98.5% | 64.4% |
| 10 tương tác | 9.6% | 97.4% | 68.4% |
| đầy đủ | 0.3% | 87.9% | 71.9% |

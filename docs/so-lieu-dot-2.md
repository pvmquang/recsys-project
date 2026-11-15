# Số liệu đợt 2: toàn danh mục, AUC, cold-start

Checklist 2.4, 2.5, 2.8. Ba mục này không đổi giao thức chung: bảng chính vẫn đo bằng 1 phim
đúng + 99 phim âm. Chúng đo thêm để biết đọc bảng chính đến đâu thì dừng.

Số chạy với giao thức đã chốt ở 2.1-2.3: phá hòa timestamp theo seed, ItemKNN k = 80 và
UserKNN k = 40 (chọn trên validation). Phần "Kết luận" do người viết. Phần "Số liệu chi tiết"
là output nguyên văn của `python scripts/exp_giao_thuc.py`. CSV kèm seed và commit nằm ở
`reports/exp_2_*.csv` + `.meta.json`.

> **Quy tắc:** chênh lệch nào có CI 95% chứa 0 thì không viết "mô hình X thắng".

## Hai kết luận đổi so với bản trước

Bản trước chạy với k = 40 cho cả hai KNN và phá hòa theo `item_id`. Đổi hai điều đó thì:

| | bản trước | bản này |
|---|---|---|
| ItemKNN − Popularity, nDCG@10, toàn danh mục | +0.007, nhiễu | +0.014, có ý nghĩa |
| UserKNN − ItemKNN ở 10 tương tác | +0.040, có ý nghĩa | +0.013, nhiễu |
| ItemKNN − UserKNN, AUC | −0.018, có ý nghĩa | +0.003, nhiễu |

Cả ba đều là chênh lệch có CI sát 0, nên một thay đổi nhỏ của giao thức đủ lật kết luận.
Trong báo cáo, những câu dựa trên chúng phải viết ở dạng "chênh lệch nhỏ, sát ngưỡng ý nghĩa".

---

## Kết luận

### 2.4: Bỏ lấy mẫu thì chênh lệch nDCG co lại khoảng 8 lần

| | lấy mẫu 99 phim âm | toàn danh mục (~9.700 phim) |
|---|---|---|
| ItemKNN − Popularity, nDCG@10 | +0.117, có ý nghĩa | +0.014, có ý nghĩa |
| UserKNN − Popularity, nDCG@10 | +0.062, có ý nghĩa | +0.007, nhiễu |
| ItemKNN − UserKNN, nDCG@10 | +0.055, có ý nghĩa | +0.007, nhiễu |
| ItemKNN − Popularity, recall@10 | +0.084, có ý nghĩa | +0.023, có ý nghĩa |
| UserKNN − Popularity, recall@10 | +0.048, có ý nghĩa | +0.021, có ý nghĩa |

- Thứ tự trung bình giữ nguyên: ItemKNN > UserKNN > Popularity > Random. Không cặp nào đảo
  chiều có ý nghĩa.
- Trên toàn danh mục, phim đúng lọt top-10 ở 24 (Popularity), 38 (ItemKNN) và 37 (UserKNN)
  trên 610 user. Với ngần ấy lượt trúng, CI rộng: ItemKNN − Popularity có cận dưới +0.003.
- Câu "+33% nDCG" chỉ đúng với giao thức lấy mẫu. Báo cáo phải ghi kèm giao thức đó.
- Muốn tách mô hình trên toàn danh mục thì cần nhiều user hơn: ml-1m có 6.040 user.

### 2.5: AUC ở giao thức này đo việc có chấm được phim đúng hay không

- AUC tính từ điểm, cặp bằng điểm tính 0.5. Cả ba cặp so sánh AUC đều là nhiễu.
- Bảng "Tách AUC" cho thấy lý do. Khi phim đúng có điểm, AUC là 0.86-0.96. Khi phim đúng
  bị `-inf`, AUC tụt xuống 0.00-0.40. ItemKNN không chấm được 23% phim đúng, UserKNN 18,7%.
- Với k = 40, ItemKNN không chấm được 29% phim đúng và AUC thua UserKNN có ý nghĩa. Với
  k = 80 thì hết. AUC vì vậy phụ thuộc vào k nhiều hơn vào chất lượng xếp hạng.
- Báo cáo nên đưa hai số tách riêng (tỉ lệ `-inf` và AUC khi có điểm), không đưa AUC gộp.

### 2.8: Cold-start: mô hình tốt nhất đổi theo độ dài lịch sử

| lịch sử train | tốt nhất (nDCG@10) | UserKNN − ItemKNN |
|---|---|---|
| 1 tương tác | Popularity | −0.164, ItemKNN hơn |
| 3 tương tác | Popularity ≈ UserKNN | +0.054, UserKNN hơn |
| 5 tương tác | UserKNN | +0.053, UserKNN hơn |
| 10 tương tác | UserKNN ≈ ItemKNN | +0.013, nhiễu |
| đầy đủ (19-2.697) | ItemKNN | −0.055, ItemKNN hơn |

- ItemKNN cần lịch sử. Với 1-5 tương tác nó không có ý kiến về 97-99% ứng viên, nên thứ tự
  phần lớn do xáo ngẫu nhiên quyết định. Nó thua Popularity ở 1 và 3 tương tác, hòa ở 5,
  hơn ở 10.
- UserKNN với 1 tương tác không có láng giềng nào (100% ứng viên `-inf`): Pearson của một
  user chỉ có một rating là không xác định.
- Đây là căn cứ cho B7: lai ghép kiểu chuyển Popularity → UserKNN → ItemKNN theo số tương
  tác. Ngưỡng chuyển nên chọn trên validation, không lấy thẳng từ bảng này.
- Hạn chế: cắt 1/4 số user làm ~10% phim mất khỏi train (phim chỉ những user đó xem). nDCG
  của Popularity gần như không đổi (0.3590 so với 0.3600-0.3613), nên ảnh hưởng nhỏ.
- Giữ n tương tác gần nhất trước thời điểm test, không giữ n tương tác đầu tiên. Lý do ghi
  ở `split.truncate_history`.

---

## Số liệu chi tiết

ml-latest-small · seed 42 · k láng giềng: ItemKNN 80, UserKNN 40 (chọn trên validation) · 610 user kiểm tra

### 2.4 + 2.5: lấy mẫu 99 item âm so với toàn danh mục


| Mô hình | ndcg@10 lấy mẫu | ndcg@10 toàn danh mục | recall@10 toàn danh mục | AUC lấy mẫu (hòa = 0.5) | ứng viên −inf |
|---|---|---|---|---|---|
| B0-Random | 0.0538 [0.0415, 0.0673] | 0.0013 [0.0000, 0.0034] | 0.0033 [0.0000, 0.0082] | 0.4956 [0.4729, 0.5187] | 0.0% |
| B0-Popularity | 0.3590 [0.3302, 0.3879] | 0.0182 [0.0109, 0.0264] | 0.0393 [0.0246, 0.0557] | 0.8315 [0.8110, 0.8501] | 0.3% |
| B1-ItemKNN | 0.4761 [0.4436, 0.5064] | 0.0327 [0.0228, 0.0438] | 0.0623 [0.0443, 0.0820] | 0.8342 [0.8135, 0.8524] | 81.2% |
| B2-UserKNN | 0.4214 [0.3912, 0.4505] | 0.0257 [0.0174, 0.0349] | 0.0607 [0.0426, 0.0787] | 0.8310 [0.8113, 0.8480] | 71.8% |

#### 2.5: tách AUC (lấy mẫu): chấm được phim đúng không, và chấm rồi thì xếp đúng không

| Mô hình | phim đúng có điểm −inf | AUC khi phim đúng có điểm | AUC khi phim đúng −inf |
|---|---|---|---|
| B0-Popularity | 3.8% (23/610) | 0.8640 | 0.0011 |
| B1-ItemKNN | 23.0% (140/610) | 0.9624 | 0.4037 |
| B2-UserKNN | 18.7% (114/610) | 0.9374 | 0.3681 |

#### Chênh lệch ghép cặp: thứ tự có giữ khi bỏ lấy mẫu không?

| So sánh | độ đo | lấy mẫu | toàn danh mục |
|---|---|---|---|
| B1-ItemKNN − B0-Popularity | ndcg@10 | +0.1171 [+0.0970, +0.1388] **có ý nghĩa** | +0.0144 [+0.0026, +0.0262] **có ý nghĩa** |
| B2-UserKNN − B0-Popularity | ndcg@10 | +0.0624 [+0.0440, +0.0815] **có ý nghĩa** | +0.0074 [-0.0002, +0.0152] nhiễu |
| B1-ItemKNN − B2-UserKNN | ndcg@10 | +0.0547 [+0.0343, +0.0738] **có ý nghĩa** | +0.0070 [-0.0045, +0.0184] nhiễu |
| B1-ItemKNN − B0-Popularity | recall@10 | +0.0836 [+0.0541, +0.1131] **có ý nghĩa** | +0.0230 [+0.0016, +0.0443] **có ý nghĩa** |
| B2-UserKNN − B0-Popularity | recall@10 | +0.0475 [+0.0197, +0.0755] **có ý nghĩa** | +0.0213 [+0.0066, +0.0361] **có ý nghĩa** |
| B1-ItemKNN − B2-UserKNN | recall@10 | +0.0361 [+0.0066, +0.0672] **có ý nghĩa** | +0.0016 [-0.0197, +0.0230] nhiễu |
| B1-ItemKNN − B0-Popularity | auc | +0.0027 [-0.0129, +0.0183] nhiễu | +0.0016 [-0.0142, +0.0170] nhiễu |
| B2-UserKNN − B0-Popularity | auc | -0.0005 [-0.0136, +0.0139] nhiễu | -0.0011 [-0.0140, +0.0128] nhiễu |
| B1-ItemKNN − B2-UserKNN | auc | +0.0032 [-0.0128, +0.0193] nhiễu | +0.0027 [-0.0134, +0.0191] nhiễu |

### 2.8: cold-start nhân tạo (cắt lịch sử train còn n tương tác gần nhất)


ndcg@10, CI 95%. Mỗi dòng đo trên đủ 610 user (4 phần, mỗi lần chỉ cắt một phần).

| giữ lại | B0-Popularity | B1-ItemKNN | B2-UserKNN | ItemKNN − Popularity | UserKNN − Popularity | UserKNN − ItemKNN |
|---|---|---|---|---|---|---|
| 1 tương tác | 0.3610 [0.3326, 0.3891] | 0.2083 [0.1787, 0.2395] | 0.0442 [0.0328, 0.0569] | -0.1528 [-0.1876, -0.1169] **có ý nghĩa** | -0.3169 [-0.3476, -0.2861] **có ý nghĩa** | -0.1641 [-0.1940, -0.1356] **có ý nghĩa** |
| 3 tương tác | 0.3613 [0.3328, 0.3894] | 0.3051 [0.2741, 0.3379] | 0.3586 [0.3292, 0.3876] | -0.0562 [-0.0923, -0.0210] **có ý nghĩa** | -0.0027 [-0.0297, +0.0225] nhiễu | +0.0535 [+0.0186, +0.0874] **có ý nghĩa** |
| 5 tương tác | 0.3600 [0.3315, 0.3883] | 0.3520 [0.3181, 0.3856] | 0.4054 [0.3757, 0.4351] | -0.0080 [-0.0421, +0.0273] nhiễu | +0.0454 [+0.0224, +0.0679] **có ý nghĩa** | +0.0534 [+0.0206, +0.0859] **có ý nghĩa** |
| 10 tương tác | 0.3612 [0.3320, 0.3891] | 0.3995 [0.3668, 0.4321] | 0.4129 [0.3832, 0.4436] | +0.0383 [+0.0069, +0.0717] **có ý nghĩa** | +0.0517 [+0.0322, +0.0719] **có ý nghĩa** | +0.0134 [-0.0162, +0.0426] nhiễu |
| đầy đủ | 0.3590 [0.3302, 0.3879] | 0.4761 [0.4436, 0.5064] | 0.4214 [0.3912, 0.4505] | +0.1171 [+0.0970, +0.1388] **có ý nghĩa** | +0.0624 [+0.0440, +0.0815] **có ý nghĩa** | -0.0547 [-0.0738, -0.0343] **có ý nghĩa** |

Tỉ lệ ứng viên −inf (mô hình không có ý kiến):

| giữ lại | B0-Popularity | B1-ItemKNN | B2-UserKNN |
|---|---|---|---|
| 1 tương tác | 10.2% | 99.2% | 100.0% |
| 3 tương tác | 10.1% | 98.0% | 67.1% |
| 5 tương tác | 9.9% | 97.1% | 64.4% |
| 10 tương tác | 9.7% | 95.2% | 68.3% |
| đầy đủ | 0.3% | 81.2% | 71.8% |

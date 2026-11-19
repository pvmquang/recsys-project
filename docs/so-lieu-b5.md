# Số liệu B5: gợi ý theo nội dung

Bậc B5 cài trong `recsys/models/content.py`. Mỗi phim là một vector TF-IDF từ metadata: thể
loại, thập niên, đạo diễn, diễn viên, từ khóa. Hồ sơ user là tổng có trọng số các vector phim
đã xem. Điểm là cosine giữa hồ sơ và phim. Metadata TMDB tải ngày 19/11/2026, phủ 9.621/9.742
phim (98,8%).

Giao thức như B3, B4: phá hòa timestamp theo seed, siêu tham số chọn trên validation, báo cáo
trên test. Phần "Kết luận" do người viết. Phần "Số liệu chi tiết" là output nguyên văn của
`python scripts/exp_content.py`. CSV kèm seed và commit nằm ở `reports/exp_content_*.csv` và
`.meta.json`.

> **Quy tắc:** chênh lệch nào có CI 95% chứa 0 thì không viết "mô hình X thắng".

---

## Kết luận

### Siêu tham số chọn trên validation

Cấu hình chọn: thể loại, thập niên, đạo diễn, diễn viên, từ khóa; trọng số `binary`; half_life
3; trọng số thể loại 1, đạo diễn 4. nDCG@10 validation 0.2571. BPR được 0.4928.

- Giá trị này ghi trong `FIELDS`, `WEIGHTING`, `HALF_LIFE`, `GENRE_WEIGHT`, `DIRECTOR_WEIGHT`
  của `recsys/models/content.py`.
- Ưu tiên phim xem gần nhất giúp nhiều nhất. Cùng bộ đặc trưng TMDB, không giảm theo thời gian
  được 0.1660, half_life 3 được 0.2245.
- Trọng số theo rating (`rating`) kém `binary` ở cả 4 bộ đặc trưng. Khớp với bài toán dự đoán
  tương tác (checklist 2.2): phim xem rồi chấm thấp vẫn báo hiệu user sẽ xem phim giống vậy.
- Thêm tóm tắt phim (overview) làm giảm từ 0.2245 xuống 0.1796. Chỉ có thể loại và thập niên
  thì được 0.2035.
- Trọng số thể loại 1 nằm ở biên của lưới (1, 2, 4), chưa thử 0. Đạo diễn 2 và 4 gần như bằng
  nhau (0.2567 so với 0.2571). Chỉ có một lần chia dữ liệu.

### Khi lấy mẫu, B5 thua cả Popularity. Trên toàn danh mục, không phân biệt được

| B5 trừ | nDCG@10, lấy mẫu | nDCG@10, toàn danh mục |
|---|---|---|
| Popularity | −0.103 [−0.140, −0.066], có ý nghĩa | +0.009 [−0.005, +0.023], nhiễu |
| ItemKNN | −0.220 [−0.257, −0.181], có ý nghĩa | −0.006 [−0.021, +0.010], nhiễu |
| BPR | −0.232 [−0.268, −0.196], có ý nghĩa | −0.010 [−0.026, +0.006], nhiễu |

- Khi lấy mẫu, B5 được nDCG@10 0.256, thấp nhất bảng chính của README trừ Random. MF explicit
  (không có trong bảng chính) được 0.259.
- Trên toàn danh mục, chênh lệch nDCG@10 với cả ba mô hình đều có CI chứa 0. Recall@10 thua
  BPR −0.028 [−0.056, −0.002], có ý nghĩa nhưng sát ngưỡng.
- Lý do khoảng cách co lại: 99 phim âm lấy mẫu đều nên phần lớn là phim ít người xem (trung
  vị 3 lượt, so với 39 lượt của phim đúng, xem checklist 2.4). B5 không biết độ phổ biến và ưa
  phim ít người xem (mục tiếp theo), nên giao thức lấy mẫu phạt nó nặng nhất.
- Báo cáo không viết "B5 ngang lọc cộng tác". CI toàn danh mục rộng vì chỉ vài chục user trúng
  top-10, nên "không phân biệt được" là tất cả những gì số này cho phép nói.
- AUC của B5 thấp hơn Popularity −0.026 [−0.050, −0.002], có ý nghĩa nhưng sát ngưỡng. B5 chỉ
  để 0,6% ứng viên ở `-inf`, nên so AUC hợp lệ.

### B5 gợi ý phim ít người xem, phủ danh mục gấp 3 lần BPR

- Top-10 của B5 phủ 27,9% danh mục (2.705 phim). BPR phủ 9,1%, Popularity 1,3%.
- Lượt xem trung vị của phim B5 gợi ý là 7. Của BPR là 108, của Popularity là 218.
- Chỉ 2,7% phim B5 gợi ý thuộc top-100 phổ biến. BPR là 48,5%.
- Đây là số đếm, chưa có CI.

### Phim mới: lọc cộng tác về 0, B5 giữ nguyên

Thí nghiệm xóa mọi tương tác train với phim test của 1/4 số user, lần lượt 4 lần (31.584 dòng
bị xóa). Phim test khi đó là phim chưa ai xem.

- Popularity và BPR được nDCG@10 bằng 0. Hai mô hình này không có điểm cho phim không có
  trong train, nên phim đúng luôn ở cuối danh sách.
- B5 được nDCG@10 0.252, gần bằng 0.256 khi phim đã có người xem.
- Trong 23 user có phim test vốn không có trong train (checklist 2.6), B5 trúng top-10 cho 5
  user khi lấy mẫu và 1 user trên toàn danh mục. B0, B1, B4 trúng 0.
- Hệ quả cho B7: đây là căn cứ đo được cho lai ghép kiểu chuyển theo PHIM. Phim đã có tương tác
  dùng lọc cộng tác, phim chưa có dùng B5.
- `recommend()` chỉ duyệt phim có trong train (`all_items_` do `fit()` của hợp đồng gán), nên
  B5 hiện chưa đưa phim mới lên giao diện. Thí nghiệm trên đo bằng `score()`. B7 phải tự mở rộng
  danh mục ứng viên.

### User mới: B5 thua Popularity và BPR ở mọi mức lịch sử

| lịch sử | Popularity | BPR | B5 |
|---|---|---|---|
| 1 | 0.361 | **0.437** | 0.160 |
| 3 | 0.361 | **0.466** | 0.209 |
| 5 | 0.360 | **0.491** | 0.234 |
| 10 | 0.361 | **0.504** | 0.251 |
| đầy đủ | 0.359 | **0.489** | 0.256 |

nDCG@10, lấy mẫu, đủ 610 user. Mọi chênh lệch B5 trừ Popularity và B5 trừ BPR đều có ý nghĩa.

- Nhánh nội dung không giúp user mới. Với user mới, BPR đã hơn Popularity (số của B4).
- Khoảng cách B5 với Popularity co từ −0.201 ở 1 tương tác xuống −0.103 ở lịch sử đầy đủ.

### Nối tên người thành một token: +0.057 nDCG@10

Tách tên như mã nguồn tham khảo (`james cameron` thành hai token) làm nDCG@10 giảm từ 0.256 xuống
0.200. Chênh lệch +0.057 [+0.040, +0.074], có ý nghĩa. Đây là con số cho mục "Một sửa lỗi có chủ
ý" trong README. Cấu hình được chọn khi tên đã nối. Bản tách tên không dò lại siêu tham số.

### Câu giải thích là phân rã chính xác

Điểm của B5 tách đúng thành tổng đóng góp của từng đặc trưng (`shared_features()`) và của từng
phim đã xem (`contributions()`). `explain()` nêu ba đặc trưng và hai phim đóng góp nhiều nhất.

Ví dụ user 2 cho thấy hệ quả của trọng số đạo diễn 4. *Happy Feet Two* được gợi ý vì cùng đạo
diễn George Miller với *Mad Max: Fury Road*, dù hai phim khác thể loại. Câu giải thích làm lộ ra
điều này, nên dùng được làm ví dụ trên slide.

---

## Số liệu chi tiết

ml-latest-small · seed 42 · 610 user kiểm tra · ItemKNN k = 80 · BPR theo hằng số bpr.py

Metadata:

```
9,742 phim | co metadata TMDB: 9,621 (98.8%)
  khong co the loai : 34
  khong co dien vien: 150
  khong co dao dien : 125
```

### 1. Chọn siêu tham số trên validation (ndcg@10)

Trọng số thể loại = 2, đạo diễn = 2. Ô đậm là tốt nhất của từng hàng.

| đặc trưng | trọng số | half_life không | half_life 50 | half_life 20 | half_life 10 | half_life 5 | half_life 3 | half_life 1 |
|---|---|---|---|---|---|---|---|---|
| the_loai | binary | 0.0985 | 0.1101 | 0.1179 | 0.1310 | 0.1451 | 0.1483 | **0.1513** |
| the_loai | rating | 0.0909 | **0.0923** | 0.0876 | 0.0911 | 0.0908 | 0.0875 | 0.0876 |
| the_loai+nam | binary | 0.1535 | 0.1634 | 0.1716 | 0.1868 | 0.1993 | **0.2035** | 0.1845 |
| the_loai+nam | rating | 0.0908 | 0.0948 | 0.0955 | **0.0994** | 0.0949 | 0.0937 | 0.0919 |
| tmdb | binary | 0.1660 | 0.1876 | 0.2009 | 0.2089 | 0.2175 | **0.2245** | 0.2063 |
| tmdb | rating | **0.1543** | 0.1543 | 0.1448 | 0.1407 | 0.1358 | 0.1241 | 0.1141 |
| tmdb+tom_tat | binary | 0.1194 | 0.1316 | 0.1436 | 0.1581 | 0.1689 | **0.1796** | 0.1684 |
| tmdb+tom_tat | rating | **0.1436** | 0.1409 | 0.1316 | 0.1261 | 0.1216 | 0.1086 | 0.1052 |

Trọng số ở cấu hình tốt nhất (tmdb, binary, half_life 3):

| trọng số thể loại | trọng số đạo diễn | ndcg@10 |
|---|---|---|
| 1 | 1 | 0.2420 |
| 1 | 2 | 0.2567 |
| 1 | 4 | 0.2571 |
| 2 | 1 | 0.2042 |
| 2 | 2 | 0.2245 |
| 2 | 4 | 0.2298 |
| 4 | 1 | 0.1691 |
| 4 | 2 | 0.1799 |
| 4 | 4 | 0.2081 |

Chọn: tmdb {'fields': ('genres', 'year', 'directors', 'cast', 'keywords'), 'weighting': 'binary', 'half_life': 3.0, 'genre_weight': 1, 'director_weight': 4}, ndcg@10 validation 0.2571.

### 2. Bảng chính trên test (1 phim đúng + 99 phim âm, CI 95%)

| Mô hình | recall@10 | ndcg@10 | AUC | coverage@10 | ứng viên −inf |
|---|---|---|---|---|---|
| B0-Popularity | 0.5967 [0.5573, 0.6361] | 0.3590 [0.3302, 0.3879] | 0.8315 [0.8110, 0.8501] | 0.0126 | 0.3% |
| B1-ItemKNN | 0.6803 [0.6410, 0.7148] | 0.4761 [0.4436, 0.5064] | 0.8342 [0.8135, 0.8524] | 0.0749 | 81.2% |
| B4-BPR | 0.7328 [0.6984, 0.7672] | 0.4886 [0.4590, 0.5176] | 0.8858 [0.8678, 0.9020] | 0.0911 | 0.3% |
| B5-Content | 0.4869 [0.4475, 0.5279] | 0.2563 [0.2319, 0.2812] | 0.8052 [0.7888, 0.8222] | 0.2788 | 0.6% |

#### Chênh lệch ghép cặp: lấy mẫu và toàn danh mục

| So sánh | độ đo | lấy mẫu | toàn danh mục |
|---|---|---|---|
| B5-Content − B0-Popularity | ndcg@10 | -0.1027 [-0.1397, -0.0661] **có ý nghĩa** | +0.0086 [-0.0054, +0.0226] nhiễu |
| B5-Content − B1-ItemKNN | ndcg@10 | -0.2198 [-0.2572, -0.1810] **có ý nghĩa** | -0.0059 [-0.0214, +0.0097] nhiễu |
| B5-Content − B4-BPR | ndcg@10 | -0.2323 [-0.2678, -0.1958] **có ý nghĩa** | -0.0095 [-0.0259, +0.0064] nhiễu |
| B5-Content − B0-Popularity | recall@10 | -0.1098 [-0.1656, -0.0557] **có ý nghĩa** | +0.0033 [-0.0197, +0.0262] nhiễu |
| B5-Content − B1-ItemKNN | recall@10 | -0.1934 [-0.2443, -0.1426] **có ý nghĩa** | -0.0197 [-0.0443, +0.0049] nhiễu |
| B5-Content − B4-BPR | recall@10 | -0.2459 [-0.2951, -0.1984] **có ý nghĩa** | -0.0279 [-0.0557, -0.0016] **có ý nghĩa** |
| B5-Content − B0-Popularity | auc | -0.0263 [-0.0504, -0.0016] **có ý nghĩa** | -0.0288 [-0.0525, -0.0037] **có ý nghĩa** |
| B5-Content − B1-ItemKNN | auc | -0.0290 [-0.0523, -0.0052] **có ý nghĩa** | -0.0305 [-0.0542, -0.0063] **có ý nghĩa** |
| B5-Content − B4-BPR | auc | -0.0806 [-0.1021, -0.0589] **có ý nghĩa** | -0.0825 [-0.1040, -0.0606] **có ý nghĩa** |

Toàn danh mục, ndcg@10: B0-Popularity 0.0182 [0.0109, 0.0264], B1-ItemKNN 0.0327 [0.0228, 0.0438], B4-BPR 0.0363 [0.0252, 0.0481], B5-Content 0.0268 [0.0161, 0.0382]

#### 23 user có phim test chưa từng xuất hiện trong train (checklist 2.6)

Số user trúng top-10 trong nhóm này:

| Mô hình | lấy mẫu | toàn danh mục |
|---|---|---|
| B0-Popularity | 0/23 | 0/23 |
| B1-ItemKNN | 0/23 | 0/23 |
| B4-BPR | 0/23 | 0/23 |
| B5-Content | 5/23 | 1/23 |

### 3. Nối tên người thành một token hay tách như mã nguồn tham khảo

| cách | ndcg@10 | recall@10 |
|---|---|---|
| nối (`james_cameron`) | 0.2563 [0.2319, 0.2812] | 0.4869 [0.4475, 0.5279] |
| tách (`james cameron`) | 0.1995 [0.1765, 0.2250] | 0.3672 [0.3279, 0.4066] |

Nối − tách, ndcg@10: +0.0568 [+0.0402, +0.0736] **có ý nghĩa**

### 4. Thiên lệch phổ biến của top-10 (toàn danh mục, mọi user)

| Mô hình | số phim khác nhau | phủ danh mục | lượt xem TB | lượt xem trung vị | tỉ lệ thuộc top-100 phổ biến |
|---|---|---|---|---|---|
| B0-Popularity | 122 | 1.3% | 232.7 | 218 | 99.3% |
| B1-ItemKNN | 727 | 7.5% | 129.6 | 122 | 56.3% |
| B4-BPR | 884 | 9.1% | 121.7 | 108 | 48.5% |
| B5-Content | 2705 | 27.9% | 19.6 | 7 | 2.7% |

### 5. Phim mới: phim test bị xóa khỏi train (item cold-start)

Mỗi lần lấy 1/4 số user, xóa mọi tương tác train với phim test của họ (tổng 31,584 dòng qua 4 lần), rồi đo trên chính các user đó. Đủ 610 user, CI 95%.

| Mô hình | recall@10 | ndcg@10 | AUC |
|---|---|---|---|
| B0-Popularity | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0080 [0.0075, 0.0085] |
| B4-BPR | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0080 [0.0075, 0.0085] |
| B5-Content | 0.4836 [0.4443, 0.5230] | 0.2519 [0.2268, 0.2765] | 0.8050 [0.7874, 0.8222] |

B5 − Popularity, ndcg@10: +0.2519 [+0.2268, +0.2765] **có ý nghĩa**
B5 − BPR, ndcg@10: +0.2519 [+0.2268, +0.2765] **có ý nghĩa**

### 6. User mới: cắt còn n tương tác gần nhất (như checklist 2.8)

ndcg@10, CI 95%, đủ 610 user.

| giữ lại | B0-Popularity | B4-BPR | B5-Content | B5 − Popularity | B5 − BPR |
|---|---|---|---|---|---|
| 1 tương tác | 0.3610 [0.3326, 0.3891] | 0.4367 [0.4082, 0.4662] | 0.1603 [0.1379, 0.1832] | -0.2008 [-0.2370, -0.1636] **có ý nghĩa** | -0.2764 [-0.3105, -0.2421] **có ý nghĩa** |
| 3 tương tác | 0.3613 [0.3328, 0.3894] | 0.4656 [0.4359, 0.4946] | 0.2092 [0.1844, 0.2349] | -0.1521 [-0.1922, -0.1152] **có ý nghĩa** | -0.2564 [-0.2922, -0.2189] **có ý nghĩa** |
| 5 tương tác | 0.3600 [0.3315, 0.3883] | 0.4908 [0.4609, 0.5211] | 0.2335 [0.2093, 0.2584] | -0.1265 [-0.1651, -0.0908] **có ý nghĩa** | -0.2573 [-0.2948, -0.2204] **có ý nghĩa** |
| 10 tương tác | 0.3612 [0.3320, 0.3891] | 0.5035 [0.4743, 0.5330] | 0.2505 [0.2255, 0.2750] | -0.1107 [-0.1489, -0.0742] **có ý nghĩa** | -0.2530 [-0.2899, -0.2181] **có ý nghĩa** |
| đầy đủ | 0.3590 [0.3302, 0.3879] | 0.4886 [0.4590, 0.5176] | 0.2563 [0.2319, 0.2812] | -0.1027 [-0.1397, -0.0661] **có ý nghĩa** | -0.2323 [-0.2678, -0.1958] **có ý nghĩa** |

### 7. Ví dụ câu giải thích thật

- **User 1** → *Back to the Future Part II (1989)*: "Vi ban hay xem phim co: Robert Zemeckis, hoverboard, Thomas F. Wilson (giong nhat: Back to the Future Part III (1990), ¡Three Amigos! (1986))"
- **User 2** → *Happy Feet Two (2011)*: "Vi ban hay xem phim co: George Miller, Robin Williams, thap nien 2010 (giong nhat: Mad Max: Fury Road (2015), Good Will Hunting (1997))"
- **User 3** → *Scooby-Doo! Curse of the Lake Monster (2010)*: "Vi ban hay xem phim co: Brian Levant, Children, Adventure (giong nhat: Snow Dogs (2002), Rescuers, The (1977))"

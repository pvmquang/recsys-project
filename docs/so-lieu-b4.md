# Số liệu B4: BPR-MF

Bậc B4 cài trong `recsys/models/bpr.py`, tự viết bằng numpy. Dạng điểm giống MF (vector user,
vector phim, thêm bias của phim), nhưng mục tiêu học khác. MF implicit ở B3 khớp từng ô của
ma trận về 0 hoặc 1. BPR chỉ đòi phim user đã xem có điểm cao hơn phim chưa xem (Rendle và
cộng sự, UAI 2009). Học bằng SGD theo lô, mỗi lô 1.024 cặp, phim âm lấy đều trong các phim
user chưa xem.

Giao thức như B3: phá hòa timestamp theo seed, siêu tham số chọn trên validation, báo cáo trên
test. Phần "Kết luận" do người viết. Phần "Số liệu chi tiết" là output nguyên văn của
`python scripts/exp_bpr.py`. CSV kèm seed và commit nằm ở `reports/exp_bpr_*.csv` và
`.meta.json`.

> **Quy tắc:** chênh lệch nào có CI 95% chứa 0 thì không viết "mô hình X thắng".

---

## Kết luận

### Siêu tham số chọn trên validation

f = 64, lr = 0.1, reg = 0.01, 80 epoch, nDCG@10 validation 0.4928. MF implicit được 0.5118,
ItemKNN được 0.4768.

- Giá trị này ghi trong `FACTORS`, `LR`, `REG`, `N_EPOCHS` của `recsys/models/bpr.py`.
- f = 128 với 160 epoch được 0.4927, gần như bằng. Từ 80 epoch trở đi, đường cong của các
  cấu hình tốt gần như phẳng. Chọn cấu hình nhỏ hơn vì nhanh hơn.
- reg = 0.03 kém rõ ở cả hai lr. reg = 0.001 thì nDCG lên xuống giữa các mốc epoch.
- Cỡ lô (1.024) và cách khởi tạo không dò. Chỉ có một seed khởi tạo và một lần chia dữ liệu.

### Trên test, BPR và MF hòa về nDCG, BPR hơn về AUC

| BPR trừ | nDCG@10, lấy mẫu | nDCG@10, toàn danh mục | AUC, lấy mẫu |
|---|---|---|---|
| MF implicit | −0.012 [−0.029, +0.007], nhiễu | +0.002 [−0.009, +0.014], nhiễu | +0.025 [+0.014, +0.037], có ý nghĩa |
| ItemKNN | +0.013 [−0.007, +0.033], nhiễu | +0.004 [−0.008, +0.017], nhiễu | (không so được, xem dưới) |
| Popularity | +0.130 [+0.108, +0.152], có ý nghĩa | +0.018 [+0.006, +0.030], có ý nghĩa | +0.054 [+0.045, +0.064], có ý nghĩa |

- Về nDCG@10 và recall@10, BPR không hơn cũng không kém MF implicit có ý nghĩa, ở cả hai giao
  thức. Báo cáo viết "hòa", không viết bên nào thắng.
- BPR hơn MF về AUC (0.886 so với 0.861). Điều này khớp lý thuyết: hàm mục tiêu của BPR là dạng
  liên tục của AUC. So sánh này hợp lệ vì cả hai mô hình chấm được gần hết ứng viên (0,3% ở
  `-inf`).
- Bảng còn ghi BPR hơn ItemKNN về AUC (+0.052). Không dùng con số này: ItemKNN để 81% ứng
  viên ở `-inf`, nên AUC của nó chủ yếu đo việc có chấm được phim đúng hay không (xem
  `docs/so-lieu-dot-2.md`, mục 2.5).
- BPR hơn ItemKNN về recall@10 khi lấy mẫu (+0.053, có ý nghĩa), nhưng trên toàn danh mục thì
  là nhiễu.
- Hơn bậc trước một chút trên AUC mà nDCG không đổi: đây là câu trả lời cho câu hỏi "B4 cải
  thiện được gì so với B3". Đổi lại, BPR cần chọn thêm số epoch và tốc độ học, và câu giải
  thích của nó kém chính xác hơn (mục cuối).

### Bias của BPR gần như chính là độ phổ biến, nhưng top-10 không thiên về phim phổ biến

- Spearman giữa bias b_i của BPR và số lượt xem trong train là 0.949. BPR tự học ra một cột
  Popularity bên trong mô hình.
- Dù vậy, top-10 của BPR có độ phổ biến tương đương MF: lượt xem trung bình 121.7 so với 124.5,
  và 48,5% thuộc top-100 phim phổ biến so với 48,3%. Phần tích vô hướng p_u · q_i đủ lớn để
  cá nhân hóa.
- Phủ danh mục: MF 10,1%, BPR 9,1%, ItemKNN 7,5%, Popularity 1,3%. Đây là số đếm, chưa có CI.

### Cold-start: BPR là mô hình đầu tiên hơn Popularity ở 1 tương tác

nDCG@10, đủ 610 user, cắt lịch sử train của từng 1/4 số user:

| lịch sử | Popularity | ItemKNN | MF implicit | BPR |
|---|---|---|---|---|
| 1 | 0.361 | 0.208 | 0.390 | **0.437** |
| 3 | 0.361 | 0.305 | **0.454** | **0.466** |
| 5 | 0.360 | 0.352 | **0.479** | **0.491** |
| 10 | 0.361 | 0.400 | **0.500** | **0.504** |
| đầy đủ | 0.359 | 0.476 | **0.500** | **0.489** |

Ô in đậm là tốt nhất hoặc hòa với tốt nhất (CI ghép cặp chứa 0). Ở hàng "đầy đủ", ItemKNN
thua MF có ý nghĩa (số của B3) nên không in đậm.

- Với 1 tương tác, BPR hơn MF +0.047 [+0.021, +0.073] và hơn Popularity +0.076 [+0.051,
  +0.101], cả hai có ý nghĩa. Ở B3, MF chỉ hòa Popularity ở mức này.
- Một cách giải thích (chưa kiểm chứng riêng): bias b_i là độ phổ biến, nên khi user có quá
  ít lịch sử, điểm của BPR tự lùi về Popularity, cộng thêm chút cá nhân hóa từ một phim đã xem.
- Từ 3 tương tác trở lên, BPR và MF hòa.
- Hệ quả cho B7: lập luận "chuyển sang Popularity khi user mới" ở đợt 2 không còn đứng được.
  Với BPR, một mô hình đơn đã hơn Popularity ở mọi mức lịch sử đo được. B7 cần căn cứ khác,
  ví dụ phim mới chưa ai xem (lọc cộng tác nào cũng chịu, cần nhánh nội dung B5).

### Câu giải thích của BPR là hậu kiểm

Ở MF implicit, điểm tách đúng thành tổng đóng góp của từng phim đã xem. Ở BPR, vector user là
tham số tự do học bằng SGD, không có đẳng thức đó. `explain()` của BPR nêu ba phim đã xem có
vector gần vector phim gợi ý nhất (cosine). Câu này nói "phim này giống những phim bạn đã xem",
không nói "điểm này đến từ đâu".

Ở cả ba ví dụ cuối file, hai cách nêu trùng nhau ít nhất một phim, user 3 trùng cả ba. Dù vậy
chỉ cách của MF là phân rã chính xác.

---

## Số liệu chi tiết

ml-latest-small · seed 42 · 610 user kiểm tra · ItemKNN k = 80 · MF theo hằng số mf.py

### 1. Chọn siêu tham số trên validation (ndcg@10)

Mỗi cấu hình huấn luyện một lần, đánh giá sau 10, 20, 40, 80, 160 epoch. Ô đậm là tốt nhất của từng cấu hình.

| factors | lr | reg | 10 epoch | 20 epoch | 40 epoch | 80 epoch | 160 epoch |
|---|---|---|---|---|---|---|---|
| 64 | 0.05 | 0.001 | 0.4124 | 0.4553 | **0.4805** | 0.4735 | 0.4787 |
| 64 | 0.05 | 0.01 | 0.4116 | 0.4583 | 0.4767 | 0.4884 | **0.4911** |
| 64 | 0.05 | 0.03 | 0.4013 | 0.4324 | 0.4542 | 0.4635 | **0.4673** |
| 64 | 0.1 | 0.001 | 0.4456 | 0.4697 | 0.4703 | **0.4826** | 0.4804 |
| 64 | 0.1 | 0.01 | 0.4451 | 0.4737 | 0.4772 | **0.4928** | 0.4918 |
| 64 | 0.1 | 0.03 | 0.4264 | 0.4540 | 0.4498 | 0.4490 | **0.4594** |
| 32 | 0.1 | 0.01 | 0.4354 | 0.4703 | 0.4756 | **0.4810** | 0.4723 |
| 128 | 0.1 | 0.01 | 0.4425 | 0.4674 | 0.4854 | 0.4906 | **0.4927** |

Chọn: {'factors': 64, 'lr': 0.1, 'reg': 0.01, 'n_epochs': 80}, ndcg@10 validation 0.4928.

### 2. Bảng chính trên test (1 phim đúng + 99 phim âm, CI 95%)

| Mô hình | recall@10 | ndcg@10 | AUC | coverage@10 | ứng viên −inf |
|---|---|---|---|---|---|
| B0-Popularity | 0.5967 [0.5573, 0.6361] | 0.3590 [0.3302, 0.3879] | 0.8315 [0.8110, 0.8501] | 0.0126 | 0.3% |
| B1-ItemKNN | 0.6803 [0.6410, 0.7148] | 0.4761 [0.4436, 0.5064] | 0.8342 [0.8135, 0.8524] | 0.0749 | 81.2% |
| B3-MF-implicit | 0.7279 [0.6918, 0.7639] | 0.5003 [0.4701, 0.5299] | 0.8605 [0.8386, 0.8802] | 0.1006 | 0.3% |
| B4-BPR | 0.7328 [0.6984, 0.7672] | 0.4886 [0.4590, 0.5176] | 0.8858 [0.8678, 0.9020] | 0.0911 | 0.3% |

#### Chênh lệch ghép cặp: lấy mẫu và toàn danh mục

| So sánh | độ đo | lấy mẫu | toàn danh mục |
|---|---|---|---|
| B4-BPR − B3-MF-implicit | ndcg@10 | -0.0117 [-0.0294, +0.0065] nhiễu | +0.0023 [-0.0087, +0.0136] nhiễu |
| B4-BPR − B1-ItemKNN | ndcg@10 | +0.0126 [-0.0065, +0.0327] nhiễu | +0.0036 [-0.0082, +0.0165] nhiễu |
| B4-BPR − B0-Popularity | ndcg@10 | +0.1297 [+0.1077, +0.1518] **có ý nghĩa** | +0.0181 [+0.0063, +0.0301] **có ý nghĩa** |
| B4-BPR − B3-MF-implicit | recall@10 | +0.0049 [-0.0213, +0.0328] nhiễu | -0.0016 [-0.0230, +0.0197] nhiễu |
| B4-BPR − B1-ItemKNN | recall@10 | +0.0525 [+0.0246, +0.0820] **có ý nghĩa** | +0.0082 [-0.0115, +0.0295] nhiễu |
| B4-BPR − B0-Popularity | recall@10 | +0.1361 [+0.1033, +0.1689] **có ý nghĩa** | +0.0311 [+0.0098, +0.0541] **có ý nghĩa** |
| B4-BPR − B3-MF-implicit | auc | +0.0253 [+0.0144, +0.0372] **có ý nghĩa** | +0.0263 [+0.0157, +0.0379] **có ý nghĩa** |
| B4-BPR − B1-ItemKNN | auc | +0.0516 [+0.0366, +0.0672] **có ý nghĩa** | +0.0521 [+0.0372, +0.0676] **có ý nghĩa** |
| B4-BPR − B0-Popularity | auc | +0.0543 [+0.0453, +0.0639] **có ý nghĩa** | +0.0537 [+0.0450, +0.0633] **có ý nghĩa** |

Toàn danh mục, ndcg@10: B0-Popularity 0.0182 [0.0109, 0.0264], B1-ItemKNN 0.0327 [0.0228, 0.0438], B3-MF-implicit 0.0340 [0.0244, 0.0454], B4-BPR 0.0363 [0.0252, 0.0481]

### 3. Thiên lệch phổ biến của top-10 (toàn danh mục, mọi user)

| Mô hình | số phim khác nhau trong top-10 | phủ danh mục | lượt xem TB của phim gợi ý | tỉ lệ thuộc top-100 phổ biến |
|---|---|---|---|---|
| B0-Popularity | 122 | 1.3% | 232.7 | 99.3% |
| B1-ItemKNN | 727 | 7.5% | 129.6 | 56.3% |
| B3-MF-implicit | 976 | 10.1% | 124.5 | 48.3% |
| B4-BPR | 884 | 9.1% | 121.7 | 48.5% |

Spearman giữa bias b_i của BPR và số lượt xem trong train: 0.949 (9701 phim).

### 4. Cold-start nhân tạo (cắt còn n tương tác gần nhất)

ndcg@10, CI 95%, đủ 610 user.

| giữ lại | B0-Popularity | B1-ItemKNN | B3-MF-implicit | B4-BPR | BPR − MF | BPR − Popularity |
|---|---|---|---|---|---|---|
| 1 tương tác | 0.3610 [0.3326, 0.3891] | 0.2083 [0.1787, 0.2395] | 0.3898 [0.3601, 0.4207] | 0.4367 [0.4082, 0.4662] | +0.0469 [+0.0210, +0.0729] **có ý nghĩa** | +0.0756 [+0.0510, +0.1011] **có ý nghĩa** |
| 3 tương tác | 0.3613 [0.3328, 0.3894] | 0.3051 [0.2741, 0.3379] | 0.4543 [0.4243, 0.4843] | 0.4656 [0.4359, 0.4946] | +0.0113 [-0.0115, +0.0342] nhiễu | +0.1043 [+0.0772, +0.1303] **có ý nghĩa** |
| 5 tương tác | 0.3600 [0.3315, 0.3883] | 0.3520 [0.3181, 0.3856] | 0.4785 [0.4472, 0.5097] | 0.4908 [0.4609, 0.5211] | +0.0123 [-0.0095, +0.0346] nhiễu | +0.1308 [+0.1037, +0.1596] **có ý nghĩa** |
| 10 tương tác | 0.3612 [0.3320, 0.3891] | 0.3995 [0.3668, 0.4321] | 0.5001 [0.4694, 0.5308] | 0.5035 [0.4743, 0.5330] | +0.0034 [-0.0173, +0.0240] nhiễu | +0.1423 [+0.1170, +0.1690] **có ý nghĩa** |
| đầy đủ | 0.3590 [0.3302, 0.3879] | 0.4761 [0.4436, 0.5064] | 0.5003 [0.4701, 0.5299] | 0.4886 [0.4590, 0.5176] | -0.0117 [-0.0294, +0.0065] nhiễu | +0.1297 [+0.1077, +0.1518] **có ý nghĩa** |

### 5. Ví dụ câu giải thích thật

- **User 1** → *Die Hard (1988)*
  - BPR: "Giong cac phim ban da xem: Indiana Jones and the Last Crusade (1989), Star Wars: Episode V - The Empire Strikes Back (1980), Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)"
  - MF: "Vi ban da xem Star Wars: Episode IV - A New Hope (1977), Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981), Star Wars: Episode VI - Return of the Jedi (1983)"
- **User 2** → *Social Network, The (2010)*
  - BPR: "Giong cac phim ban da xem: Dark Knight, The (2008), Inception (2010), Inglourious Basterds (2009)"
  - MF: "Vi ban da xem Inglourious Basterds (2009), Inception (2010), Girl with the Dragon Tattoo, The (2011)"
- **User 3** → *Platoon (1986)*
  - BPR: "Giong cac phim ban da xem: Road Warrior, The (Mad Max 2) (1981), Deer Hunter, The (1978), Thing, The (1982)"
  - MF: "Vi ban da xem Deer Hunter, The (1978), Road Warrior, The (Mad Max 2) (1981), Thing, The (1982)"

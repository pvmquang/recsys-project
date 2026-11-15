# Số liệu B3: phân rã ma trận (ALS)

Bậc B3 cài trong `recsys/models/mf.py`, tự viết bằng numpy, không thêm thư viện. Có hai chế độ:
`implicit` học việc user có xem hay không (Hu, Koren & Volinsky 2008), `explicit` học rating có
bias (Koren, Bell & Volinsky 2009). Chế độ mặc định của `MF()` là implicit.

Giao thức như đợt 2: phá hòa timestamp theo seed, siêu tham số chọn trên validation (tương tác
áp chót), báo cáo trên test. Phần "Kết luận" do người viết. Phần "Số liệu chi tiết" là output
nguyên văn của `python scripts/exp_mf.py`. CSV kèm seed và commit nằm ở `reports/exp_mf_*.csv`
và `.meta.json`.

> **Quy tắc:** chênh lệch nào có CI 95% chứa 0 thì không viết "mô hình X thắng".

---

## Kết luận

### Siêu tham số chọn trên validation

| chế độ | factors | reg | alpha | nDCG@10 validation |
|---|---|---|---|---|
| implicit | 32 | 10 | 10 | 0.5118 |
| explicit | 32 | 30 | | 0.2705 |
| tham chiếu: ItemKNN k = 80 | | | | 0.4768 |

- Các giá trị này ghi trong `FACTORS`, `REG`, `ALPHA`, `FACTORS_EXPLICIT`, `REG_EXPLICIT` của
  `recsys/models/mf.py`. Chạy lại mà validation chọn khác thì `exp_mf.py` in cảnh báo.
- Implicit: tăng số nhân tố quá 32 làm giảm điểm (f = 64 được 0.5003, f = 128 được 0.4782).
  Dữ liệu chỉ có 610 user nên nhiều nhân tố hơn thì học thuộc train.
- Lưới tìm thô. Cặp reg = 100, alpha = 40 được 0.5050, gần với cặp tốt nhất. Không nên
  coi reg = 10, alpha = 10 là điểm tối ưu chính xác.
- Explicit gần như không đổi theo số nhân tố (0.2693 đến 0.2705). Điểm xếp hạng của nó chủ
  yếu đến từ phần bias, không đến từ nhân tố.
- Số vòng lặp ALS (15) không dò. Chỉ có một seed khởi tạo và một lần chia dữ liệu.
- Cột "giây fit" của f = 128 ghi 5.797 giây. Số này sai do máy bị treo hoặc ngủ giữa lúc
  chạy. Đo riêng trước đó, f = 128 mất khoảng 90 giây. Đừng dùng cột này để so chi phí.

### Trên test, MF implicit hơn ItemKNN, nhưng chỉ với giao thức lấy mẫu

| MF implicit trừ | nDCG@10, lấy mẫu 99 phim âm | nDCG@10, toàn danh mục |
|---|---|---|
| ItemKNN | +0.024 [+0.002, +0.046], có ý nghĩa | +0.001 [−0.009, +0.013], nhiễu |
| UserKNN | +0.079 [+0.055, +0.101], có ý nghĩa | +0.008 [−0.001, +0.019], nhiễu |
| Popularity | +0.141 [+0.117, +0.166], có ý nghĩa | +0.016 [+0.005, +0.028], có ý nghĩa |

- MF implicit có nDCG@10 cao nhất bảng: 0.5003, so với 0.4761 của ItemKNN.
- Chênh lệch với ItemKNN có cận dưới +0.002, rất sát 0. Bỏ lấy mẫu thì chênh lệch này thành
  nhiễu. Báo cáo nên viết "MF nhỉnh hơn ItemKNN một chút, sát ngưỡng ý nghĩa, và hòa khi
  xếp trên toàn danh mục".
- Trên toàn danh mục, chỉ còn hai kết luận có ý nghĩa: MF implicit hơn Popularity, và hơn
  MF explicit.
- coverage@10 của MF implicit là 10,1%, cao hơn ItemKNN (7,5%) và UserKNN (2,9%). Ở B1/B2,
  mô hình chính xác hơn luôn phủ ít hơn. MF là mô hình đầu tiên không theo đánh đổi đó.
  Coverage là một con số, chưa có CI.
- AUC của MF là 0.8605, so với khoảng 0.83 của B0 đến B2. Không so trực tiếp được: KNN để
  71-81% ứng viên ở `-inf`, còn MF chấm được mọi phim có trong train (chỉ 0,3% ứng viên là
  `-inf`). Xem giải thích AUC ở `docs/so-lieu-dot-2.md`, mục 2.5.

### MF explicit thua cả Popularity khi xếp hạng

- Lấy mẫu: nDCG@10 thấp hơn Popularity 0.100 [0.073, 0.126], có ý nghĩa. Toàn danh mục:
  +0.000, nhiễu.
- Cùng bài học với chế độ `weighted` của KNN. Explicit chỉ nhìn các ô đã đánh giá, nên không
  biết phim nào được nhiều người xem. Một phim ít người xem mà được chấm 5 sao sẽ lên đầu.
- Explicit vẫn có ích nếu đề bài cần dự đoán rating (MAE, RMSE). Với top-K thì dùng implicit.

### Cold-start: MF hơn hai KNN ở mọi độ dài lịch sử, hòa Popularity ở 1 tương tác

nDCG@10, đủ 610 user, cắt lịch sử train của từng 1/4 số user:

| lịch sử | Popularity | ItemKNN | UserKNN | MF implicit |
|---|---|---|---|---|
| 1 | **0.361** | 0.208 | 0.044 | **0.390** |
| 3 | 0.361 | 0.305 | 0.359 | **0.454** |
| 5 | 0.360 | 0.352 | 0.405 | **0.479** |
| 10 | 0.361 | 0.400 | 0.413 | **0.500** |
| đầy đủ | 0.359 | 0.476 | 0.421 | **0.500** |

Ô in đậm là tốt nhất hoặc hòa với tốt nhất (CI ghép cặp chứa 0).

- MF hơn ItemKNN và UserKNN có ý nghĩa ở mọi mức, kể cả 1 tương tác.
- Với 1 tương tác, MF − Popularity là +0.029 [−0.002, +0.062], nhiễu. Từ 3 tương tác trở lên
  MF hơn Popularity có ý nghĩa.
- Điều này đổi lập luận cho lai ghép kiểu chuyển ở B7. Ở đợt 2, căn cứ là "Popularity tốt
  nhất ở 1 tương tác, UserKNN tốt nhất ở 3-5". Có MF thì một mô hình đã không thua mô hình
  nào ở mọi mức. B7 cần một căn cứ khác, hoặc đo xem chuyển sang Popularity ở 1 tương tác
  có giúp gì không.

### Câu giải thích của MF là phân rã chính xác của điểm

Với implicit ALS, vector user là một tổ hợp tuyến tính của vector các phim user đã xem. Vì vậy
điểm của một phim gợi ý tách được đúng thành tổng đóng góp của từng phim đã xem, không xấp xỉ
(`MF.contributions()`, có test chốt). `explain()` nêu ba phim đóng góp nhiều nhất.

ItemKNN thì nêu các láng giềng gần nhất. Hai cách cho ra danh sách phim khác nhau cho cùng một
gợi ý (xem ví dụ ở cuối). Báo cáo có thể dùng điều này cho chuẩn đầu ra M2.

---

## Số liệu chi tiết

ml-latest-small · seed 42 · 610 user kiểm tra · ItemKNN k = 80, UserKNN k = 40

### 1. Chọn siêu tham số trên validation (ndcg@10)

| kind | factors | reg | alpha | ndcg@10 val | recall@10 val | giây fit |
|---|---|---|---|---|---|---|
| implicit | 32 | 1 | 1 | 0.4938 | 0.7213 | 6.4 |
| implicit | 32 | 1 | 10 | 0.5064 | 0.7459 | 8.3 |
| implicit | 32 | 1 | 40 | 0.4839 | 0.7377 | 7.9 |
| implicit | 32 | 10 | 1 | 0.4991 | 0.7328 | 7.8 |
| implicit | 32 | 10 | 10 | 0.5118 | 0.7492 | 8.1 |
| implicit | 32 | 10 | 40 | 0.4817 | 0.7344 | 8.5 |
| implicit | 32 | 100 | 1 | 0.3547 | 0.6131 | 9 |
| implicit | 32 | 100 | 10 | 0.4866 | 0.7410 | 9.7 |
| implicit | 32 | 100 | 40 | 0.5050 | 0.7672 | 8.3 |
| implicit | 16 | 10 | 10 | 0.4904 | 0.7590 | 4.4 |
| implicit | 64 | 10 | 10 | 0.5003 | 0.7246 | 21.3 |
| implicit | 128 | 10 | 10 | 0.4782 | 0.6869 | 5796.6 |
| explicit | 32 | 1 | 0 | 0.2048 | 0.3525 | 4.9 |
| explicit | 32 | 10 | 0 | 0.2550 | 0.4131 | 4.8 |
| explicit | 32 | 30 | 0 | 0.2705 | 0.4279 | 4.8 |
| explicit | 16 | 30 | 0 | 0.2693 | 0.4279 | 2.2 |
| explicit | 64 | 30 | 0 | 0.2697 | 0.4279 | 11 |

Chọn: implicit {'factors': 32, 'reg': 10.0, 'alpha': 10.0}, explicit {'factors': 32, 'reg': 30.0}.

### 2. Bảng chính trên test (1 phim đúng + 99 phim âm, CI 95%)

| Mô hình | recall@10 | ndcg@10 | AUC | coverage@10 | ứng viên −inf |
|---|---|---|---|---|---|
| B0-Popularity | 0.5967 [0.5573, 0.6361] | 0.3590 [0.3302, 0.3879] | 0.8315 [0.8110, 0.8501] | 0.0126 | 0.3% |
| B1-ItemKNN | 0.6803 [0.6410, 0.7148] | 0.4761 [0.4436, 0.5064] | 0.8342 [0.8135, 0.8524] | 0.0749 | 81.2% |
| B2-UserKNN | 0.6443 [0.6066, 0.6803] | 0.4214 [0.3912, 0.4505] | 0.8310 [0.8113, 0.8480] | 0.0288 | 71.8% |
| B3-MF-implicit | 0.7279 [0.6918, 0.7639] | 0.5003 [0.4701, 0.5299] | 0.8605 [0.8386, 0.8802] | 0.1006 | 0.3% |
| B3-MF-explicit | 0.4115 [0.3721, 0.4508] | 0.2592 [0.2314, 0.2874] | 0.6352 [0.6047, 0.6647] | 0.0105 | 0.3% |

#### Chênh lệch ghép cặp: lấy mẫu và toàn danh mục

| So sánh | độ đo | lấy mẫu | toàn danh mục |
|---|---|---|---|
| B3-MF-implicit − B1-ItemKNN | ndcg@10 | +0.0242 [+0.0024, +0.0460] **có ý nghĩa** | +0.0013 [-0.0089, +0.0128] nhiễu |
| B3-MF-implicit − B2-UserKNN | ndcg@10 | +0.0789 [+0.0554, +0.1014] **có ý nghĩa** | +0.0083 [-0.0014, +0.0191] nhiễu |
| B3-MF-implicit − B0-Popularity | ndcg@10 | +0.1413 [+0.1170, +0.1656] **có ý nghĩa** | +0.0158 [+0.0054, +0.0277] **có ý nghĩa** |
| B3-MF-explicit − B0-Popularity | ndcg@10 | -0.0998 [-0.1258, -0.0733] **có ý nghĩa** | +0.0003 [-0.0083, +0.0093] nhiễu |
| B3-MF-implicit − B3-MF-explicit | ndcg@10 | +0.2411 [+0.2085, +0.2726] **có ý nghĩa** | +0.0155 [+0.0024, +0.0292] **có ý nghĩa** |
| B3-MF-implicit − B1-ItemKNN | recall@10 | +0.0475 [+0.0148, +0.0820] **có ý nghĩa** | +0.0098 [-0.0115, +0.0328] nhiễu |
| B3-MF-implicit − B2-UserKNN | recall@10 | +0.0836 [+0.0459, +0.1197] **có ý nghĩa** | +0.0115 [-0.0082, +0.0328] nhiễu |
| B3-MF-implicit − B0-Popularity | recall@10 | +0.1311 [+0.0902, +0.1721] **có ý nghĩa** | +0.0328 [+0.0115, +0.0557] **có ý nghĩa** |
| B3-MF-explicit − B0-Popularity | recall@10 | -0.1852 [-0.2279, -0.1426] **có ý nghĩa** | -0.0066 [-0.0230, +0.0082] nhiễu |
| B3-MF-implicit − B3-MF-explicit | recall@10 | +0.3164 [+0.2721, +0.3607] **có ý nghĩa** | +0.0393 [+0.0164, +0.0623] **có ý nghĩa** |

Toàn danh mục, ndcg@10: B0-Popularity 0.0182 [0.0109, 0.0264], B1-ItemKNN 0.0327 [0.0228, 0.0438], B2-UserKNN 0.0257 [0.0174, 0.0349], B3-MF-implicit 0.0340 [0.0244, 0.0454], B3-MF-explicit 0.0185 [0.0105, 0.0281]

### 3. Cold-start nhân tạo (cắt còn n tương tác gần nhất)

ndcg@10, CI 95%, đủ 610 user.

| giữ lại | B0-Popularity | B1-ItemKNN | B2-UserKNN | B3-MF-implicit | MF − ItemKNN | MF − UserKNN | MF − Popularity |
|---|---|---|---|---|---|---|---|
| 1 tương tác | 0.3610 [0.3326, 0.3891] | 0.2083 [0.1787, 0.2395] | 0.0442 [0.0328, 0.0569] | 0.3898 [0.3601, 0.4207] | +0.1815 [+0.1520, +0.2114] **có ý nghĩa** | +0.3456 [+0.3141, +0.3786] **có ý nghĩa** | +0.0287 [-0.0022, +0.0623] nhiễu |
| 3 tương tác | 0.3613 [0.3328, 0.3894] | 0.3051 [0.2741, 0.3379] | 0.3586 [0.3292, 0.3876] | 0.4543 [0.4243, 0.4843] | +0.1493 [+0.1185, +0.1793] **có ý nghĩa** | +0.0958 [+0.0655, +0.1265] **có ý nghĩa** | +0.0930 [+0.0618, +0.1246] **có ý nghĩa** |
| 5 tương tác | 0.3600 [0.3315, 0.3883] | 0.3520 [0.3181, 0.3856] | 0.4054 [0.3757, 0.4351] | 0.4785 [0.4472, 0.5097] | +0.1265 [+0.0970, +0.1545] **có ý nghĩa** | +0.0731 [+0.0451, +0.1003] **có ý nghĩa** | +0.1185 [+0.0868, +0.1506] **có ý nghĩa** |
| 10 tương tác | 0.3612 [0.3320, 0.3891] | 0.3995 [0.3668, 0.4321] | 0.4129 [0.3832, 0.4436] | 0.5001 [0.4694, 0.5308] | +0.1007 [+0.0752, +0.1269] **có ý nghĩa** | +0.0873 [+0.0592, +0.1144] **có ý nghĩa** | +0.1390 [+0.1093, +0.1704] **có ý nghĩa** |
| đầy đủ | 0.3590 [0.3302, 0.3879] | 0.4761 [0.4436, 0.5064] | 0.4214 [0.3912, 0.4505] | 0.5003 [0.4701, 0.5299] | +0.0242 [+0.0024, +0.0460] **có ý nghĩa** | +0.0789 [+0.0554, +0.1014] **có ý nghĩa** | +0.1413 [+0.1170, +0.1656] **có ý nghĩa** |

### 4. Ví dụ câu giải thích thật

- **User 1** → *Army of Darkness (1993)*
  - MF: "Vi ban da xem Highlander (1986), Labyrinth (1986), Stargate (1994)"
  - ItemKNN: "Vi ban da danh gia 30 phim tuong tu (gan nhat: RoboCop (1987), Labyrinth (1986), Alien (1979))"
- **User 2** → *Fight Club (1999)*
  - MF: "Vi ban da xem Shawshank Redemption, The (1994), Gladiator (2000), Dark Knight, The (2008)"
  - ItemKNN: "Vi ban da danh gia 8 phim tuong tu (gan nhat: Kill Bill: Vol. 1 (2003), Inception (2010), Shawshank Redemption, The (1994))"
- **User 3** → *King Kong (1933)*
  - MF: "Vi ban da xem Patton (1970), Doors, The (1991), Road Warrior, The (Mad Max 2) (1981)"
  - ItemKNN: "Vi ban da danh gia 6 phim tuong tu (gan nhat: Highlander (1986), Thing, The (1982), Road Warrior, The (Mad Max 2) (1981))"

# Số liệu B8: LightGCN

Bậc B8 cài trong `recsys/models/lightgcn.py`, tự viết bằng numpy và scipy. LightGCN (He và cộng
sự, SIGIR 2020) dùng cùng hàm mất mát và cùng dạng điểm với BPR ở B4. Khác B4 ở một chỗ: vector
của user và phim được lan truyền K lớp trên đồ thị user-phim trước khi tính điểm. Mỗi lớp lấy
trung bình có chuẩn hóa của hàng xóm, không có ma trận trọng số, không có hàm kích hoạt. Với
K = 0, mô hình là BPR-MF không bias.

Đề xuất đề tài ghi "B8 NeuMF hoặc LightGCN". Nhóm chọn LightGCN vì nó chỉ khác B4 đúng một thay
đổi, nên chênh lệch với B4 quy được cho đồ thị. Gradient của bước lan truyền tính bằng chính phép
lan truyền, nên không cần PyTorch.

Giao thức như B3-B7: phá hòa timestamp theo seed, siêu tham số chọn trên validation, báo cáo trên
test. Phần "Kết luận" do người viết. Phần "Số liệu chi tiết" là output nguyên văn của
`python scripts/exp_lightgcn.py`, chỉ hạ một cấp tiêu đề. CSV kèm seed và commit nằm ở
`reports/exp_lightgcn_*.csv` và `.meta.json`.

> **Quy tắc:** chênh lệch nào có CI 95% chứa 0 thì không viết "mô hình X thắng".

---

## Kết luận

### Siêu tham số chọn trên validation

Cấu hình chọn: K = 4 lớp, lr = 0.01, reg = 1e-4, 300 epoch, f = 64, lô 8.192 cặp, Adam. nDCG@10
validation 0.5077. BPR được 0.4928, MF implicit được 0.5118.

- Giá trị này ghi trong `N_LAYERS`, `LR`, `REG`, `N_EPOCHS` của `recsys/models/lightgcn.py`.
- **K = 4 và 300 epoch đều nằm ở biên của lưới.** Chưa thử K = 5 hay quá 300 epoch.
- K = 3 cùng cấu hình được 0.5054, kém K = 4 chỉ 0.0023. Chỉ có một lần chia dữ liệu, chưa có
  CI trên validation.
- reg = 0.01 kém rõ ở cả ba lr, không cấu hình nào vượt 0.452.
- lr = 0.05 đạt đỉnh 0.4789 sau 25 epoch rồi giảm dần.
- Cỡ lô chọn trước lưới, trên validation với K = 3. Lô 2.048 như bản gốc mất khoảng 1,2 giây mỗi
  epoch. Lô 8.192 với lr 0.02 đạt 0.4973 sau 160 epoch, nhanh hơn khoảng 3 lần tính theo thời gian.

### Khi lấy mẫu, B8 hơn BPR, sát ngưỡng. Trên toàn danh mục là nhiễu

| B8 trừ | nDCG@10, lấy mẫu | nDCG@10, toàn danh mục | AUC, lấy mẫu |
|---|---|---|---|
| BPR | +0.022 [+0.006, +0.038], có ý nghĩa | +0.001 [−0.009, +0.010], nhiễu | +0.000 [−0.004, +0.004], nhiễu |
| MF implicit | +0.011 [−0.006, +0.027], nhiễu | +0.003 [−0.006, +0.011], nhiễu | +0.025 [+0.015, +0.037], có ý nghĩa |
| ItemKNN | +0.035 [+0.015, +0.056], có ý nghĩa | +0.004 [−0.006, +0.015], nhiễu | (không so được) |
| Popularity | +0.152 [+0.128, +0.177], có ý nghĩa | +0.019 [+0.008, +0.030], có ý nghĩa | +0.054 [+0.044, +0.064], có ý nghĩa |

- Trên test, B8 được recall@10 0.7574, nDCG@10 0.5112, MRR 0.4436, AUC 0.8858, coverage@10 0.1055.
- **Chênh lệch với BPR phụ thuộc lựa chọn K.** Với K = 3, B8 hơn BPR +0.014 nDCG@10, CI
  [−0.002, +0.030], tức là nhiễu. K = 3 và K = 4 chỉ cách nhau 0.0023 trên validation. Báo cáo
  viết chênh lệch này là "nhỏ, sát ngưỡng", không viết "LightGCN thắng BPR".
- Về recall@10 khi lấy mẫu, B8 hơn BPR +0.025 và MF +0.030, đều có ý nghĩa nhưng cận dưới sát 0
  (+0.002 và +0.003). Trên toàn danh mục, cả hai là nhiễu.
- AUC của B8 bằng BPR (0.8858). Hai mô hình cùng tối ưu BPR-Opt, dạng liên tục của AUC. Cả hai hơn
  MF khoảng 0.025 AUC.
- Trên toàn danh mục, B8, BPR, MF và ItemKNN hòa nhau. Chỉ chênh lệch với Popularity còn có ý nghĩa.

### Đồ thị đóng góp so với K = 0, nhưng một phần là do K = 0 không được chỉnh riêng

| K | nDCG@10 test | K trừ (K = 0) | K trừ B4-BPR |
|---|---|---|---|
| 0 | 0.430 | | −0.059, có ý nghĩa |
| 1 | 0.475 | +0.045, có ý nghĩa | −0.014, nhiễu |
| 2 | 0.496 | +0.066, có ý nghĩa | +0.007, nhiễu |
| 3 | 0.502 | +0.072, có ý nghĩa | +0.014, nhiễu |
| 4 | 0.511 | +0.081, có ý nghĩa | +0.022, có ý nghĩa |

Mọi dòng dùng cùng lr, reg và 300 epoch đã chọn cho K = 4.

- nDCG@10 tăng đều theo K, cả trên validation lẫn test.
- **Không đọc +0.081 là phần đóng góp của đồ thị.** K = 0 đạt đỉnh validation 0.4492 ở 100
  epoch, rồi giảm còn 0.4218 ở 300 epoch. Dòng K = 0 trong bảng là mô hình đã học quá lâu.
- So với B4, mô hình đã chỉnh siêu tham số riêng, đồ thị chỉ thêm +0.022 ở K = 4. Ở K = 1 đến 3,
  chênh lệch với B4 là nhiễu.

### Độ phủ: B8 trả về nhiều phim khác nhau nhất trong các mô hình lọc cộng tác

- Top-10 của B8 có 1.023 phim khác nhau, phủ 10,5% danh mục. MF 976, BPR 884, ItemKNN 727.
- Phim B8 gợi ý phổ biến hơn BPR một chút: lượt xem trung bình 130,2 so với 121,7, và 51,6% thuộc
  top-100 phổ biến so với 48,5%.
- Đây là số đếm trên một lần chia, chưa có CI.
- B8 không có bias. Độ dài vector phim của B8 có Spearman 0.827 với số lượt xem, của BPR là 0.880.
  Bias của BPR có Spearman 0.949. Cả hai mô hình đều mã hóa độ phổ biến vào độ dài vector.

### User mới: B8 kém BPR ở 1 tương tác, hòa từ 3 tương tác

| lịch sử | BPR | B8 | B8 − BPR | B8 − Popularity |
|---|---|---|---|---|
| 1 | 0.437 | 0.408 | −0.029 [−0.052, −0.006], có ý nghĩa | +0.047, có ý nghĩa |
| 3 | 0.466 | 0.475 | +0.010, nhiễu | +0.114, có ý nghĩa |
| 5 | 0.491 | 0.487 | −0.004, nhiễu | +0.127, có ý nghĩa |
| 10 | 0.504 | 0.508 | +0.004, nhiễu | +0.147, có ý nghĩa |
| đầy đủ | 0.489 | 0.511 | +0.022 [+0.006, +0.038], có ý nghĩa | +0.152, có ý nghĩa |

nDCG@10, lấy mẫu, đủ 610 user, cắt lịch sử của từng 1/4 số user như checklist 2.8.

- Lan truyền trên đồ thị không giúp user chỉ có 1 tương tác. Ở mức đó, B8 kém BPR có ý nghĩa.
- Với user mới, B7 vẫn nên dùng BPR. B8 không thay BPR trong B7 được nếu chỉ dựa vào bảng này.

### Câu giải thích là hậu kiểm, giống B4

- `explain()` kế thừa từ BPR: nêu ba phim đã xem có vector cuối gần phim gợi ý nhất (cosine).
- Câu này nói "phim này giống những phim bạn đã xem", không nói điểm đến từ đâu.
- Ở cả ba ví dụ, LightGCN và BPR nêu chung hai trên ba phim.

### Giá phải trả

- Fit trên train mất 376 giây. BPR mất 32 giây, MF 8 giây. B8 chậm hơn BPR khoảng 12 lần.
- `run_eval.py` vì vậy chạy lâu hơn trước khoảng 6 phút.
- Cả script `exp_lightgcn.py` chạy khoảng 1,5 đến 2 giờ trên một máy.

---

## Số liệu chi tiết

ml-latest-small · seed 42 · 610 user kiểm tra · ItemKNN k = 80 · MF, BPR theo hằng số mf.py, bpr.py

### 1. Chọn siêu tham số trên validation (ndcg@10)

Mỗi cấu hình huấn luyện một lần, đánh giá sau 25, 50, 100, 200, 300 epoch. Ô đậm là tốt nhất của từng cấu hình. f = 64, lô 8.192 cặp, Adam.

| số lớp K | lr | reg | 25 epoch | 50 epoch | 100 epoch | 200 epoch | 300 epoch |
|---|---|---|---|---|---|---|---|
| 3 | 0.01 | 0.0001 | 0.4468 | 0.4720 | 0.4936 | 0.5018 | **0.5054** |
| 3 | 0.01 | 0.01 | 0.4288 | 0.4442 | 0.4472 | 0.4472 | **0.4512** |
| 3 | 0.02 | 0.0001 | 0.4728 | 0.4891 | 0.4940 | **0.5028** | 0.4935 |
| 3 | 0.02 | 0.01 | 0.4430 | **0.4518** | 0.4429 | 0.4476 | 0.4512 |
| 3 | 0.05 | 0.0001 | **0.4789** | 0.4721 | 0.4779 | 0.4762 | 0.4566 |
| 3 | 0.05 | 0.01 | 0.4427 | 0.4512 | 0.4431 | 0.4493 | **0.4517** |
| 0 | 0.01 | 0.0001 | 0.4458 | 0.4492 | **0.4492** | 0.4441 | 0.4218 |
| 1 | 0.01 | 0.0001 | 0.4752 | 0.4854 | **0.4887** | 0.4854 | 0.4737 |
| 2 | 0.01 | 0.0001 | 0.4603 | 0.4751 | 0.4908 | **0.4977** | 0.4922 |
| 4 | 0.01 | 0.0001 | 0.4347 | 0.4656 | 0.4877 | 0.4990 | **0.5077** |

Chọn: {'n_layers': 4, 'lr': 0.01, 'reg': 0.0001, 'n_epochs': 300}, ndcg@10 validation 0.5077.

### 2. Bảng chính trên test (1 phim đúng + 99 phim âm, CI 95%)

| Mô hình | recall@10 | ndcg@10 | AUC | coverage@10 | ứng viên −inf | giây fit |
|---|---|---|---|---|---|---|
| B0-Popularity | 0.5967 [0.5573, 0.6361] | 0.3590 [0.3302, 0.3879] | 0.8315 [0.8110, 0.8501] | 0.0126 | 0.3% | 0 |
| B1-ItemKNN | 0.6803 [0.6410, 0.7148] | 0.4761 [0.4436, 0.5064] | 0.8342 [0.8135, 0.8524] | 0.0749 | 81.2% | 3 |
| B3-MF-implicit | 0.7279 [0.6918, 0.7639] | 0.5003 [0.4701, 0.5299] | 0.8605 [0.8386, 0.8802] | 0.1006 | 0.3% | 8 |
| B4-BPR | 0.7328 [0.6984, 0.7672] | 0.4886 [0.4590, 0.5176] | 0.8858 [0.8678, 0.9020] | 0.0911 | 0.3% | 32 |
| B8-LightGCN | 0.7574 [0.7213, 0.7885] | 0.5112 [0.4800, 0.5385] | 0.8858 [0.8673, 0.9026] | 0.1055 | 0.3% | 376 |

#### Chênh lệch ghép cặp: lấy mẫu và toàn danh mục

| So sánh | độ đo | lấy mẫu | toàn danh mục |
|---|---|---|---|
| B8-LightGCN − B4-BPR | ndcg@10 | +0.0225 [+0.0062, +0.0378] **có ý nghĩa** | +0.0006 [-0.0094, +0.0099] nhiễu |
| B8-LightGCN − B3-MF-implicit | ndcg@10 | +0.0109 [-0.0059, +0.0273] nhiễu | +0.0029 [-0.0063, +0.0112] nhiễu |
| B8-LightGCN − B1-ItemKNN | ndcg@10 | +0.0351 [+0.0148, +0.0555] **có ý nghĩa** | +0.0042 [-0.0060, +0.0148] nhiễu |
| B8-LightGCN − B0-Popularity | ndcg@10 | +0.1522 [+0.1279, +0.1766] **có ý nghĩa** | +0.0187 [+0.0082, +0.0298] **có ý nghĩa** |
| B8-LightGCN − B4-BPR | recall@10 | +0.0246 [+0.0016, +0.0459] **có ý nghĩa** | +0.0066 [-0.0131, +0.0262] nhiễu |
| B8-LightGCN − B3-MF-implicit | recall@10 | +0.0295 [+0.0033, +0.0574] **có ý nghĩa** | +0.0049 [-0.0148, +0.0230] nhiễu |
| B8-LightGCN − B1-ItemKNN | recall@10 | +0.0770 [+0.0475, +0.1066] **có ý nghĩa** | +0.0148 [-0.0066, +0.0361] nhiễu |
| B8-LightGCN − B0-Popularity | recall@10 | +0.1607 [+0.1246, +0.1967] **có ý nghĩa** | +0.0377 [+0.0164, +0.0590] **có ý nghĩa** |
| B8-LightGCN − B4-BPR | auc | +0.0000 [-0.0044, +0.0043] nhiễu | -0.0000 [-0.0040, +0.0038] nhiễu |
| B8-LightGCN − B3-MF-implicit | auc | +0.0253 [+0.0151, +0.0365] **có ý nghĩa** | +0.0262 [+0.0164, +0.0371] **có ý nghĩa** |
| B8-LightGCN − B1-ItemKNN | auc | +0.0516 [+0.0362, +0.0678] **có ý nghĩa** | +0.0520 [+0.0367, +0.0681] **có ý nghĩa** |
| B8-LightGCN − B0-Popularity | auc | +0.0543 [+0.0444, +0.0644] **có ý nghĩa** | +0.0537 [+0.0437, +0.0638] **có ý nghĩa** |

Toàn danh mục, ndcg@10: B0-Popularity 0.0182 [0.0109, 0.0264], B1-ItemKNN 0.0327 [0.0228, 0.0438], B3-MF-implicit 0.0340 [0.0244, 0.0454], B4-BPR 0.0363 [0.0252, 0.0481], B8-LightGCN 0.0369 [0.0268, 0.0484]

### 3. Đóng góp của đồ thị: số lớp lan truyền K, cùng lr, reg, epoch

K = 0 là BPR-MF không bias, học bằng Adam. Mọi dòng dùng cùng lr, reg, số epoch đã chọn ở mục 1.

| K | ndcg@10 validation | ndcg@10 test | K − (K = 0), test | K − B4-BPR, test |
|---|---|---|---|---|
| 0 | 0.4218 | 0.4299 [0.3985, 0.4594] |  | -0.0587 [-0.0792, -0.0382] **có ý nghĩa** |
| 1 | 0.4737 | 0.4748 [0.4430, 0.5032] | +0.0448 [+0.0283, +0.0620] **có ý nghĩa** | -0.0139 [-0.0324, +0.0045] nhiễu |
| 2 | 0.4922 | 0.4959 [0.4649, 0.5239] | +0.0660 [+0.0479, +0.0838] **có ý nghĩa** | +0.0073 [-0.0093, +0.0237] nhiễu |
| 3 | 0.5054 | 0.5023 [0.4716, 0.5292] | +0.0724 [+0.0531, +0.0916] **có ý nghĩa** | +0.0137 [-0.0024, +0.0296] nhiễu |
| 4 | 0.5077 | 0.5112 [0.4800, 0.5385] | +0.0813 [+0.0616, +0.1008] **có ý nghĩa** | +0.0225 [+0.0062, +0.0378] **có ý nghĩa** |

### 4. Thiên lệch phổ biến của top-10 (toàn danh mục, mọi user)

| Mô hình | số phim khác nhau trong top-10 | phủ danh mục | lượt xem TB | trung vị lượt xem | tỉ lệ thuộc top-100 phổ biến |
|---|---|---|---|---|---|
| B0-Popularity | 122 | 1.3% | 232.7 | 218 | 99.3% |
| B1-ItemKNN | 727 | 7.5% | 129.6 | 122 | 56.3% |
| B3-MF-implicit | 976 | 10.1% | 124.5 | 109 | 48.3% |
| B4-BPR | 884 | 9.1% | 121.7 | 108 | 48.5% |
| B8-LightGCN | 1023 | 10.5% | 130.2 | 115 | 51.6% |

- Spearman giữa độ dài vector phim của B4-BPR và số lượt xem: 0.880.
- Spearman giữa độ dài vector phim của B8-LightGCN và số lượt xem: 0.827.
- Tham chiếu: Spearman giữa bias b_i của B4-BPR và số lượt xem: 0.949.

### 5. Cold-start nhân tạo (cắt còn n tương tác gần nhất)

ndcg@10, CI 95%, đủ 610 user.

| giữ lại | B0-Popularity | B3-MF-implicit | B4-BPR | B8-LightGCN | LightGCN − BPR | LightGCN − Popularity |
|---|---|---|---|---|---|---|
| 1 tương tác | 0.3610 [0.3326, 0.3891] | 0.3898 [0.3601, 0.4207] | 0.4367 [0.4082, 0.4662] | 0.4077 [0.3772, 0.4387] | -0.0289 [-0.0523, -0.0057] **có ý nghĩa** | +0.0467 [+0.0176, +0.0776] **có ý nghĩa** |
| 3 tương tác | 0.3613 [0.3328, 0.3894] | 0.4543 [0.4243, 0.4843] | 0.4656 [0.4359, 0.4946] | 0.4751 [0.4447, 0.5064] | +0.0095 [-0.0093, +0.0297] nhiễu | +0.1138 [+0.0822, +0.1451] **có ý nghĩa** |
| 5 tương tác | 0.3600 [0.3315, 0.3883] | 0.4785 [0.4472, 0.5097] | 0.4908 [0.4609, 0.5211] | 0.4867 [0.4560, 0.5174] | -0.0042 [-0.0214, +0.0134] nhiễu | +0.1267 [+0.0974, +0.1562] **có ý nghĩa** |
| 10 tương tác | 0.3612 [0.3320, 0.3891] | 0.5001 [0.4694, 0.5308] | 0.5035 [0.4743, 0.5330] | 0.5077 [0.4783, 0.5368] | +0.0042 [-0.0133, +0.0221] nhiễu | +0.1465 [+0.1179, +0.1748] **có ý nghĩa** |
| đầy đủ | 0.3590 [0.3302, 0.3879] | 0.5003 [0.4701, 0.5299] | 0.4886 [0.4590, 0.5176] | 0.5112 [0.4800, 0.5385] | +0.0225 [+0.0062, +0.0378] **có ý nghĩa** | +0.1522 [+0.1279, +0.1766] **có ý nghĩa** |

### 6. Ví dụ câu giải thích thật

- **User 1** → *Die Hard (1988)*
  - LightGCN: "Giong cac phim ban da xem: Indiana Jones and the Last Crusade (1989), Star Wars: Episode VI - Return of the Jedi (1983), Star Wars: Episode V - The Empire Strikes Back (1980)"
  - BPR: "Giong cac phim ban da xem: Indiana Jones and the Last Crusade (1989), Star Wars: Episode V - The Empire Strikes Back (1980), Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)"
- **User 2** → *Social Network, The (2010)*
  - LightGCN: "Giong cac phim ban da xem: Inception (2010), Dark Knight, The (2008), Django Unchained (2012)"
  - BPR: "Giong cac phim ban da xem: Dark Knight, The (2008), Inception (2010), Inglourious Basterds (2009)"
- **User 3** → *Alien (1979)*
  - LightGCN: "Giong cac phim ban da xem: Thing, The (1982), Deer Hunter, The (1978), Road Warrior, The (Mad Max 2) (1981)"
  - BPR: "Giong cac phim ban da xem: Schindler's List (1993), Thing, The (1982), Deer Hunter, The (1978)"

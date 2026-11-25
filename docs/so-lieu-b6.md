# Số liệu B6: luật kết hợp theo phiên

Bậc B6 cài trong `recsys/models/rules.py`. Phiên là chuỗi rating của một user, hai rating liền
nhau cách không quá 1800 giây. Mỗi phiên là một giao dịch. `mlxtend.fpgrowth` khai phá luật
{a} → {j}. Điểm của phim j là confidence lớn nhất của các luật có a nằm trong 10 phim gần nhất.

Giao thức như B3, B4, B5: phá hòa timestamp theo seed, siêu tham số chọn trên validation, báo
cáo trên test. Phần "Kết luận" do người viết. Phần "Số liệu chi tiết" là output nguyên văn của
`python scripts/exp_rules.py`. CSV kèm seed và commit nằm ở `reports/exp_rules_*.csv` và
`.meta.json`.

> **Quy tắc:** chênh lệch nào có CI 95% chứa 0 thì không viết "mô hình X thắng".

---

## Kết luận

### Siêu tham số chọn trên validation

Cấu hình chọn: khoảng nghỉ 1800 giây, min_support 0,0005, max_len 2, ngữ cảnh 10 phim gần
nhất, confidence, lấy max. nDCG@10 validation 0.4995. BPR được 0.4928.

- Giá trị này ghi trong `SESSION_GAP`, `MIN_SUPPORT`, `MAX_LEN`, `CONTEXT`, `METRIC`, `AGG`
  của `recsys/models/rules.py`.
- Hạ min_support thì nDCG@10 tăng đều, từ 0.3169 (0,005) lên 0.4867 (0,0005). Số luật tăng
  từ 40.040 lên 2.303.914. Mức 0,0005 tương ứng "cặp phim chung ít nhất 2 phiên", nên không
  hạ tiếp.
- Khoảng nghỉ 1 ngày hơn 30 phút ở min_support 0,005 và 0,002, kém ở 0,001 và 0,0005. Ở
  0,0005, khoảng nghỉ 1 ngày được 0.4669 và khai phá mất 984 giây.
- Ngữ cảnh 10 phim gần nhất được 0.4995, 3 phim 0.4970, phiên cuối 0.4867, 1 phim 0.4315.
  Chênh lệch giữa 10 phim và 3 phim là 0.0025, chỉ đo trên một lần chia.
- lift kém confidence ở cả 4 ngữ cảnh. Cấu hình lift tốt nhất được 0.3718.
- max_len 3 sinh 348.968 luật, gấp 9 lần max_len 2, và chỉ tăng từ 0.3050 lên 0.3081.

### B6 hơn Popularity, hòa ItemKNN và BPR về nDCG@10

| B6 trừ | nDCG@10, lấy mẫu | nDCG@10, toàn danh mục |
|---|---|---|
| Popularity | +0.119 [+0.098, +0.141], có ý nghĩa | +0.024 [+0.012, +0.037], có ý nghĩa |
| ItemKNN | +0.002 [−0.020, +0.023], nhiễu | +0.010 [−0.004, +0.024], nhiễu |
| BPR | −0.011 [−0.034, +0.013], nhiễu | +0.006 [−0.007, +0.020], nhiễu |

- Trên test, B6 được recall@10 0.7213, nDCG@10 0.4780, MRR 0.4131, AUC 0.8689.
- Về recall@10 khi lấy mẫu, B6 hơn ItemKNN +0.041, CI [+0.010, +0.071]. Trên toàn danh mục,
  chênh lệch này là nhiễu.
- Trên toàn danh mục, nDCG@10 của B6 là 0.0427, BPR 0.0363, ItemKNN 0.0327. Cả hai chênh
  lệch ghép cặp đều có CI chứa 0.

### 556/610 phim test nằm cùng phiên với rating cuối trong train

Phim test cách rating cuối trong train không quá 1800 giây ở 556 user (91%). Trên MovieLens,
phiên là một buổi chấm điểm hàng loạt, không phải một buổi xem phim.

- Nhóm cùng phiên: B6 0.4942, BPR 0.5008, chênh lệch −0.007 [−0.030, +0.018], nhiễu.
- Nhóm phiên mới (54 user): B6 0.3112, BPR 0.3636, chênh lệch −0.052 [−0.123, +0.014],
  nhiễu. Nhóm này có 54 user nên CI rộng 0,14.
- Chỉ nhìn 10 phim gần nhất, B6 hòa BPR, mô hình học trên toàn bộ lịch sử.

### B6 trả về nhiều phim khác nhau hơn BPR, nhưng phim phổ biến hơn

| Mô hình | số phim khác nhau | lượt xem trung vị | thuộc top-100 phổ biến |
|---|---|---|---|
| ItemKNN | 727 | 122 | 56.3% |
| BPR | 884 | 108 | 48.5% |
| B6 | 1037 | 137 | 59.9% |

- 5.328/9.701 phim (54,9%) làm vế phải của ít nhất một luật. 71,4% ứng viên trong bảng test
  nhận −inf vì không có luật khớp.
- 99,5% user vẫn nhận đủ 10 phim. 0,5% còn lại nhận ít hơn 10.

### User mới: B6 hơn Popularity từ 1 tương tác, thua BPR ở 1, 5, 10 tương tác

| giữ lại | B6 trừ Popularity | B6 trừ BPR |
|---|---|---|
| 1 tương tác | +0.027 [+0.002, +0.053], có ý nghĩa | −0.049 [−0.073, −0.026], có ý nghĩa |
| 3 tương tác | +0.089 [+0.067, +0.110], có ý nghĩa | −0.016 [−0.038, +0.007], nhiễu |
| 5 tương tác | +0.102 [+0.081, +0.122], có ý nghĩa | −0.029 [−0.054, −0.007], có ý nghĩa |
| 10 tương tác | +0.104 [+0.083, +0.127], có ý nghĩa | −0.038 [−0.061, −0.017], có ý nghĩa |

Bảng này bất lợi cho B6. Luật khai phá một lần mỗi fold trên train **không có user của fold**,
tức khoảng 3/4 số user (4 fold). Lịch sử cắt của user trong fold chỉ được nạp vào ngữ cảnh qua
`fold_in()`. Vì vậy ở 10 tương tác B6 được 0.4653, thấp hơn 0.4780 khi khai phá trên đủ user,
dù ngữ cảnh đều là 10 phim gần nhất.

### Câu giải thích là luật mạnh nhất kèm số phiên làm bằng chứng

`explain()` nêu luật có confidence lớn nhất cho phim được gợi ý, kèm confidence và số phiên
chứa cả hai phim. Ví dụ với user 3: "Người chấm Rescuers, The (1977) trong cùng một phiên
thường chấm cả phim này (độ tin cậy 77%, 10 phiên)". Câu trong mã viết không dấu.

### Chi phí

Khai phá ở min_support 0,0005 mất 218 giây trên ml-latest-small. Mỗi lần `fit()` đều khai
phá lại, kể cả trong `run_eval.py`. `rescore()` đổi cách chấm điểm, `fold_in()` thay lịch sử
user, cả hai không khai phá lại.

---

## Số liệu chi tiết

ml-latest-small · seed 42 · 610 user kiểm tra · ItemKNN k = 80 · BPR theo hằng số bpr.py

### 1. Chọn siêu tham số trên validation (ndcg@10)

#### a. Khoảng nghỉ cắt phiên và min_support

Ngữ cảnh là phiên cuối, điểm là confidence lớn nhất, max_len = 2.

| khoảng nghỉ | min_support | số luật | ndcg@10 | recall@10 | ứng viên −inf | giây khai phá |
|---|---|---|---|---|---|---|
| 1800 giây | 0.005 | 40,040 | 0.3169 | 0.4197 | 96.7% | 4 |
| 1800 giây | 0.002 | 252,190 | 0.4138 | 0.5934 | 90.4% | 25 |
| 1800 giây | 0.001 | 802,054 | 0.4666 | 0.6852 | 81.5% | 80 |
| 1800 giây | 0.0005 | 2,303,914 | 0.4867 | 0.7311 | 66.3% | 218 |
| 86400 giây | 0.005 | 88,720 | 0.3526 | 0.4951 | 94.6% | 11 |
| 86400 giây | 0.002 | 551,726 | 0.4326 | 0.6393 | 85.5% | 94 |
| 86400 giây | 0.001 | 1,325,106 | 0.4556 | 0.6918 | 76.1% | 231 |
| 86400 giây | 0.0005 | 3,605,216 | 0.4669 | 0.7197 | 59.2% | 984 |

#### b. Cách chấm điểm (khoảng nghỉ 1800 giây, min_support 0.0005)

| ngữ cảnh | confidence, max | confidence, sum | lift, max | lift, sum |
|---|---|---|---|---|
| 1 phim gần nhất | 0.4315 | 0.4315 | 0.2746 | 0.2746 |
| 3 phim gần nhất | 0.4970 | 0.4905 | 0.2717 | 0.3454 |
| 10 phim gần nhất | **0.4995** | 0.4934 | 0.2040 | 0.3718 |
| phiên cuối | 0.4867 | 0.4700 | 0.1913 | 0.3609 |

#### c. max_len 3 so với 2 (min_support 0.005, cách chấm điểm ở b)

| max_len | số luật | ndcg@10 | giây khai phá |
|---|---|---|---|
| 2 | 40,040 | 0.3050 | 6 |
| 3 | 348,968 | 0.3081 | 16 |

Chọn: {'session_gap': 1800, 'min_support': 0.0005, 'max_len': 2, 'context': 10, 'metric': 'confidence', 'agg': 'max'}, ndcg@10 validation 0.4995.

### 2. Bảng chính trên test (1 phim đúng + 99 phim âm, CI 95%)

| Mô hình | recall@10 | ndcg@10 | AUC | coverage@10 | ứng viên −inf |
|---|---|---|---|---|---|
| B0-Popularity | 0.5967 [0.5573, 0.6361] | 0.3590 [0.3302, 0.3879] | 0.8315 [0.8110, 0.8501] | 0.0126 | 0.3% |
| B1-ItemKNN | 0.6803 [0.6410, 0.7148] | 0.4761 [0.4436, 0.5064] | 0.8342 [0.8135, 0.8524] | 0.0749 | 81.2% |
| B4-BPR | 0.7328 [0.6984, 0.7672] | 0.4886 [0.4590, 0.5176] | 0.8858 [0.8678, 0.9020] | 0.0911 | 0.3% |
| B6-Rules | 0.7213 [0.6852, 0.7557] | 0.4780 [0.4483, 0.5071] | 0.8689 [0.8519, 0.8847] | 0.1069 | 71.4% |

#### Chênh lệch ghép cặp: lấy mẫu và toàn danh mục

| So sánh | độ đo | lấy mẫu | toàn danh mục |
|---|---|---|---|
| B6-Rules − B0-Popularity | ndcg@10 | +0.1190 [+0.0975, +0.1409] **có ý nghĩa** | +0.0244 [+0.0118, +0.0374] **có ý nghĩa** |
| B6-Rules − B1-ItemKNN | ndcg@10 | +0.0019 [-0.0202, +0.0233] nhiễu | +0.0100 [-0.0038, +0.0241] nhiễu |
| B6-Rules − B4-BPR | ndcg@10 | -0.0107 [-0.0340, +0.0133] nhiễu | +0.0064 [-0.0073, +0.0203] nhiễu |
| B6-Rules − B0-Popularity | recall@10 | +0.1246 [+0.0950, +0.1541] **có ý nghĩa** | +0.0410 [+0.0197, +0.0623] **có ý nghĩa** |
| B6-Rules − B1-ItemKNN | recall@10 | +0.0410 [+0.0098, +0.0705] **có ý nghĩa** | +0.0180 [-0.0066, +0.0410] nhiễu |
| B6-Rules − B4-BPR | recall@10 | -0.0115 [-0.0443, +0.0180] nhiễu | +0.0098 [-0.0148, +0.0328] nhiễu |

Toàn danh mục, ndcg@10: B0-Popularity 0.0182 [0.0109, 0.0264], B1-ItemKNN 0.0327 [0.0228, 0.0438], B4-BPR 0.0363 [0.0252, 0.0481], B6-Rules 0.0427 [0.0301, 0.0561]

### 3. Phim test cùng phiên hay phiên mới (khoảng nghỉ 1800 giây)

ndcg@10, lấy mẫu, CI 95%. "Cùng phiên": phim test cách rating cuối trong train không quá 1800 giây.

| nhóm | số user | B0-Popularity | B1-ItemKNN | B4-BPR | B6-Rules | B6 − ItemKNN | B6 − BPR |
|---|---|---|---|---|---|---|---|
| cùng phiên | 556 | 0.3684 [0.3376, 0.3980] | 0.4906 [0.4568, 0.5214] | 0.5008 [0.4684, 0.5306] | 0.4942 [0.4629, 0.5251] | +0.0036 [-0.0188, +0.0277] nhiễu | -0.0066 [-0.0299, +0.0175] nhiễu |
| phiên mới | 54 | 0.2622 [0.1756, 0.3516] | 0.3268 [0.2311, 0.4208] | 0.3636 [0.2765, 0.4482] | 0.3112 [0.2231, 0.4022] | -0.0156 [-0.0781, +0.0447] nhiễu | -0.0524 [-0.1233, +0.0143] nhiễu |

### 4. Phạm vi của luật và thiên lệch phổ biến của top-10

2,311,672 luật trên 3,870 phiên có từ 2 phim. 5,328/9,701 phim (54.9%) làm vế phải của ít nhất một luật. Lượt xem trung vị của các phim đó: 7.

| Mô hình | user nhận đủ 10 phim | số phim khác nhau | phủ danh mục | lượt xem trung vị | tỉ lệ thuộc top-100 phổ biến |
|---|---|---|---|---|---|
| B0-Popularity | 100.0% | 122 | 1.3% | 218 | 99.3% |
| B1-ItemKNN | 100.0% | 727 | 7.5% | 122 | 56.3% |
| B4-BPR | 100.0% | 884 | 9.1% | 108 | 48.5% |
| B6-Rules | 99.5% | 1037 | 10.7% | 137 | 59.9% |

### 5. User mới: cắt còn n tương tác gần nhất (như checklist 2.8)

ndcg@10, CI 95%, đủ 610 user.

| giữ lại | B0-Popularity | B4-BPR | B6-Rules | B6 − Popularity | B6 − BPR |
|---|---|---|---|---|---|
| 1 tương tác | 0.3610 [0.3326, 0.3891] | 0.4367 [0.4082, 0.4662] | 0.3879 [0.3557, 0.4181] | +0.0268 [+0.0020, +0.0527] **có ý nghĩa** | -0.0488 [-0.0725, -0.0260] **có ý nghĩa** |
| 3 tương tác | 0.3613 [0.3328, 0.3894] | 0.4656 [0.4359, 0.4946] | 0.4498 [0.4199, 0.4777] | +0.0885 [+0.0673, +0.1100] **có ý nghĩa** | -0.0158 [-0.0383, +0.0071] nhiễu |
| 5 tương tác | 0.3600 [0.3315, 0.3883] | 0.4908 [0.4609, 0.5211] | 0.4616 [0.4315, 0.4896] | +0.1015 [+0.0810, +0.1224] **có ý nghĩa** | -0.0293 [-0.0536, -0.0067] **có ý nghĩa** |
| 10 tương tác | 0.3612 [0.3320, 0.3891] | 0.5035 [0.4743, 0.5330] | 0.4653 [0.4356, 0.4934] | +0.1041 [+0.0825, +0.1269] **có ý nghĩa** | -0.0382 [-0.0609, -0.0166] **có ý nghĩa** |
| đầy đủ | 0.3590 [0.3302, 0.3879] | 0.4886 [0.4590, 0.5176] | 0.4780 [0.4483, 0.5071] | +0.1190 [+0.0975, +0.1409] **có ý nghĩa** | -0.0107 [-0.0340, +0.0133] nhiễu |

### 6. Ví dụ câu giải thích thật

- **User 1** → *My Cousin Vinny (1992)*: "Nguoi cham McHale's Navy (1997) trong cung mot phien thuong cham ca phim nay (do tin cay 71%, 5 phien)"
- **User 2** → *Forrest Gump (1994)*: "Nguoi cham Warrior (2011) trong cung mot phien thuong cham ca phim nay (do tin cay 64%, 7 phien)"
- **User 3** → *Alice in Wonderland (1951)*: "Nguoi cham Rescuers, The (1977) trong cung mot phien thuong cham ca phim nay (do tin cay 77%, 10 phien)"

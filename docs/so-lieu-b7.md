# Số liệu B7: lai ghép

Bậc B7 cài trong `recsys/models/hybrid.py`. Mô hình ghép ba bậc đã có theo ba nhánh. User chưa có
lịch sử dùng Popularity (B0). Phim BPR không chấm được dùng nội dung (B5). Phần còn lại trộn
`(1 − α_u) · BPR + α_u · B5`, sau khi đổi điểm của mỗi nhánh sang phân vị trong toàn danh mục,
riêng cho từng user. α_u tăng theo số tương tác n của user: `α_u = α · n / (n + c)`.

Giao thức như B3-B5: phá hòa timestamp theo seed, siêu tham số chọn trên validation, báo cáo trên
test. Phần "Kết luận" do người viết. Phần "Số liệu chi tiết" là output nguyên văn của
`python scripts/exp_hybrid.py`, chỉ hạ một cấp tiêu đề. CSV kèm seed và commit nằm ở
`reports/exp_hybrid_*.csv` và `.meta.json`.

> **Quy tắc:** chênh lệch nào có CI 95% chứa 0 thì không viết "mô hình X thắng".

Trong các bảng, "B7 α chung" là cùng mô hình đã fit nhưng mọi user dùng α = 0.1 (mục 1). Đây là
cấu hình B7 trước khi có α theo lịch sử.

---

## Kết luận

### Siêu tham số chọn trên validation, hai bước

Bước 1 chọn cách chuẩn hóa trên validation thường, mọi user cùng α: `rank`, α = 0.1, nDCG@10 0.5161.
BPR được 0.4928, B5 được 0.2571. α = 0 (chuyển thuần, phim cũ xếp đúng như BPR) được 0.4991.

Bước 2 chọn α và dạng hàm của α_u trên validation có cắt lịch sử. Mỗi kịch bản cắt lịch sử của
từng 1/4 số user xuống 1, 3, 5, 10 tương tác, cộng thêm kịch bản lịch sử đầy đủ. Tiêu chí là trung
bình nDCG@10 của 5 kịch bản, trọng số bằng nhau. Cấu hình chọn: α = 0.1, dạng co rút, c = 30,
trung bình 0.4816.

- Giá trị này ghi trong `ALPHA`, `NORM`, `RAMP`, `RAMP_N` của `recsys/models/hybrid.py`.
- Với c = 30, user 1 tương tác có α_u = 0.003, user 100 tương tác có α_u = 0.077.
- c = 30 nằm ở biên của lưới (1, 3, 10, 30), chưa thử lớn hơn.
- **BPR một mình được 0.4831 theo cùng tiêu chí, cao hơn mọi cấu hình B7.** Mục "User mới" ở dưới
  giải thích phần thiếu này không do α.
- Chỉ có một lần chia dữ liệu, chưa có CI trên validation.

### α theo lịch sử giúp user 1 tương tác, không làm mất gì ở lịch sử đầy đủ

| lịch sử | BPR | B7 α chung | B7 | B7 − B7 α chung | B7 − BPR |
|---|---|---|---|---|---|
| 1 | 0.437 | 0.406 | 0.426 | +0.020 [+0.004, +0.036], có ý nghĩa | −0.010 [−0.018, −0.002], có ý nghĩa |
| 3 | 0.466 | 0.444 | 0.456 | +0.012 [−0.003, +0.027], nhiễu | −0.009 [−0.018, +0.000], nhiễu |
| 5 | 0.491 | 0.483 | 0.487 | +0.004, nhiễu | −0.004, nhiễu |
| 10 | 0.504 | 0.494 | 0.498 | +0.004, nhiễu | −0.006, nhiễu |
| đầy đủ | 0.489 | 0.527 | 0.522 | −0.006 [−0.013, +0.002], nhiễu | +0.033 [+0.018, +0.047], có ý nghĩa |

nDCG@10, lấy mẫu, đủ 610 user, trên test.

- Trước khi có α theo lịch sử, B7 kém BPR có ý nghĩa ở 1 và 3 tương tác (−0.030, −0.022). Giờ chỉ
  còn kém ở 1 tương tác, mức −0.010.
- Ở 0 tương tác, B7 dùng Popularity, được 0.361. BPR chấm `-inf` mọi phim nên chỉ được 0.044.

### Phần B7 còn kém BPR ở user mới đến từ nhánh phim mới, không từ α

Trên validation, các dòng `nguong 20` cho α_u = 0 với mọi user bị cắt lịch sử. Khi đó B7 xếp phim
cũ đúng như BPR. Dù vậy B7 vẫn thấp hơn BPR ở cả bốn mức:

| lịch sử | BPR | B7, α_u = 0 | chênh |
|---|---|---|---|
| 1 | 0.4166 | 0.4082 | −0.008 |
| 3 | 0.4812 | 0.4714 | −0.010 |
| 5 | 0.5113 | 0.5019 | −0.009 |
| 10 | 0.5138 | 0.5060 | −0.008 |

Phần chênh này là phim chưa ai xem nằm trong 99 phim âm. B7 chấm chúng bằng phân vị nội dung, nên
một số phim mới chen lên trên phim đúng. Tính từ bảng mục 1b và mục 1: α = 0 cho mọi user được
trung bình 0.4773, thấp hơn cấu hình đã chọn. Số chênh ở bảng trên chưa có CI.

Hướng sửa, chưa làm: giảm điểm phim mới theo lịch sử user, giống α_u. Đổi lại, B7 sẽ gợi ý phim
mới ít hơn. Hiện B7 trúng 7/23 phim test mới (mục dưới), con số đó có thể giảm.

### Khi lấy mẫu, B7 hơn B0, B3, B4, B5. Trên toàn danh mục, nDCG@10 chỉ còn hơn Popularity

| B7 trừ | nDCG@10, lấy mẫu | nDCG@10, toàn danh mục |
|---|---|---|
| Popularity | +0.163 [+0.138, +0.188], có ý nghĩa | +0.019 [+0.007, +0.031], có ý nghĩa |
| MF (B3) | +0.021 [+0.001, +0.042], có ý nghĩa, sát ngưỡng | +0.004 [−0.009, +0.015], nhiễu |
| BPR (B4) | +0.033 [+0.018, +0.047], có ý nghĩa | +0.001 [−0.010, +0.012], nhiễu |
| B5 | +0.265 [+0.232, +0.295], có ý nghĩa | +0.011 [−0.003, +0.025], nhiễu |

- Khi lấy mẫu, B7 được nDCG@10 0.522, recall@10 0.766, MRR 0.456. MF được 0.500, BPR 0.489.
- Trên toàn danh mục, chênh lệch nDCG@10 với MF, BPR, B5 đều có CI chứa 0. Câu "B7 hơn BPR" trong
  báo cáo phải ghi kèm giao thức lấy mẫu.
- Bỏ 23 user có phim test mới, B7 vẫn hơn BPR +0.027 [+0.014, +0.041]. So với MF thì thành nhiễu:
  +0.015 [−0.006, +0.036].

### AUC tăng chủ yếu nhờ 23 user có phim test mới

B7 hơn BPR về AUC +0.031 [+0.021, +0.043] trên 610 user. Bỏ 23 user có phim test mới, chênh lệch
còn +0.0057 [+0.0041, +0.0075]. Ở 23 user đó BPR chấm phim đúng `-inf`, nên AUC gần 0. B7 chấm
được phim đó bằng nhánh nội dung.

So với MF, B7 hơn về AUC +0.056 trên 610 user, +0.032 khi bỏ nhóm này. BPR vốn đã hơn MF +0.025
về AUC (docs/so-lieu-b4.md).

### 23 user có phim test chưa từng xuất hiện trong train (checklist 2.6)

Cả 23 phim test đi qua nhánh nội dung. B7 trúng top-10 cho 7/23 user khi lấy mẫu, B5 trúng 5/23.
Trên toàn danh mục, B7 trúng 2/23, B5 trúng 1/23. B0-B4 trúng 0/23 ở cả hai giao thức. Đây là số
đếm trên 23 user, chưa có CI.

### Độ phủ: B7 ngang BPR, và đưa được phim mới lên giao diện

Cùng mẫu số 9.742 phim cho mọi mô hình. Đây là số đếm, chưa có CI.

- Top-10 của B7 có 958 phim khác nhau (9,8% danh mục). BPR có 884 (9,1%), MF có 976 (10,0%).
- 43,2% phim B7 gợi ý thuộc top-100 phổ biến, BPR là 48,5%. Lượt xem trung vị 101, BPR 108.
- 26/41 phim chưa ai xem lọt vào top-10 của ít nhất một user. B0-B5 không đưa được phim nào trong
  số đó lên, vì `recommend()` của hợp đồng chỉ duyệt phim có trong train.
- Phim mới chiếm 2,07% số chỗ trong top-10, trong khi chỉ là 0,4% danh mục. Tức B7 đưa phim mới
  lên nhiều gấp khoảng 5 lần tỉ lệ của chúng. Chưa đo nguyên nhân.

### Phim mới: B7 hơn cả B5, nhưng con số này lẫn hai hiệu ứng

| Mô hình | nDCG@10, lấy mẫu | nDCG@10, toàn danh mục |
|---|---|---|
| Popularity, BPR | 0.000 | 0.000 |
| B5 | 0.252 | 0.022 |
| B7 | 0.329 | 0.052 |

B7 trừ B5: +0.077 [+0.063, +0.092] khi lấy mẫu, +0.030 [+0.021, +0.039] trên toàn danh mục, đều có
ý nghĩa.

- Giữa các phim mới với nhau, B7 xếp đúng như B5, vì cả hai dùng cùng điểm nội dung. Phần hơn đến
  từ chỗ phim cũ bị xếp thấp hơn so với phim mới.
- Con số này vì vậy đo chung hai điều: B7 tìm được phim mới, và B7 đưa phim mới lên nhiều hơn tỉ
  lệ của chúng (mục trên). Báo cáo không viết "nhánh nội dung của B7 tốt hơn B5".
- Trên toàn danh mục, mỗi lần xóa có 143-150 phim mới cạnh tranh với phim đúng. Kết quả không đến
  từ chỗ phim đúng là phim mới duy nhất.

### Câu giải thích lấy từ nhánh đóng góp nhiều hơn vào điểm

- Phim đã có người xem: với α_u ≤ 0.1, phần nội dung chỉ lớn hơn phần BPR khi phân vị BPR thấp hơn
  1/9 phân vị nội dung. Cả hai ví dụ nhánh trộn ở mục 6 đều lấy câu của BPR, tức giải thích hậu
  kiểm, không phân rã điểm (xem B4).
- Phim mới lấy câu của B5, là phân rã chính xác, thêm tiền tố "Phim moi, chua ai danh gia".
- User chưa có lịch sử nhận câu của Popularity.

---

## Số liệu chi tiết

ml-latest-small · seed 42 · 610 user kiểm tra · BPR, B5 theo hằng số bpr.py, content.py


### 1. Chọn cách chuẩn hóa trên validation, mọi user cùng α (ndcg@10)

Chọn theo ndcg@10 lấy mẫu, như mọi bậc trước. Cột toàn danh mục chỉ để tham khảo. Ô đậm là tốt nhất của từng cột.

| chuẩn hóa | alpha | ndcg@10 lấy mẫu | ndcg@10 toàn danh mục | coverage@10 |
|---|---|---|---|---|
| rank | 0 | 0.4991 | 0.0298 | 0.0923 |
| rank | 0.05 | 0.5124 | 0.0304 | 0.0943 |
| rank | 0.1 | **0.5161** | 0.0319 | 0.0968 |
| rank | 0.2 | 0.5078 | 0.0330 | 0.1005 |
| rank | 0.3 | 0.4948 | 0.0317 | 0.1065 |
| rank | 0.5 | 0.4635 | 0.0326 | 0.1159 |
| rank | 0.7 | 0.4190 | 0.0305 | 0.1316 |
| rank | 1 | 0.2571 | 0.0266 | 0.2818 |
| z | 0 | 0.4958 | 0.0275 | 0.0915 |
| z | 0.05 | 0.5083 | 0.0327 | 0.0954 |
| z | 0.1 | 0.5117 | **0.0389** | 0.1153 |
| z | 0.2 | 0.5099 | 0.0341 | 0.2012 |
| z | 0.3 | 0.5000 | 0.0315 | 0.2568 |
| z | 0.5 | 0.4741 | 0.0282 | 0.2822 |
| z | 0.7 | 0.4171 | 0.0268 | **0.2833** |
| z | 1 | 0.2571 | 0.0266 | 0.2818 |

Tham chiếu cùng validation, ndcg@10 lấy mẫu: BPR 0.4928, B5 0.2571.

Chọn: chuẩn hóa rank, α chung 0.1, ndcg@10 validation 0.5161. Cách chuẩn hóa này giữ cho mục 1b.

### 1b. α theo số tương tác, chọn trên validation có cắt lịch sử (ndcg@10)

Chuẩn hóa rank. Mỗi kịch bản cắt lịch sử của từng 1/4 số user validation xuống n tương tác gần nhất, như mục 5 nhưng trên validation. Chọn theo cột trung bình (5 kịch bản, trọng số bằng nhau). Ô đậm là tốt nhất của từng cột.

Dạng hàm: `hang` = mọi user cùng α; `nguong m` = α bằng 0 khi dưới m tương tác; `co c` = α · n / (n + c).

| α | dạng | 1 tương tác | 3 tương tác | 5 tương tác | 10 tương tác | đầy đủ | trung bình |
|---|---|---|---|---|---|---|---|
| 0.05 | hang | 0.4053 | 0.4695 | 0.4971 | 0.5094 | 0.5124 | 0.4788 |
| 0.05 | nguong 3 | 0.4082 | 0.4695 | 0.4971 | 0.5094 | 0.5124 | 0.4793 |
| 0.05 | nguong 5 | 0.4082 | **0.4714** | 0.4971 | 0.5094 | 0.5124 | 0.4797 |
| 0.05 | nguong 10 | 0.4082 | **0.4714** | 0.5019 | 0.5094 | 0.5124 | 0.4807 |
| 0.05 | nguong 20 | 0.4082 | **0.4714** | 0.5019 | 0.5060 | 0.5119 | 0.4799 |
| 0.05 | co 1 | 0.4143 | 0.4657 | 0.4990 | 0.5109 | 0.5130 | 0.4806 |
| 0.05 | co 3 | **0.4155** | 0.4619 | 0.4994 | **0.5117** | 0.5129 | 0.4803 |
| 0.05 | co 10 | 0.4117 | 0.4668 | 0.5013 | 0.5112 | 0.5126 | 0.4807 |
| 0.05 | co 30 | 0.4099 | 0.4695 | **0.5048** | 0.5094 | 0.5120 | 0.4811 |
| 0.1 | hang | 0.3926 | 0.4649 | 0.4880 | 0.4985 | 0.5161 | 0.4720 |
| 0.1 | nguong 3 | 0.4082 | 0.4649 | 0.4880 | 0.4985 | 0.5161 | 0.4752 |
| 0.1 | nguong 5 | 0.4082 | **0.4714** | 0.4880 | 0.4985 | 0.5161 | 0.4765 |
| 0.1 | nguong 10 | 0.4082 | **0.4714** | 0.5019 | 0.4985 | 0.5161 | 0.4792 |
| 0.1 | nguong 20 | 0.4082 | **0.4714** | 0.5019 | 0.5060 | 0.5159 | 0.4807 |
| 0.1 | co 1 | 0.4053 | 0.4645 | 0.4938 | 0.5018 | 0.5165 | 0.4764 |
| 0.1 | co 3 | 0.4143 | 0.4695 | 0.4955 | 0.5061 | 0.5181 | 0.4807 |
| 0.1 | co 10 | 0.4149 | 0.4608 | 0.4982 | 0.5094 | 0.5175 | 0.4802 |
| 0.1 | co 30 | 0.4102 | 0.4668 | 0.5028 | 0.5112 | 0.5173 | **0.4816** |
| 0.2 | hang | 0.3760 | 0.4470 | 0.4745 | 0.4793 | 0.5078 | 0.4569 |
| 0.2 | nguong 3 | 0.4082 | 0.4470 | 0.4745 | 0.4793 | 0.5078 | 0.4634 |
| 0.2 | nguong 5 | 0.4082 | **0.4714** | 0.4745 | 0.4793 | 0.5078 | 0.4682 |
| 0.2 | nguong 10 | 0.4082 | **0.4714** | 0.5019 | 0.4793 | 0.5078 | 0.4737 |
| 0.2 | nguong 20 | 0.4082 | **0.4714** | 0.5019 | 0.5060 | 0.5094 | 0.4794 |
| 0.2 | co 1 | 0.3926 | 0.4508 | 0.4783 | 0.4832 | 0.5081 | 0.4626 |
| 0.2 | co 3 | 0.4053 | 0.4649 | 0.4838 | 0.4920 | 0.5094 | 0.4711 |
| 0.2 | co 10 | 0.4133 | 0.4703 | 0.4943 | 0.4985 | 0.5127 | 0.4779 |
| 0.2 | co 30 | 0.4131 | 0.4634 | 0.4988 | 0.5094 | **0.5200** | 0.4809 |
| 0.3 | hang | 0.3508 | 0.4229 | 0.4543 | 0.4643 | 0.4948 | 0.4374 |
| 0.3 | nguong 3 | 0.4082 | 0.4229 | 0.4543 | 0.4643 | 0.4948 | 0.4489 |
| 0.3 | nguong 5 | 0.4082 | **0.4714** | 0.4543 | 0.4643 | 0.4948 | 0.4586 |
| 0.3 | nguong 10 | 0.4082 | **0.4714** | 0.5019 | 0.4643 | 0.4948 | 0.4681 |
| 0.3 | nguong 20 | 0.4082 | **0.4714** | 0.5019 | 0.5060 | 0.4976 | 0.4770 |
| 0.3 | co 1 | 0.3814 | 0.4393 | 0.4671 | 0.4688 | 0.4957 | 0.4505 |
| 0.3 | co 3 | 0.3982 | 0.4508 | 0.4764 | 0.4733 | 0.4956 | 0.4589 |
| 0.3 | co 10 | 0.4139 | 0.4666 | 0.4880 | 0.4929 | 0.5014 | 0.4726 |
| 0.3 | co 30 | 0.4147 | 0.4619 | 0.4979 | 0.5061 | 0.5097 | 0.4781 |
| BPR | | 0.4166 | 0.4812 | 0.5113 | 0.5138 | 0.4928 | 0.4831 |

Chọn: α 0.1, co 30, trung bình 0.4816.

### 2. Bảng chính trên test (1 phim đúng + 99 phim âm, CI 95%)

| Mô hình | recall@10 | ndcg@10 | AUC | coverage@10 | ứng viên −inf |
|---|---|---|---|---|---|
| B0-Popularity | 0.5967 [0.5573, 0.6361] | 0.3590 [0.3302, 0.3879] | 0.8315 [0.8110, 0.8501] | 0.0126 | 0.3% |
| B3-MF | 0.7279 [0.6918, 0.7639] | 0.5003 [0.4701, 0.5299] | 0.8605 [0.8386, 0.8802] | 0.1006 | 0.3% |
| B4-BPR | 0.7328 [0.6984, 0.7672] | 0.4886 [0.4590, 0.5176] | 0.8858 [0.8678, 0.9020] | 0.0911 | 0.3% |
| B5-Content | 0.4869 [0.4475, 0.5279] | 0.2563 [0.2319, 0.2812] | 0.8052 [0.7888, 0.8222] | 0.2788 | 0.6% |
| B7 α chung | 0.7721 [0.7377, 0.8033] | 0.5273 [0.4970, 0.5564] | 0.9178 [0.9053, 0.9292] | 0.0987 | 0.0% |
| B7-Hybrid | 0.7656 [0.7311, 0.7984] | 0.5215 [0.4910, 0.5490] | 0.9170 [0.9046, 0.9283] | 0.0983 | 0.0% |

Coverage của B7 chia cho 9,742 phim (danh mục đã mở rộng), của các mô hình khác chia cho 9,701 phim có trong train.

#### Chênh lệch ghép cặp: lấy mẫu và toàn danh mục

| So sánh | độ đo | lấy mẫu | toàn danh mục |
|---|---|---|---|
| B7-Hybrid − B0-Popularity | ndcg@10 | +0.1625 [+0.1379, +0.1876] **có ý nghĩa** | +0.0193 [+0.0071, +0.0314] **có ý nghĩa** |
| B7-Hybrid − B3-MF | ndcg@10 | +0.0212 [+0.0006, +0.0421] **có ý nghĩa** | +0.0035 [-0.0092, +0.0153] nhiễu |
| B7-Hybrid − B4-BPR | ndcg@10 | +0.0328 [+0.0175, +0.0474] **có ý nghĩa** | +0.0012 [-0.0098, +0.0118] nhiễu |
| B7-Hybrid − B5-Content | ndcg@10 | +0.2651 [+0.2318, +0.2951] **có ý nghĩa** | +0.0107 [-0.0033, +0.0250] nhiễu |
| B7-Hybrid − B7 α chung | ndcg@10 | -0.0059 [-0.0134, +0.0017] nhiễu | -0.0027 [-0.0076, +0.0018] nhiễu |
| B7-Hybrid − B0-Popularity | recall@10 | +0.1689 [+0.1344, +0.2049] **có ý nghĩa** | +0.0443 [+0.0197, +0.0672] **có ý nghĩa** |
| B7-Hybrid − B3-MF | recall@10 | +0.0377 [+0.0098, +0.0656] **có ý nghĩa** | +0.0115 [-0.0131, +0.0344] nhiễu |
| B7-Hybrid − B4-BPR | recall@10 | +0.0328 [+0.0164, +0.0508] **có ý nghĩa** | +0.0131 [-0.0082, +0.0328] nhiễu |
| B7-Hybrid − B5-Content | recall@10 | +0.2787 [+0.2328, +0.3230] **có ý nghĩa** | +0.0410 [+0.0148, +0.0672] **có ý nghĩa** |
| B7-Hybrid − B7 α chung | recall@10 | -0.0066 [-0.0180, +0.0049] nhiễu | -0.0066 [-0.0197, +0.0049] nhiễu |
| B7-Hybrid − B0-Popularity | auc | +0.0855 [+0.0716, +0.1004] **có ý nghĩa** | +0.0840 [+0.0699, +0.0986] **có ý nghĩa** |
| B7-Hybrid − B3-MF | auc | +0.0564 [+0.0412, +0.0723] **có ý nghĩa** | +0.0565 [+0.0415, +0.0725] **có ý nghĩa** |
| B7-Hybrid − B4-BPR | auc | +0.0312 [+0.0207, +0.0429] **có ý nghĩa** | +0.0303 [+0.0200, +0.0419] **có ý nghĩa** |
| B7-Hybrid − B5-Content | auc | +0.1118 [+0.0947, +0.1293] **có ý nghĩa** | +0.1128 [+0.0958, +0.1297] **có ý nghĩa** |
| B7-Hybrid − B7 α chung | auc | -0.0009 [-0.0016, -0.0001] **có ý nghĩa** | -0.0007 [-0.0012, -0.0002] **có ý nghĩa** |

Toàn danh mục, ndcg@10: B0-Popularity 0.0182 [0.0109, 0.0264], B3-MF 0.0340 [0.0244, 0.0454], B4-BPR 0.0363 [0.0252, 0.0481], B5-Content 0.0268 [0.0161, 0.0382], B7 α chung 0.0402 [0.0298, 0.0523], B7-Hybrid 0.0375 [0.0272, 0.0487]

#### 23 user có phim test chưa từng xuất hiện trong train (checklist 2.6)

Nhánh B7 chấm phim test của nhóm này: noi_dung 23.

Số user trúng top-10 trong nhóm này:

| Mô hình | lấy mẫu | toàn danh mục |
|---|---|---|
| B0-Popularity | 0/23 | 0/23 |
| B3-MF | 0/23 | 0/23 |
| B4-BPR | 0/23 | 0/23 |
| B5-Content | 5/23 | 1/23 |
| B7 α chung | 7/23 | 2/23 |
| B7-Hybrid | 7/23 | 2/23 |

Trên 587 user còn lại (phim test đã có người xem), lấy mẫu:

- B7-Hybrid − B3-MF, ndcg@10: +0.0148 [-0.0055, +0.0363] nhiễu
- B7-Hybrid − B4-BPR, ndcg@10: +0.0269 [+0.0137, +0.0411] **có ý nghĩa**
- B7-Hybrid − B5-Content, ndcg@10: +0.2725 [+0.2392, +0.3054] **có ý nghĩa**
- B7-Hybrid − B7 α chung, ndcg@10: -0.0059 [-0.0139, +0.0020] nhiễu
- B7-Hybrid − B3-MF, auc: +0.0320 [+0.0204, +0.0437] **có ý nghĩa**
- B7-Hybrid − B4-BPR, auc: +0.0057 [+0.0041, +0.0075] **có ý nghĩa**
- B7-Hybrid − B5-Content, auc: +0.1154 [+0.0973, +0.1336] **có ý nghĩa**
- B7-Hybrid − B7 α chung, auc: -0.0009 [-0.0017, -0.0001] **có ý nghĩa**

### 3. Độ phủ và thiên lệch phổ biến của top-10 (mọi user)

Phủ danh mục tính trên cùng mẫu số 9,742 phim cho mọi mô hình. Có 41 phim chưa ai xem trong train.

| Mô hình | số phim khác nhau | phủ danh mục | lượt xem trung vị | tỉ lệ thuộc top-100 phổ biến | tỉ lệ phim mới | số phim mới khác nhau |
|---|---|---|---|---|---|---|
| B0-Popularity | 122 | 1.3% | 218 | 99.3% | 0.00% | 0 |
| B3-MF | 976 | 10.0% | 109 | 48.3% | 0.00% | 0 |
| B4-BPR | 884 | 9.1% | 108 | 48.5% | 0.00% | 0 |
| B5-Content | 2705 | 27.8% | 7 | 2.7% | 0.00% | 0 |
| B7 α chung | 962 | 9.9% | 98 | 41.7% | 2.26% | 26 |
| B7-Hybrid | 958 | 9.8% | 101 | 43.2% | 2.07% | 26 |

### 4. Phim mới: phim test bị xóa khỏi train (item cold-start)

Mỗi lần lấy 1/4 số user, xóa mọi tương tác train với phim test của họ (tổng 31,584 dòng qua 4 lần), rồi đo trên chính các user đó. Toàn danh mục: phim đúng cạnh tranh với mọi phim mới khác của cùng lần xóa. Đủ 610 user, CI 95%.

| Mô hình | recall@10 lấy mẫu | ndcg@10 lấy mẫu | ndcg@10 toàn danh mục |
|---|---|---|---|
| B0-Popularity | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| B4-BPR | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| B5-Content | 0.4836 [0.4443, 0.5230] | 0.2519 [0.2268, 0.2765] | 0.0221 [0.0129, 0.0326] |
| B7-Hybrid | 0.5213 [0.4836, 0.5623] | 0.3287 [0.2993, 0.3589] | 0.0522 [0.0368, 0.0668] |

- B7 − B4-BPR, ndcg@10 lấy mẫu: +0.3287 [+0.2993, +0.3589] **có ý nghĩa**
- B7 − B4-BPR, ndcg@10 toàn danh mục: +0.0522 [+0.0368, +0.0668] **có ý nghĩa**
- B7 − B5-Content, ndcg@10 lấy mẫu: +0.0769 [+0.0626, +0.0921] **có ý nghĩa**
- B7 − B5-Content, ndcg@10 toàn danh mục: +0.0301 [+0.0208, +0.0393] **có ý nghĩa**

### 5. User mới: cắt còn n tương tác gần nhất (như checklist 2.8)

ndcg@10, CI 95%, đủ 610 user. 0 tương tác = user bị xóa khỏi train, BPR khi đó chấm −inf mọi phim và xếp hạng là xáo ngẫu nhiên. B7 α chung: cùng lần fit, mọi user dùng α 0.1 (mục 1).

| giữ lại | B0-Popularity | B4-BPR | B7 α chung | B7-Hybrid | B7 − Popularity | B7 − BPR | B7 − B7 α chung |
|---|---|---|---|---|---|---|---|
| 0 tương tác | 0.3610 [0.3325, 0.3890] | 0.0442 [0.0328, 0.0569] | 0.3610 [0.3325, 0.3890] | 0.3610 [0.3325, 0.3890] | +0.0000 [+0.0000, +0.0000] nhiễu | +0.3168 [+0.2860, +0.3476] **có ý nghĩa** | +0.0000 [+0.0000, +0.0000] nhiễu |
| 1 tương tác | 0.3610 [0.3326, 0.3891] | 0.4367 [0.4082, 0.4662] | 0.4063 [0.3787, 0.4353] | 0.4264 [0.3978, 0.4560] | +0.0653 [+0.0407, +0.0914] **có ý nghĩa** | -0.0103 [-0.0182, -0.0023] **có ý nghĩa** | +0.0200 [+0.0044, +0.0361] **có ý nghĩa** |
| 3 tương tác | 0.3613 [0.3328, 0.3894] | 0.4656 [0.4359, 0.4946] | 0.4441 [0.4156, 0.4723] | 0.4564 [0.4272, 0.4853] | +0.0951 [+0.0674, +0.1228] **có ý nghĩa** | -0.0092 [-0.0183, +0.0004] nhiễu | +0.0123 [-0.0026, +0.0267] nhiễu |
| 5 tương tác | 0.3600 [0.3315, 0.3883] | 0.4908 [0.4609, 0.5211] | 0.4830 [0.4532, 0.5135] | 0.4865 [0.4569, 0.5164] | +0.1265 [+0.0995, +0.1560] **có ý nghĩa** | -0.0043 [-0.0138, +0.0048] nhiễu | +0.0035 [-0.0106, +0.0176] nhiễu |
| 10 tương tác | 0.3612 [0.3320, 0.3891] | 0.5035 [0.4743, 0.5330] | 0.4942 [0.4657, 0.5228] | 0.4979 [0.4682, 0.5272] | +0.1367 [+0.1098, +0.1637] **có ý nghĩa** | -0.0056 [-0.0160, +0.0044] nhiễu | +0.0037 [-0.0087, +0.0164] nhiễu |
| đầy đủ | 0.3590 [0.3302, 0.3879] | 0.4886 [0.4590, 0.5176] | 0.5273 [0.4970, 0.5564] | 0.5215 [0.4910, 0.5490] | +0.1625 [+0.1379, +0.1876] **có ý nghĩa** | +0.0328 [+0.0175, +0.0474] **có ý nghĩa** | -0.0059 [-0.0134, +0.0017] nhiễu |

### 6. Ví dụ câu giải thích thật, mỗi nhánh một ví dụ

- **User 1**, nhánh trộn → *Back to the Future Part II (1989)*: "Giong cac phim ban da xem: Back to the Future Part III (1990), Indiana Jones and the Temple of Doom (1984), Back to the Future (1985)"
- **User 2**, nhánh trộn → *Argo (2012)*: "Giong cac phim ban da xem: Interstellar (2014), Django Unchained (2012), Girl with the Dragon Tattoo, The (2011)"
- **User 5**, phim chưa ai xem → *Jack Reacher: Never Go Back (2016)*: "Phim moi, chua ai danh gia. Vi ban hay xem phim co: Edward Zwick, investigation, intense (giong nhat: Legends of the Fall (1994), In the Line of Fire (1993))"
- **User chưa có lịch sử** → *Forrest Gump (1994)*: "Ban chua co lich su xem. Phim pho bien -- 328 nguoi da danh gia"

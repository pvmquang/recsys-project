# Số liệu và câu cần sửa cho slide

Dành cho người làm slide. Số chạy sau khi chốt giao thức đợt 2 (checklist 2.1-2.3): phá hòa
timestamp theo seed, k chọn trên validation (ItemKNN k = 80, UserKNN k = 40). **Mọi số đã đổi
so với bản trước**, kể cả bảng chính. Các mục S1-S5 tương ứng [checklist.md](checklist.md).

Phần "Sửa gì trên từng slide" dưới đây là gợi ý. Phần "Số liệu chi tiết" là output nguyên
văn của `python scripts/so_lieu_slide.py`. Lấy số từ đó, đừng lấy từ bản cũ của file này.

> **Quy tắc:** chênh lệch nào có CI 95% chứa 0 thì không viết "mô hình X thắng".

---

## Sửa gì trên từng slide

### Slide 4 và 8: "mô hình không nhìn thấy tương lai" (S2)

- Đổi thành: **"mô hình không nhìn thấy tương lai *của chính người dùng đó*"**.
- Thêm cách chọn siêu tham số: *"chọn trên tập validation (tương tác áp chót), không chọn
  trên tập kiểm tra"*.
- Gọi đúng tên bài toán: *"dự đoán tương tác tiếp theo"*. 16,2% phim kiểm tra bị chấm
  dưới 3 sao, nên không viết "dự đoán phim người dùng thích".
- Nếu có chỗ, thêm hạn chế: *"train vẫn chứa tương tác về sau của người dùng khác"*.

### Slide 7: thí nghiệm theo nhóm user (S3)

Bảng cũ có nhóm "5-20 tương tác" thực chất là 29 user có 19-20 tương tác, chênh lệch là
nhiễu. Kết luận "UserKNN tốt hơn ở user ít lịch sử" từ bảng đó phải bỏ.

Bảng mới (3 nhóm theo phân vị, mỗi nhóm ~200 user):
- ItemKNN thắng UserKNN ở cả 3 nhóm, chênh lệch có ý nghĩa.
- Popularity thấp nhất ở mọi nhóm.
- Mọi mô hình kém nhất ở nhóm nhiều lịch sử nhất (127+ tương tác).

Ghi chú bắt buộc trên slide: *nhóm thưa nhất vẫn có ≥ 19 tương tác, đây không phải
cold-start*.

Nếu slide 5 hoặc 7 cần căn cứ cho B7, lấy từ cold-start thật ở
[so-lieu-dot-2.md](so-lieu-dot-2.md): với 1 tương tác Popularity tốt nhất; với 3 và 5 tương tác
UserKNN hơn ItemKNN; với lịch sử đầy đủ ItemKNN tốt nhất.

### Slide 10: thay [Phim A], [Phim B] (S4)

Dùng một trong ba ví dụ thật ở mục "Ví dụ câu giải thích thật" cuối file. Ví dụ dễ hiểu nhất
là User 1 → *Aliens (1986)* vì đã xem *Alien (1979)*, *The Terminator (1984)*, *Star Wars V*.

Câu do code sinh ra đang không dấu (theo quy ước code). Trên slide viết lại có dấu, ví dụ:
*"Vì bạn đã đánh giá 48 phim tương tự (gần nhất: Alien, The Terminator, Star Wars V)"*.
Tên phim MovieLens ghi kiểu "Terminator, The (1984)", viết lại thành "The Terminator".

### Slide 11: bảng kết quả (S5)

1. **Thay cả bảng** bằng bảng ở mục "Bảng chính". Cột coverage ghi rõ **coverage@10**. Ghi k
   cạnh tên mô hình: ItemKNN (k = 80), UserKNN (k = 40).
2. Ghi giao thức ngay dưới bảng: *"1 phim đúng + 99 phim âm lấy mẫu"*. Trên toàn danh mục,
   chênh lệch nhỏ hơn nhiều (xem [so-lieu-dot-2.md](so-lieu-dot-2.md)).
3. "Khung đo tính đúng" đổi thành "Kiểm tra bằng Random không phát hiện lỗi".
4. Số của Random: trung bình 10 seed, recall@10 = 0.101 ± 0.012, AUC = 0.503 ± 0.012
   (kỳ vọng 0.10 và 0.50).
5. **Bỏ mọi câu so sánh AUC.** Cả ba cặp đều là nhiễu. AUC ở giao thức này chủ yếu đo việc
   mô hình có chấm được phim đúng hay không.
6. Câu nên viết (đều có ý nghĩa thống kê):
   - ItemKNN hơn Popularity +0.117 nDCG@10 (+33%) và phủ danh mục gấp 6 lần (7,5% so với 1,26%).
   - ItemKNN hơn UserKNN +0.055 nDCG@10 và +0.036 recall@10.
   - Cả hai KNN đều hơn Popularity về nDCG@10 và recall@10.

### Slide 6: "Chín bậc"

Slide đúng (B0-B8 là 9 bậc).

---

## Số liệu chi tiết

ml-latest-small · seed 42 · k láng giềng: ItemKNN 80, UserKNN 40 (chọn trên validation) · 610 user kiểm tra · 1 phim đúng + 99 phim âm

### Bảng chính (CI 95% bootstrap theo user)

| Mô hình | recall@10 | ndcg@10 | MRR | AUC | coverage@10 | coverage@20 |
|---|---|---|---|---|---|---|
| B0-Random | 0.1180 [0.0934, 0.1443] | 0.0538 [0.0415, 0.0673] | 0.0561 | 0.4956 [0.4729, 0.5187] | 0.4664 | 0.7136 |
| B0-Popularity | 0.5967 [0.5573, 0.6361] | 0.3590 [0.3302, 0.3879] | 0.3019 | 0.8315 [0.8110, 0.8501] | 0.0126 | 0.0198 |
| B1-ItemKNN | 0.6803 [0.6410, 0.7148] | 0.4761 [0.4436, 0.5064] | 0.4225 | 0.8342 [0.8135, 0.8524] | 0.0749 | 0.1041 |
| B2-UserKNN | 0.6443 [0.6066, 0.6803] | 0.4214 [0.3912, 0.4505] | 0.3647 | 0.8310 [0.8113, 0.8480] | 0.0288 | 0.0444 |

### B0-Random qua 10 seed (kỳ vọng: recall@10 = 0.10, AUC = 0.50)

| | trung bình | độ lệch chuẩn | nhỏ nhất | lớn nhất |
|---|---|---|---|---|
| recall@10 | 0.1013 | 0.0121 | 0.0869 | 0.1246 |
| AUC | 0.5031 | 0.0119 | 0.4856 | 0.5171 |
| coverage@10 | 0.4671 | 0.0029 | 0.4637 | 0.4741 |

### Chênh lệch giữa các mô hình (ghép cặp theo user)

| So sánh | độ đo | chênh lệch | CI 95% | kết luận |
|---|---|---|---|---|
| B1-ItemKNN − B0-Popularity | ndcg@10 | +0.1171 | [+0.0970, +0.1388] | **có ý nghĩa** |
| B2-UserKNN − B0-Popularity | ndcg@10 | +0.0624 | [+0.0440, +0.0815] | **có ý nghĩa** |
| B1-ItemKNN − B2-UserKNN | ndcg@10 | +0.0547 | [+0.0343, +0.0738] | **có ý nghĩa** |
| B1-ItemKNN − B0-Popularity | recall@10 | +0.0836 | [+0.0541, +0.1131] | **có ý nghĩa** |
| B2-UserKNN − B0-Popularity | recall@10 | +0.0475 | [+0.0197, +0.0755] | **có ý nghĩa** |
| B1-ItemKNN − B2-UserKNN | recall@10 | +0.0361 | [+0.0066, +0.0672] | **có ý nghĩa** |
| B1-ItemKNN − B0-Popularity | auc | +0.0027 | [-0.0129, +0.0183] | nhiễu (CI chứa 0) |
| B2-UserKNN − B0-Popularity | auc | -0.0005 | [-0.0136, +0.0139] | nhiễu (CI chứa 0) |
| B1-ItemKNN − B2-UserKNN | auc | +0.0032 | [-0.0128, +0.0193] | nhiễu (CI chứa 0) |

### Theo nhóm user (phân vị số tương tác trong train) — ndcg@10

⚠ Nhóm thưa nhất vẫn có ≥ 19 tương tác: đây **không** phải phân tích cold-start.

| Nhóm (số tương tác) | số user | B0-Popularity | B1-ItemKNN | B2-UserKNN | ItemKNN − UserKNN, CI 95% |
|---|---|---|---|---|---|
| 19-44 | 209 | 0.4181 | 0.5612 | 0.5043 | +0.0570 [+0.0157, +0.0963] **có ý nghĩa** |
| 45-126 | 198 | 0.3989 | 0.5321 | 0.4695 | +0.0627 [+0.0281, +0.0982] **có ý nghĩa** |
| 127-2697 | 203 | 0.2591 | 0.3337 | 0.2892 | +0.0445 [+0.0152, +0.0759] **có ý nghĩa** |

### k chọn trên validation so với k = 40 cũ (ItemKNN, ndcg@10, tập test)

| So sánh | chênh lệch | CI 95% | kết luận |
|---|---|---|---|
| k=80 − k=40 | +0.0073 | [-0.0061, +0.0209] | nhiễu (CI chứa 0) |

### Ví dụ câu giải thích thật (cho slide 10)

- **User 1** → *Aliens (1986)*
  - ItemKNN: "Vi ban da danh gia 48 phim tuong tu (gan nhat: Alien (1979), Terminator, The (1984), Star Wars: Episode V - The Empire Strikes Back (1980))"
  - UserKNN: "19 nguoi co so thich giong ban da danh gia phim nay"
  - Popularity: "Phim pho bien -- 125 nguoi da danh gia"
- **User 2** → *Social Network, The (2010)*
  - ItemKNN: "Vi ban da danh gia 13 phim tuong tu (gan nhat: Inglourious Basterds (2009), Inception (2010), Dark Knight, The (2008))"
  - UserKNN: "12 nguoi co so thich giong ban da danh gia phim nay"
  - Popularity: "Phim pho bien -- 54 nguoi da danh gia"
- **User 3** → *RoboCop (1987)*
  - ItemKNN: "Vi ban da danh gia 5 phim tuong tu (gan nhat: Tron (1982), Highlander (1986), Road Warrior, The (Mad Max 2) (1981))"
  - UserKNN: "5 nguoi co so thich giong ban da danh gia phim nay"
  - Popularity: "Phim pho bien -- 70 nguoi da danh gia"

### Hạn chế của giao thức (cho slide 4 và 8)

- 94/610 user có nhiều phim trùng timestamp cuối (nhiều nhất 34 phim cùng một giây) → phim test chọn ngẫu nhiên theo seed trong nhóm trùng.
- 23/610 phim test chưa từng xuất hiện trong train.
- 16.2% phim test bị user chấm dưới 3 sao (12.6% từ 2 sao trở xuống): bài toán là dự đoán tương tác, không phải dự đoán sở thích.
- Train vẫn chứa tương tác về sau của user KHÁC (ví dụ số đếm của Popularity).

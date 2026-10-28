# Số liệu và câu cần sửa cho slide

Dành cho người làm slide. Số chạy trên nhánh `fix/dot-1-truoc-trinh-bay`, sau khi
sửa xong checklist 1.1–1.7 và 1.9. Các mục S1–S5 tương ứng [checklist.md](checklist.md).

Phần **"Sửa gì trên từng slide"** dưới đây là gợi ý. Phần **"Số liệu chi tiết"** là output
nguyên văn của `python scripts/so_lieu_slide.py` — lấy số từ đó, đừng lấy từ README cũ.

> **Quy tắc:** chênh lệch nào có CI 95% chứa 0 thì **không** viết "mô hình X thắng".

---

## Sửa gì trên từng slide

### Slide 4 và 8 — "mô hình không nhìn thấy tương lai" (S2)

- Đổi thành: **"mô hình không nhìn thấy tương lai *của chính người dùng đó*"**.
- Nếu có chỗ, thêm một dòng hạn chế: *"train vẫn chứa tương tác về sau của người dùng
  khác; 15% người dùng có nhiều phim trùng thời điểm cuối"* (số ở mục "Hạn chế" cuối file).

### Slide 7 — thí nghiệm theo nhóm user (S3)

**Kết luận cũ không còn đúng.** Bảng cũ có nhóm "5–20 tương tác" với 29 user, nên rút ra
"UserKNN/Popularity tốt hơn ở user ít lịch sử → căn cứ cho B7". Thực ra nhóm đó là 29 user có
19–20 tương tác và chênh lệch là nhiễu.

Bảng mới (chia 3 nhóm đều theo phân vị, mỗi nhóm ~200 user) cho thấy:
- **ItemKNN thắng UserKNN ở cả 3 nhóm**, chênh lệch có ý nghĩa.
- **Popularity thấp nhất ở mọi nhóm.**
- **Mọi mô hình kém nhất ở nhóm nhiều lịch sử nhất** (127+ tương tác) — kết luận này vẫn giữ.

Gợi ý tiêu đề: *"ItemKNN thắng ở mọi mức độ hoạt động — nhưng chưa đo được cold-start"*.

Ghi chú bắt buộc trên slide: *nhóm thưa nhất vẫn có ≥ 19 tương tác; cold-start thật sẽ đo bằng
cách cắt bớt lịch sử (đợt 2)*. **Bỏ** việc dùng bảng này làm căn cứ cho B7 và cho kịch bản
"Người dùng mới" ở slide 5.

### Slide 10 — thay [Phim A], [Phim B] (S4)

Dùng một trong ba ví dụ thật ở mục **"Ví dụ câu giải thích thật"** cuối file. Ví dụ dễ hiểu nhất là
User 1 → *Aliens (1986)* vì đã xem *Alien (1979)*, *The Terminator (1984)*, *Star Wars V*.

Câu do code sinh ra đang **không dấu** (theo quy ước code). Trên slide có thể viết lại có dấu,
ví dụ: *"Vì bạn đã đánh giá 30 phim tương tự (gần nhất: Alien, The Terminator, Star Wars V)"*.
Tên phim MovieLens ghi kiểu "Terminator, The (1984)" — viết lại thành "The Terminator" cho dễ đọc.

### Slide 11 — bảng kết quả (S5)

1. **Thay cả bảng** bằng bảng ở mục "Bảng chính". Cột coverage đổi tên thành **coverage@10**.
   Số coverage cũ (0.0199 / 0.1089 / 0.0470) là @20, không dùng nữa.
2. **"Khung đo tính đúng"** → **"Kiểm tra bằng Random không phát hiện lỗi"**.
3. **Số của Random:** dùng trung bình 10 seed — recall@10 = **0.101 ± 0.012**,
   AUC = **0.503 ± 0.012** (kỳ vọng 0.10 và 0.50). Số cũ AUC 0.5258 là do lỗi seed, đã sửa.
4. **Bỏ các câu về AUC**: "UserKNN thắng AUC", "Popularity AUC cao hơn ItemKNN — hay ở toàn
   cục, dở ở đỉnh". Hai chênh lệch so với Popularity đều là nhiễu. Chênh ItemKNN − UserKNN
   có ý nghĩa nhưng nhiều khả năng do cách xử lý thế hòa (88% ứng viên của ItemKNN có điểm
   bằng nhau) — chưa nên đưa lên slide.
5. **Câu nên giữ / nên viết** (đều có ý nghĩa thống kê):
   - ItemKNN hơn Popularity **+0.114 nDCG@10 (+32%)** và phủ danh mục **gấp 6 lần** (7,7% so với 1,25%).
   - ItemKNN hơn UserKNN **+0.051 nDCG@10**.
   - Cả hai KNN đều hơn Popularity về nDCG@10 và recall@10.
   - ItemKNN − UserKNN về **recall@10** là nhiễu — đừng viết "ItemKNN thắng recall".

### Slide 6 — "Chín bậc"

Slide đúng (B0–B8 là 9 bậc). README đã sửa theo.

---

## Số liệu chi tiết
ml-latest-small · seed 42 · k láng giềng = 40 · 610 user kiểm tra · 1 phim đúng + 99 phim âm

### Bảng chính (CI 95% bootstrap theo user)

| Mô hình | recall@10 | ndcg@10 | MRR | AUC | coverage@10 | coverage@20 |
|---|---|---|---|---|---|---|
| B0-Random | 0.1180 [0.0934, 0.1443] | 0.0538 [0.0415, 0.0673] | 0.0561 | 0.4956 [0.4729, 0.5187] | 0.4670 | 0.7138 |
| B0-Popularity | 0.6033 [0.5623, 0.6410] | 0.3603 [0.3317, 0.3891] | 0.3015 | 0.8319 [0.8117, 0.8506] | 0.0125 | 0.0199 |
| B1-ItemKNN | 0.6754 [0.6377, 0.7115] | 0.4744 [0.4422, 0.5053] | 0.4203 | 0.8205 [0.7970, 0.8426] | 0.0768 | 0.1089 |
| B2-UserKNN | 0.6508 [0.6131, 0.6869] | 0.4238 [0.3938, 0.4528] | 0.3662 | 0.8412 [0.8221, 0.8594] | 0.0288 | 0.0449 |

### B0-Random qua 10 seed (kỳ vọng: recall@10 = 0.10, AUC = 0.50)

| | trung bình | độ lệch chuẩn | nhỏ nhất | lớn nhất |
|---|---|---|---|---|
| recall@10 | 0.1013 | 0.0121 | 0.0869 | 0.1246 |
| AUC | 0.5031 | 0.0119 | 0.4856 | 0.5171 |
| coverage@10 | 0.4672 | 0.0026 | 0.4648 | 0.4739 |

### Chênh lệch giữa các mô hình (ghép cặp theo user)

| So sánh | độ đo | chênh lệch | CI 95% | kết luận |
|---|---|---|---|---|
| B1-ItemKNN − B0-Popularity | ndcg@10 | +0.1141 | [+0.0914, +0.1348] | **có ý nghĩa** |
| B2-UserKNN − B0-Popularity | ndcg@10 | +0.0635 | [+0.0452, +0.0826] | **có ý nghĩa** |
| B1-ItemKNN − B2-UserKNN | ndcg@10 | +0.0506 | [+0.0283, +0.0720] | **có ý nghĩa** |
| B1-ItemKNN − B0-Popularity | recall@10 | +0.0721 | [+0.0410, +0.1033] | **có ý nghĩa** |
| B2-UserKNN − B0-Popularity | recall@10 | +0.0475 | [+0.0213, +0.0770] | **có ý nghĩa** |
| B1-ItemKNN − B2-UserKNN | recall@10 | +0.0246 | [-0.0049, +0.0557] | nhiễu (CI chứa 0) |
| B1-ItemKNN − B0-Popularity | auc | -0.0114 | [-0.0300, +0.0080] | nhiễu (CI chứa 0) |
| B2-UserKNN − B0-Popularity | auc | +0.0093 | [-0.0056, +0.0244] | nhiễu (CI chứa 0) |
| B1-ItemKNN − B2-UserKNN | auc | -0.0207 | [-0.0386, -0.0021] | **có ý nghĩa** |

### Theo nhóm user (phân vị số tương tác trong train) — ndcg@10

⚠ Nhóm thưa nhất vẫn có ≥ 19 tương tác: đây **không** phải phân tích cold-start.

| Nhóm (số tương tác) | số user | B0-Popularity | B1-ItemKNN | B2-UserKNN | ItemKNN − UserKNN, CI 95% |
|---|---|---|---|---|---|
| 19-44 | 209 | 0.4241 | 0.5563 | 0.5071 | +0.0492 [+0.0073, +0.0901] **có ý nghĩa** |
| 45-126 | 198 | 0.3955 | 0.5287 | 0.4722 | +0.0564 [+0.0195, +0.0944] **có ý nghĩa** |
| 127-2697 | 203 | 0.2603 | 0.3371 | 0.2908 | +0.0463 [+0.0189, +0.0762] **có ý nghĩa** |

### Số láng giềng k = 40 so với k = 80 (ItemKNN, ndcg@10)

| So sánh | chênh lệch | CI 95% | kết luận |
|---|---|---|---|
| k=80 − k=40 | +0.0008 | [-0.0121, +0.0141] | nhiễu (CI chứa 0) |

### Ví dụ câu giải thích thật (cho slide 10)

- **User 1** → *Aliens (1986)*
  - ItemKNN: "Vi ban da danh gia 30 phim tuong tu (gan nhat: Alien (1979), Terminator, The (1984), Star Wars: Episode V - The Empire Strikes Back (1980))"
  - UserKNN: "18 nguoi co so thich giong ban da danh gia phim nay"
  - Popularity: "Phim pho bien -- 125 nguoi da danh gia"
- **User 2** → *Grand Budapest Hotel, The (2014)*
  - ItemKNN: "Vi ban da danh gia 10 phim tuong tu (gan nhat: Django Unchained (2012), Mad Max: Fury Road (2015), Dark Knight Rises, The (2012))"
  - UserKNN: "9 nguoi co so thich giong ban da danh gia phim nay"
  - Popularity: "Phim pho bien -- 52 nguoi da danh gia"
- **User 3** → *Superman II (1980)*
  - ItemKNN: "Vi ban da danh gia 5 phim tuong tu (gan nhat: Highlander (1986), Star Trek: The Motion Picture (1979), Conan the Barbarian (1982))"
  - UserKNN: "6 nguoi co so thich giong ban da danh gia phim nay"
  - Popularity: "Phim pho bien -- 49 nguoi da danh gia"

### Hạn chế của giao thức (cho slide 4 và 8)

- 94/610 user có nhiều phim trùng timestamp cuối (nhiều nhất 34 phim cùng một giây) → phim test do phá hòa theo item_id.
- 23/610 phim test chưa từng xuất hiện trong train.
- Train vẫn chứa tương tác về sau của user KHÁC (ví dụ số đếm của Popularity).

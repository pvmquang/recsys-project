# Checklist sửa repo và slide

Gộp từ hai nguồn, mỗi việc chỉ ghi một lần:

- **FB**: file feedback của thành viên nhóm, 27/10/2026. Đọc code và slide, chưa chạy lại số.
- **RV**: review ngày 28/10/2026, có chạy lại trên ml-latest-small, seed 42, k=40.

Mọi con số trong file này đều đã được đo lại; chỗ nào chưa kiểm được thì ghi rõ.
Mục slide lấy từ FB vì RV chưa xem được slide.

**Cách dùng:** điền tên vào `@...`, xong thì đánh `[x]` và ghi mã mục (ví dụ `1.3`)
vào commit message. Mục có 🗳️ phải cả nhóm đồng ý trước khi làm, vì đổi số của mọi bậc.

---

## Đợt 1 — Trước buổi trình bày

Các mục này đổi số liệu trên slide. Làm 1.1–1.6 xong mới chạy lại (1.8).

### Repo

- [x] **1.1 Sửa seed của `RandomRec`** · @... · *FB#3a + RV*
  - `default_rng(self.seed + user_id)` làm các seed chồng nhau: user `u` ở seed 1 dùng
    đúng luồng của user `u+1` ở seed 0. Seed 0–5 cho recall@10 = 0.1115 ở 5/6 lần.
  - Với 10 seed độc lập: AUC trung bình **0.4965** (0.479–0.509), recall@10 **0.093**,
    khớp lý thuyết. Vậy AUC 0.5258 là một lần rút xui, **khung đo không sai**.
  - Sửa: `np.random.default_rng([self.seed, int(user_id)])`.

- [x] **1.2 `recommend()` bỏ item điểm `-inf`** · @... · *RV*
  - Một user không có láng giềng UserKNN nhận 20 phim tùy ý, và chúng được đếm vào
    coverage. Sau này sẽ hiện lên giao diện Flask.
  - Đổi hành vi, không đổi giao diện hợp đồng ở `base.py`. Làm trước khi B3 bắt đầu.

- [x] **1.3 Coverage tính cho từng k, bảng chính dùng coverage@10** · @... · *FB#2*
  - Hiện là coverage@20 (`recommend(k=max(ks))`) nhưng đặt cạnh các cột @10.
  - Số @10 hiện tại: Popularity 0.0125 · ItemKNN 0.0768 · UserKNN 0.0298.
    ItemKNN vẫn phủ gấp ~6 lần Popularity.

- [x] **1.4 Bootstrap CI theo user trong runner** · @... · *FB#7 + RV*
  - Số đo hiện tại (95% CI, 610 user):

    | chênh lệch | giá trị | CI | kết luận |
    |---|---|---|---|
    | AUC UserKNN − Popularity | +0.0093 | [−0.0056, +0.0244] | nhiễu |
    | AUC Popularity − ItemKNN | +0.0114 | [−0.0081, +0.0311] | nhiễu |
    | nDCG@10 ItemKNN − UserKNN | +0.0506 | [+0.0293, +0.0716] | **đứng được** |

  - k=40 so với k=80 chênh 0.0008 nDCG — gần như chắc chắn là nhiễu.

- [x] **1.5 Nhóm user chia theo phân vị** · @... · *FB#1 + RV*
  - Nhóm "5–20" thực chất là **29 user có 19–20 tương tác** (mọi user ml-latest-small
    có ≥ 20 rating). Chênh nDCG UserKNN − ItemKNN trong nhóm: +0.028, CI [−0.090, +0.159].
  - Chia 3 nhóm bằng nhau theo số tương tác. Sửa luôn `ks[1]` gán cứng trong
    `_by_activity` (`runner.py`).

- [x] **1.6 `exp_knn.py` lưu CSV cho cả 3 thí nghiệm** · @... · *RV*
  - Hiện chỉ thí nghiệm 3 được ghi file; số thí nghiệm 1–2 trong README là chép tay.

- [x] **1.7 Lỗi chữ** · @... · *FB#8, #10, #11*
  - "tám bậc B0–B8" → "chín bậc": `README.md:47` và docstring `recsys/base.py:3`.
  - Docstring `exp_knn.py`: "Hai thi nghiem" → "Ba thi nghiem".
  - README thêm `source .venv/bin/activate` cho macOS/Linux.

- [x] **1.8 Chạy lại `run_eval.py` + `exp_knn.py`, cập nhật bảng trong README** · @...
  - Sau khi xong 1.1–1.6. Số mới là số dùng cho slide.

- [x] **1.9 `explain()` hiện tên phim** · @... · *FB#9 + RV*
  - Đang in mã phim ("gan nhat: 318, 296"). Cần cho slide 10.
  - Truyền dict `titles` tùy chọn khi tạo mô hình; **không** gọi `load_movies()` bên trong
    mô hình (mô hình sẽ phụ thuộc file/mạng, test chậm).

### Slide

Số liệu mới và gợi ý câu chữ cho từng slide: [so-lieu-slide.md](so-lieu-slide.md).

- [ ] **S1 Điền chỗ trống** · @... · *FB#4*
  - Slide 1: tên nhóm, giảng viên hướng dẫn. Slide 2: họ tên, MSSV, email.

- [ ] **S2 Slide 4 và 8: "không nhìn thấy tương lai"** · @... · *FB#3b + RV*
  - Đổi thành "không nhìn thấy tương lai **của chính người dùng đó**".
  - Lý do: train vẫn chứa tương tác về sau của user khác (ví dụ số đếm của Popularity).

- [ ] **S3 Slide 7: viết lại** · @... · *FB#1*
  - Bỏ kết luận về "user ít lịch sử" — nhóm đó là 19–20 tương tác, 29 user, chênh lệch là nhiễu.
  - Không dùng bảng này làm căn cứ cho B7, hay cho kịch bản "Người dùng mới" ở slide 5.
  - Dùng số mới từ 1.5, hoặc ghi "cold-start sẽ đo ở 2.8".

- [ ] **S4 Slide 10: thay [Phim A], [Phim B]** · @... · *FB#4, #9*
  - Dùng output thật của `explain()` sau 1.9.

- [ ] **S5 Slide 11: bảng kết quả** · @... · *FB#2, #3a + RV*
  - Cột coverage ghi rõ @10, dùng số từ 1.8.
  - "Khung đo tính đúng" → "kiểm tra bằng Random không phát hiện lỗi".
  - Random báo trung bình nhiều seed (sau 1.1).
  - Bỏ các câu "UserKNN thắng AUC", "Popularity AUC cao hơn ItemKNN" (xem 1.4).

---

## Đợt 2 — Trước khi bắt đầu B3

Đổi giao thức chung, nên **mọi số B0–B2 sẽ thay đổi**. Thống nhất xong thì làm
một lần, rồi chạy lại toàn bộ.

- [ ] 🗳️ **2.1 Tách tập validation** · @... · *FB#6*
  - `exp_knn.py` đang chọn k ngay trên test. Từ B3 có nhiều siêu tham số hơn.
  - Đề xuất: validation = tương tác áp chót mỗi user. Chọn siêu tham số trên
    validation, rồi huấn luyện lại trên train + validation và đo trên test.

- [ ] 🗳️ **2.2 Item test có rating thấp** · @... · *RV*
  - 15.4% item test được chấm < 3 sao (11.8% ≤ 2). Mô hình đang được thưởng khi
    gợi ý trúng phim user không thích.
  - Chọn một: chỉ lấy item test ≥ 3.5 sao, hoặc giữ nguyên và ghi rõ bài toán là
    "dự đoán tương tác", không phải "dự đoán sở thích".

- [ ] 🗳️ **2.3 Phá hòa timestamp** · @... · *FB#3b + RV*
  - 94/610 user (15%) có nhiều phim trùng timestamp cuối, nhiều nhất 34 phim cùng
    một giây. Item test khi đó do `item_id` lớn nhất quyết định — thiên về phim mới.
  - Phá hòa ngẫu nhiên theo seed, hoặc ghi rõ thành hạn chế.

- [ ] **2.4 Đánh giá thêm trên toàn danh mục** · @... · *FB#5 + RV*
  - Trung vị độ phổ biến: item đúng 39 lượt, item âm 3 lượt → lấy mẫu đều có lợi cho
    Popularity. Xem Krichene & Rendle, KDD 2020.
  - 610 user × 9.724 phim vẫn chạy nhanh. Bảng cuối kỳ có thêm một bảng này.

- [ ] **2.5 AUC xử lý thế hòa + báo tỉ lệ ứng viên `-inf`** · @... · *RV*
  - Ứng viên điểm `-inf`: ItemKNN 87.9%, UserKNN 71.9%, Popularity 0.3%.
    AUC của KNN chủ yếu đo "item đúng có láng giềng không".
  - Tính AUC từ điểm, cặp bằng nhau tính 0.5.

- [ ] **2.6 Ghi vào báo cáo: 23/610 item test không có trong train** · @... · *RV*
  - Không mô hình lọc cộng tác nào đoán được → recall có trần ~96%.

- [ ] **2.7 Ghi vào báo cáo: rò rỉ giữa các user** · @... · *FB#3b + RV*
  - Leave-one-out theo từng user vẫn để train chứa tương lai của user khác
    (Ji et al., 2020). Phương án mạnh hơn: cắt theo một mốc thời gian chung.

- [ ] **2.8 Đo cold-start thật** · @... · *FB#1*
  - Cắt lịch sử train của một phần user xuống 1, 3, 5 tương tác rồi đo lại.
    Đây mới là căn cứ cho B7.

---

## Đợt 3 — Khi rảnh

- [ ] **3.1 Khử trùng `tmdb_id`** · @... · *RV*
  - `links.csv` có 1 `tmdb_id` trùng → `build_cache` tải 2 lần → `load_metadata`
    nhân đôi dòng. Khử trùng `todo` và `drop_duplicates("tmdb_id")` trong `load_cache`.
- [ ] **3.2 CI GitHub Actions chạy `pytest`** · @... · *RV*
- [ ] **3.3 `pyproject.toml` + `pip install -e .`**, bỏ `sys.path.insert` trong script · @... · *RV*
- [ ] **3.4 README hướng dẫn cài bằng `requirements-lock.txt`** để 3 máy cùng phiên bản · @... · *RV*
- [ ] **3.5 Lưu kết quả kèm seed + git commit hash** · @... · *RV*

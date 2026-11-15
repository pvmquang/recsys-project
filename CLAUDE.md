# CLAUDE.md

Đồ án IT4613: hệ gợi ý phim lai ghép trên MovieLens + TMDB, chín bậc mô hình B0-B8 đo
chung một giao thức. Nhóm 3 người làm song song. Đọc [README.md](README.md) để biết kết
quả hiện có, [docs/checklist.md](docs/checklist.md) để biết việc còn lại.

## Git: người dùng tự commit

Không chạy `git commit`, `git push`, không mở PR. Làm xong thì viết thẳng trong câu trả lời,
cho từng commit:

1. Danh sách file cần `git add`, mỗi commit một mục đích.
2. Message chỉ MỘT dòng, không body, không trailer.

Không ghi message ra file, không đưa lệnh `git commit -F`.

Quy ước message: `type(scope): mô tả` bằng tiếng Việt có dấu, thể mệnh lệnh, tối đa 72 ký
tự, có thể kèm mã mục checklist cuối dòng, ví dụ `(2.1, 2.3)`. Scope đang dùng: `core`,
`data`, `eval`, `models`, `scripts`, `base`.

Làm trên nhánh riêng, PR vào `main`. `graphify-out/` là output của công cụ ngoài, không
đưa vào commit.

## Lệnh

```bash
.venv/Scripts/python -m pytest -q             # toàn bộ test, không gọi mạng, ~2 giây
.venv/Scripts/python scripts/run_eval.py      # bảng B0-B3, thêm --full-catalog để đo toàn danh mục
.venv/Scripts/python scripts/so_lieu_slide.py # số cho slide, kèm CI 95%
.venv/Scripts/python scripts/exp_knn.py       # ba thí nghiệm B1/B2, ~1 phút
.venv/Scripts/python scripts/exp_giao_thuc.py # toàn danh mục, tách AUC, cold-start, ~3 phút
.venv/Scripts/python scripts/exp_mf.py        # B3: chọn siêu tham số trên validation, ~15-20 phút
```

Cài mới: `pip install -r requirements-lock.txt` rồi `pip install -e . --no-deps`.

## Môi trường Windows

- Gọi Python bằng `.venv/Scripts/python`, không dùng `python` hệ thống.
- Script in tiếng Việt có dấu. Khi chuyển hướng output ra file, đặt `PYTHONIOENCODING=utf-8`.
- Heredoc trong Bash làm hỏng ký tự ngoài ASCII (dấu tiếng Việt, `—`). File có ký tự đó
  thì ghi bằng công cụ Write hoặc Edit, không ghi bằng heredoc.

## Quy ước code

- Comment và docstring trong `.py` viết tiếng Việt **không dấu**. README và `docs/` viết có dấu.
- Docstring giải thích vì sao, kèm con số đo thật khi có. Giữ kiểu này khi viết code mới.
- Test đặt tên tiếng Việt không dấu, mô tả hành vi (`test_user_bi_cat_con_dung_n_tuong_tac`).
  Test dùng dữ liệu đồ chơi, không đọc MovieLens, không gọi mạng.

## Ràng buộc không được phá

- Hợp đồng ở [recsys/base.py](recsys/base.py) đã chốt. Mô hình mới chỉ cài `_fit`, `score`,
  `explain`. Không ghi đè `fit()`, không đổi chữ ký các hàm này.
- `score()` trả `-inf` cho item không có bằng chứng, không trả 0.
- Mô hình không tự đọc file hay gọi mạng. Dữ liệu phụ (ví dụ tên phim) truyền qua tham số.
- `seed = 42` cho mọi phép chia. Mọi mô hình đo trên cùng một `Split`.
- Siêu tham số chọn trên `split.validation()` (tương tác áp chót), báo cáo trên test. Không
  chọn trên test. k của KNN nằm ở `K_ITEM`, `K_USER` trong `recsys/models/knn.py`.
  Mô hình mới có siêu tham số thì làm theo cùng cách và ghi giá trị đã chọn vào một hằng số.
- Bài toán là dự đoán tương tác, không phải dự đoán sở thích (16,2% phim test dưới 3 sao).
- Thay đổi nào làm đổi số của mọi bậc (giao thức chia, cách chọn item âm, độ đo) thì hỏi
  người dùng trước, vì cả nhóm phải đồng ý.

## Số liệu

- Số trong README và `docs/` phải lấy từ output của script, không chép từ terminal của một
  lần chạy thử. Phần "Số liệu chi tiết" trong `docs/so-lieu-*.md` là output nguyên văn.
- Chênh lệch có CI 95% chứa 0 thì không viết "mô hình X thắng".
- Bảng chính đo trên 1 phim đúng + 99 phim âm lấy mẫu. Trên toàn danh mục, chênh lệch
  nDCG@10 co lại khoảng 8 lần. Câu nào trích bảng chính phải ghi kèm giao thức.
- Chênh lệch có CI sát 0 có thể lật khi đổi k hay cách phá hòa (đã xảy ra ở đợt 2, xem
  `docs/so-lieu-dot-2.md`). Viết chúng ở dạng "nhỏ, sát ngưỡng", không viết thành kết luận.
- Kiểm tra khung đo bằng B0-Random qua 10 seed: recall@10 phải quanh 0.10, AUC quanh 0.50.
  Sửa code đo mà hai số này lệch thì lỗi nằm ở khung đo.
- Sửa `recsys/eval/` hoặc `recsys/data/split.py` thì chạy lại `so_lieu_slide.py`, so sánh
  với `docs/so-lieu-slide.md` và báo cột nào đổi.
- CSV trong `reports/` có file `.meta.json` đi kèm. `git_dirty: true` nghĩa là số chưa tái
  tạo được từ commit ghi trong file.

## Checklist

Làm xong một mục trong [docs/checklist.md](docs/checklist.md) thì đánh `[x]` và thêm một
dòng "Kết quả:" ghi hàm hoặc script đã thêm và con số chính.

## Văn bản tiếng Việt

README, `docs/` và câu trả lời cho người dùng theo skill `quy-tac-viet-tieng-viet`: câu
đầu nêu ý chính, câu tối đa 25 từ, không gạch ngang dài, không emoji, không tính từ đánh
giá thiếu số liệu.

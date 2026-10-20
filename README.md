# Hệ gợi ý phim lai ghép — đồ án IT4613

Nghiên cứu so sánh các kỹ thuật gợi ý trên MovieLens + TMDB, từ baseline
đến mô hình lai ghép, đo trên cùng một giao thức đánh giá.

## Chạy thử

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

pytest tests/ -q                # 102 test
python scripts/run_eval.py      # tự tải MovieLens ở lần chạy đầu
```

## Kết quả hiện có (ml-latest-small, seed 42, k=40)

| Mô hình | recall@10 | ndcg@10 | MRR | AUC | coverage |
|---|---|---|---|---|---|
| B0-Random | 0.1115 | 0.0511 | 0.0565 | 0.5258 | 0.7173 |
| B0-Popularity | 0.6033 | 0.3603 | 0.3015 | **0.8319** | 0.0199 |
| **B1-ItemKNN** | **0.6754** | **0.4744** | **0.4203** | 0.8205 | 0.1089 |
| B2-UserKNN | 0.6508 | 0.4238 | 0.3662 | **0.8412** | 0.0470 |

**Random rơi đúng vào kỳ vọng lý thuyết** — recall@10 ≈ 10/100 = 0.10 và
AUC ≈ 0.50. Đó là bằng chứng khung đánh giá tính đúng. Nếu sau này sửa gì
mà hai số này lệch đi, lỗi nằm ở khung đo chứ không ở mô hình.

**Cá nhân hóa thắng, và thắng trên CẢ HAI mặt.** ItemKNN hơn Popularity
+32% nDCG@10 *và* phủ danh mục gấp 5,5 lần (10,9% ↔ 2,0%). Ở đây không có
đánh đổi — đó là câu trả lời cho thanh ngang mà Popularity đặt ra.

**Ba độ đo xếp hạng ba mô hình khác nhau** — lý do phải bày cả AUC lẫn nDCG@K:

| | thắng ai | nghĩa là gì |
|---|---|---|
| nDCG@10, MRR | ItemKNN | xếp tốt nhất ở **đỉnh** danh sách |
| AUC | UserKNN | xếp tốt nhất trên **toàn** danh sách |
| coverage | Random | (vô dụng, nhưng là mốc trên) |

Với giao diện chỉ hiện 10 phim thì ItemKNN là lựa chọn đúng. Popularity còn
có AUC (0,8319) **cao hơn** ItemKNN (0,8205) — hay ở toàn cục, dở ở đỉnh.

## Thêm một bậc mô hình

Toàn bộ tám bậc B0–B8 cài đặt chung một hợp đồng ở [`recsys/base.py`](recsys/base.py):

```python
class MyModel(Recommender):
    name = "B3-MF"

    def _fit(self, train): ...                    # train: user_id, item_id, rating, timestamp
    def score(self, user_id, item_ids): ...       # trả mảng điểm, cao = gợi ý trước
    def explain(self, user_id, item_id): ...      # câu giải thích (chuẩn đầu ra M2)
```

`recommend()` được suy ra từ `score()`, không cần viết lại. Viết xong thì
thêm một dòng vào `MODELS` trong `scripts/run_eval.py` là đo được ngay.

**Đừng đổi hợp đồng này sau tuần 2.** Mỗi lần đổi là phải sửa lại mọi bậc
đã viết xong, và ba nhánh đang làm song song sẽ không khớp nhau.

## Giao thức đánh giá

Leave-one-out **theo thời gian**: tương tác cuối cùng của mỗi user vào tập
kiểm tra, chấm điểm trên 1 phim đúng trộn với 99 phim lấy ngẫu nhiên.

Không dùng chia ngẫu nhiên — nó cho mô hình nhìn thấy tương lai của chính
user đó, làm mọi chỉ số cao lên một cách giả tạo.

`seed` mặc định 42 và **bắt buộc giữ nguyên giữa ba thành viên**. Ba người
chạy trên ba lần chia khác nhau thì số của ba người không so sánh được, và
không có gì trên bảng kết quả nói ra điều đó.

> **Recall@K = HitRate@K và MAP@K = MRR** ở giao thức này, vì mỗi user chỉ
> giữ lại một item đúng. Bày chúng như ba bằng chứng độc lập trong báo cáo
> là nói quá. Chi tiết ở đầu [`recsys/eval/metrics.py`](recsys/eval/metrics.py).

## Ba thí nghiệm của bậc B1/B2

```bash
python scripts/exp_knn.py
```

### 1. Theo nhóm user — giả thuyết của doc B0 chỉ đúng một nửa

nDCG@10, và **Popularity vẫn thắng recall@10 ở nhóm user ít lịch sử**:

| nhóm | số user | Popularity | ItemKNN | UserKNN |
|---|---|---|---|---|
| 5–20 tương tác | 29 | 0.3828 | 0.4406 | **0.4685** |
| 20–100 | 337 | 0.4238 | **0.5594** | 0.5051 |
| >100 | 244 | 0.2698 | **0.3610** | 0.3062 |

- **ItemKNN không thắng ở mọi nhóm.** Nhóm 5–20 thì UserKNN tốt hơn, và
  recall@10 của Popularity (0.8276) còn cao hơn cả hai. Đây là bằng chứng
  ĐO ĐƯỢC cho nhánh lai ghép **chuyển dịch** ở bậc B7, không phải suy đoán.
- **Mọi mô hình đều kém nhất ở nhóm >100** — trong khi Random phẳng qua ba
  nhóm. Vậy độ khó của user nhiều lịch sử là tính chất của **dữ liệu**, không
  phải điểm yếu riêng của Popularity như doc B0 phỏng đoán.
- ⚠️ Nhóm 5–20 chỉ có **29 user** — quá nhỏ để kết luận. Ghi ra cho đầy đủ.

### 2. Hai chế độ chấm điểm

| | recall@10 | ndcg@10 | MRR | coverage |
|---|---|---|---|---|
| ItemKNN `sum` | 0.6754 | **0.4744** | 0.4203 | 0.1089 |
| ItemKNN `weighted` | 0.6361 | 0.3354 | 0.2537 | **0.3315** |
| UserKNN `sum` | 0.6508 | **0.4238** | 0.3662 | 0.0470 |
| UserKNN `weighted` | 0.4672 | 0.2111 | 0.1583 | **0.1692** |

`weighted` là công thức **dự đoán rating** ở slide L2 tr.17/22. Nó tụt 29–50%
nDCG nhưng phủ danh mục gấp **3 lần**. Lý do: phép chia cho `Σ sim` xoá mất
tín hiệu *"có bao nhiêu bằng chứng"*, nên một phim ít người biết mà có đúng
một láng giềng rất giống được xếp ngang phim nổi tiếng có ba mươi láng giềng.

Đây không phải lỗi — **hai công thức trả lời hai câu hỏi khác nhau**:
`weighted` cho MAE/RMSE, `sum` cho top-K.

### 3. Quét số láng giềng k

| k | ItemKNN nDCG@10 | UserKNN nDCG@10 | ItemKNN coverage |
|---|---|---|---|
| 5 | 0.3633 | 0.2944 | 0.1446 |
| 20 | 0.4654 | 0.4036 | 0.1113 |
| 40 | 0.4744 | 0.4238 | 0.1089 |
| 80 | **0.4752** | **0.4269** | 0.1041 |
| 160 | 0.4738 | 0.4218 | 0.1018 |

Đường cong **phẳng từ k = 40 trở đi** — vượt 40 không mua được gì đo được.
Khớp với slide L2 tr.22 (thực nghiệm trên MovieLens gợi ý 20–50), và đây là
lần kiểm lại con số đó bằng chính dữ liệu của nhóm.

⚠️ Coverage **giảm đều** khi k tăng: nhiều láng giềng ⇒ nhiều đồng thuận ⇒
phim phổ biến. Cùng một đánh đổi, lần thứ ba xuất hiện.

## Metadata TMDB

```bash
python scripts/fetch_tmdb.py --limit 20   # thử nhanh 20 phim trước
python scripts/fetch_tmdb.py              # tải hết (~40 phút, chạy một lần)
python scripts/fetch_tmdb.py --report     # xem cache đang phủ được bao nhiêu
```

Cache ghi kiểu JSONL ở `data/cache/tmdb_movies.jsonl`, **nối tiếp được**:
đứt mạng ở phim thứ 5.000 thì chạy lại chỉ tải 4.724 phim còn lại.

**Chưa có TMDB key vẫn chạy được.** `movies.csv` của MovieLens đã có sẵn
thể loại, đủ để bắt đầu bậc B5 ngay hôm nay — 9.708/9.742 phim có ít nhất
một thể loại. Có cache TMDB thì bổ sung thêm diễn viên, đạo diễn, từ khóa.

Cột `has_tmdb` nói rõ từng phim có metadata TMDB hay không. Không có cột này
thì phim *thiếu* dữ liệu trông y hệt phim có dữ liệu *rỗng*, và B5 sẽ báo
"phim này không giống phim nào" cho một thứ thật ra là thiếu metadata.

### Một sửa lỗi có chủ ý so với repo tham khảo

Repo gốc đưa thẳng tên người vào `CountVectorizer`, nên `James Cameron` bị
tách thành `james` + `cameron`. Hậu quả: mọi phim có diễn viên tên Chris,
John hay Lee đều tương đồng giả với nhau. `tmdb._token()` nối tên lại thành
một token — `chris_evans` và `chris_pratt` là hai token khác hẳn nhau.

Đây là cải tiến nhỏ, đo được bằng số, giải thích được bằng một câu — nên
báo cáo nên đo **cả hai cách** rồi trình bày chênh lệch.

## Cấu trúc

```
recsys/
  base.py              hợp đồng chung — đọc file này trước
  data/movielens.py    tải + nạp dữ liệu, nối sang TMDB qua links.csv
  data/tmdb.py         metadata TMDB có cache; chuỗi đặc trưng cho B5
  data/split.py        leave-one-out theo thời gian + lấy mẫu item âm
  eval/metrics.py      P@K, R@K, MAP, MRR, nDCG, AUC, coverage
  eval/runner.py       evaluate() một mô hình, compare() nhiều mô hình
  models/baselines.py  B0 — Popularity, Random
  models/knn.py        B1 — Item-KNN · B2 — User-KNN (Pearson)
scripts/run_eval.py    CLI chạy so sánh
scripts/fetch_tmdb.py  tải metadata TMDB về cache
scripts/exp_knn.py     ba thí nghiệm của bậc B1/B2
tests/                 chốt độ đo, phép chia, đọc metadata, KNN (102 test)
```

## Mã nguồn và dữ liệu tái sử dụng

| Thành phần | Nguồn |
|---|---|
| MovieLens | GroupLens Research |
| TMDB API | The Movie Database |
| `pandas`, `scikit-learn`, `PyTorch`, `implicit`, `mlxtend`, `Flask` | thư viện mã nguồn mở |
| `End-to-End-Movie-Recommendation-System` | [KalyanM45](https://github.com/KalyanM45/End-to-End-Movie-Recommendation-System), GPL v3 — dùng làm **baseline được đo đạc**, không phải nền tảng |

Repo tham khảo nằm ngoài dự án này và không được sửa. Nó cài đặt đúng một
kỹ thuật (tương đồng cosine giữa phim theo diễn viên/đạo diễn/thể loại),
không có người dùng, không có lịch sử xem, không có đánh giá chất lượng —
tương ứng bậc B5 và còn thiếu phần hồ sơ người dùng.

## Khóa API

Chép `.env.example` thành `.env` rồi điền `TMDB_API_KEY`. File `.env` đã nằm
trong `.gitignore`. Không nhúng khóa vào JavaScript như repo tham khảo — ai
mở trang web cũng đọc được và dùng hết quota của nhóm.

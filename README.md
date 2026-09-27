# Semantic Search API

Dự án Semantic Search API xây dựng với **FastAPI**, **PostgreSQL** kết hợp extension **pgvector** và mô hình nhúng văn bản **Sentence-Transformers**.

---

## Cấu trúc thư mục

```text
semantic-search/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI entry point, lifecycle và routes
│   ├── database.py        # Quản lý connection pool asyncpg và init_db
│   ├── models.py          # Data models
│   ├── schemas.py         # Pydantic schemas cho request/response
│   ├── embedding.py       # Tích hợp model Sentence-Transformers
│   ├── chunking.py        # Logic chia nhỏ văn bản (chunking)
│   └── search.py          # Logic vector similarity search
├── scripts/               # Scripts nạp dữ liệu và đánh giá
├── data/                  # Thư mục chứa dữ liệu thô / xử lý
├── tests/                 # Unit tests và integration tests
├── docker-compose.yml     # Khởi chạy PostgreSQL 16 tích hợp pgvector
├── requirements.txt       # Danh sách dependencies
├── .env.example           # Mẫu biến môi trường
└── README.md
```

---

## Hướng dẫn cài đặt (Setup)

### 1. Khởi động PostgreSQL với pgvector qua Docker Compose

Khởi chạy container PostgreSQL 16 (có sẵn extension pgvector) ở chế độ background:

```bash
docker compose up -d
```

Kiểm tra container đang chạy:

```bash
docker ps
```

### 2. Tạo và kích hoạt Virtual Environment (venv)

Trên **Windows (PowerShell)**:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Trên **Linux / macOS**:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

### 4. Cấu hình biến môi trường

Sao chép file cấu hình mẫu `.env.example` thành `.env`:

```bash
# Windows PowerShell
cp .env.example .env

# Linux / macOS
cp .env.example .env
```

Các biến môi trường mặc định:
- `DATABASE_URL`: `postgresql://postgres:postgres@localhost:5432/semantic_search`
- `EMBEDDING_MODEL_NAME`: `all-MiniLM-L6-v2` (tạo vector 384 chiều)

### 5. Chạy ứng dụng FastAPI với Uvicorn

```bash
uvicorn app.main:app --reload
```

Sau khi khởi động:
- Ứng dụng sẽ tự động gọi hàm `init_db()` để tạo extension `vector` và bảng `documents`.
- API Server chạy tại: [http://localhost:8000](http://localhost:8000)
- Swagger UI Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Kiểm tra hoạt động

### 1. Kiểm tra Health Check Endpoint

```bash
curl http://localhost:8000/health
```

Kết quả trả về:
```json
{"status": "ok"}
```

### 2. Kiểm tra bảng `documents` trong PostgreSQL

Sử dụng lệnh `psql` bên trong Docker container:

```bash
docker exec -it semantic_search_postgres psql -U postgres -d semantic_search -c "\d documents"
```

Bảng `documents` sẽ hiển thị:
- `id`: integer / SERIAL PRIMARY KEY
- `parent_title`: text
- `chunk_index`: integer
- `content`: text
- `embedding`: USER-DEFINED (vector(384))

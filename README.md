# AI HelpDesk Frontend

Hệ thống hỗ trợ khách hàng thông minh sử dụng AI, được xây dựng với Fastapi và Langchain Framework

## 🚀 Project Structure

Cấu trúc thư mục của dự án AI HelpDesk Backend:

AI_HelpDesk_Backend/
├── src/                          # Thư mục nguồn chính
│   ├── Domain/                   # Domain layer - Chứa các entity cốt lõi
│   │   └── base_entities.py      # Các entity cơ bản của hệ thống
│   ├── Features/                 # Feature modules - Các tính năng chính
│   │   ├── AuthAPI/             # API Xác thực người dùng
│   │   ├── DeparmentAPI/        # API Quản lý phòng ban
│   │   ├── LangChainAPI/        # API Tích hợp AI/LangChain
│   │   ├── RealTimeAPI/         # API Real-time (WebSocket)
│   │   │   ├── Chat/            # Module Chat real-time
│   │   │   └── Storage/         # Module lưu trữ real-time
│   │   ├── TicketAPI/           # API Quản lý ticket
│   │   └── TicketRepliesAPI/    # API Trả lời ticket
│   ├── SharedKernel/            # Shared components - Các thành phần dùng chung
│   │   ├── ai/                  # Các service liên quan đến AI
│   │   ├── base/                # Base classes và interfaces
│   │   ├── exception/           # Custom exceptions
│   │   ├── persistence/         # Database và persistence
│   │   ├── socket/              # WebSocket management
│   │   ├── utils/               # Utility functions
│   │   └── Utils.py             # Utils chung
│   └── main.py                  # Entry point của ứng dụng
├── ui/                          # Frontend/UI components
│   └── app.py                   # Main UI application
├── config_env/                  # Configuration files
│   ├── config.yaml              # Main application config
│   ├── redis.yaml               # Redis connection settings
│   └── redis_index.yaml         # Redis vector index schema
├── static/                      # Static files (assets, images, etc.)
├── .gitignore                   # Git ignore file
├── LICENSE                      # License file
├── READ.md                      # Documentation (file hiện tại)
├── packages.sh                  # Package installation script
├── script.sh                    # Main setup script
├── pyrightconfig.json           # Python type checking config
└── test_main.http               # HTTP testing file

## ⚙️ Configuration

Hệ thống sử dụng các file cấu hình YAML trong thư mục `config_env/`:

### 1. config.yaml

```yaml
app:
  name: my-service
  debug: true
  port: 8080
  host: localhost

cors:
  config: ["http://localhost:3000"]

openapi:
  title: My API
  version: 1.0.0
  description: Demo Litestar API
  robyn:
    url: http://localhost:8080/openapi.json
  litestar:
    url: http://localhost:8080/docs/openapi.json

database:
  type: MYSQL
  mysql:
    url: mysql+aiomysql://root:@localhost:3306/AI_HelpDesk

redis:
  url: redis://localhost:6379

llm:
  provider: ollama
  
  mistral:
    model: mistral-small-2501
    api_key: <api_key>
    embed: codestral-embed
  
  ollama:
    host: http://localhost:11434
    model: hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
    embed: nomic-embed-text:latest

  splitter:
    fixed:
      chunk_size: 512
      chunk_overlap: 50
    PaC:
      parent_chunk_size: 2048
      parent_chunk_overlap: 400
      child_chunk_size: 512
      child_chunk_overlap: 100

cloudinary:
  url: cloudinary://<cloudinary_url>

jwt:
  secret: <jwt_secret>
  algorithm: HS256
  expire_minutes: 60
  refresh_expire_days: 7

vector_store:
  provider: redis

threading:
  max_workers: 10
  timeout: 300
  queue_size: 100

performance:
  enabled: true
  sizing_strategy: "auto"
  metrics_enabled: true
  circuit_breaker:
    enabled: true
    failure_threshold: 5
    recovery_timeout: 60

neo4j:
  uri: bolt://localhost:7687
  user: neo4j
  password: password

lexical_graph:
  section_size: 10
  embedding_batch_size: 100
  separators:
    law: ["\nĐiều\\s+\\d+", "\n\\d+\\.\\s", "\n[a-z]\\)\\s", "\n\n", "\n"]
    markdown: ["\n## ", "\n### ", "\n#### ", "\n\n", "\n"]
    html: ["</h1>", "</h2>", "</h3>", "</p>", "</div>", "\n"]
    pdf: ["\n\n\n", "\n\n", "\n"]
    txt: ["\n\n", "\n", ". ", " "]
    generic: ["\n\n\n", "\n\n", "\n", ". ", " "]
  entity_types:
    universal: [PERSON, ORGANIZATION, LOCATION, DATE, NUMBER, CONCEPT, ACTION, DOCUMENT]
    law: [LAW, ARTICLE, DECREE, CHAPTER]
    technical: [FUNCTION, VARIABLE, MODULE, API]
    business: [PRODUCT, SERVICE, CUSTOMER, TRANSACTION]
```

### 2. redis.yaml

```yaml
redis:
  connection_pool:
    max_connections: 50
    socket_keepalive: true
    socket_connect_timeout: 5
    socket_timeout: 10
    retry_on_timeout: true
    health_check_interval: 30
  cache_index: true
  lazy_connect: true
```

### 3. redis_index.yaml

```yaml
version: "0.1.0"

index:
  name: doc_idx
  prefix: doc_idx
  storage_type: hash

fields:
  - name: embedding
    type: vector
    attrs:
      dims: 2560
      algorithm: hnsw
      distance_metric: cosine
      datatype: float32

  - name: text
    type: text

  - name: source
    type: tag

  - name: language
    type: tag

  - name: content_type
    type: tag

  - name: page_number
    type: numeric

  - name: pages
    type: tag

  - name: page_span
    type: tag

  - name: chunk_index
    type: numeric

  - name: total_chunks
    type: numeric

  - name: timestamp
    type: numeric

  - name: content_length
    type: numeric

  - name: parent_id
    type: tag
```

## 📋 Prerequisites (Windows)
Cài đặt **Scoop** và các apps cần thiết:

### 1. Cài đặt Scoop
```powershell
# Mở PowerShell và chạy
iwr -useb get.scoop.sh | iex
```

### 2. Cài đặt các apps cần thiết
```powershell
# Core tools
scoop install git

# Database
scoop install sqlite

# PDF processing
scoop install poppler

# OCR
scoop install tesseract

# Fuzzy finder (menu TUI)
scoop install fzf

# Ollama (AI)
scoop install ollama

# Node.js (dev dependencies)
scoop install nodejs
```

### 3. Cài đặt Python packages
```bash
# Tạo virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# hoặc
source venv/bin/activate  # Linux/WSL

# Cài packages
pip install -r packages.sh
```

### 4. Kiểm tra cài đặt
```bash
# Sau khi chạy script.sh, menu sẽ hoạt động nếu fzf đã được cài
bash script.sh

nhớ active rồi mới chạy server
```                             

## 🛠 Tech Stack
- **Framework**: Fastapi, Langchain
- **Language**: Python
- **Package Manager**: pip, venv
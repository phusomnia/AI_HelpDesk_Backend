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
├── config/                      # Configuration files
│   └── redis_index.yaml         # Redis configuration
├── static/                      # Static files (assets, images, etc.)
├── .gitignore                   # Git ignore file
├── LICENSE                      # License file
├── READ.md                      # Documentation (file hiện tại)
├── packages.sh                  # Package installation script
├── script.sh                    # Main setup script
├── pyrightconfig.json           # Python type checking config
└── test_main.http               # HTTP testing file

## 🧞 Commands
| Command                   | Action                                           |
| :------------------------ | :----------------------------------------------- |
| `source script.sh`        | Sử dụng các lệnh                                 |

## 🛠 Tech Stack
- **Framework**: Fastapi, Langchain
- **Language**: Pythom
- **Package Manager**: pip, venv
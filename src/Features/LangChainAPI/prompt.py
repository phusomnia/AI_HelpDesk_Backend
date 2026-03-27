from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from Features.LangChainAPI.LangChainDTO import ChatRequest

def System_Instruction(req: ChatRequest):
  return f"""
  ## AI Instructions

  ## Communication
  - **Response Language**: VIETNAMESE.

  Request: {req.message}
  """

# ====================
# STRUCTED OUTPUT
# ====================
YOUTUBE_TEMPLATE="""
Trích xuất thông tin video YouTube từ đoạn mô tả sau.

Mô tả:
{description}
"""

YOUTUBE_DESCRIPTION = {
    "description": """
    Video "Learn FastAPI in 30 Minutes" được đăng bởi kênh CodeMaster.
    Video đạt 125000 lượt xem sau 2 ngày.
    Đây không phải là YouTube Shorts.
    Ngày đăng: January 10, 2025.
    """
}

def PaC_template(context, query):
  print(context)
  system_prompt = """
  Bạn là trợ lý AI chuyên nghiệp.

  Hãy trả lời câu hỏi của người dùng dựa trên context

  YÊU CẦU BẮT BUỘC:
  1. Trả lời câu hỏi dựa trên ngữ cảnh
  2. KẾT THÚC câu trả lời với 3 dòng thông tin nguồn:

  Trong ngữ cảnh có metadata ở cuối mỗi tài liệu với định dạng:
  Source: <tên file>, Page: <trang>

  Hãy trích xuất thông tin từ metadata này và trình bày lại theo định dạng sau:

  - Nguồn: <tên file>
  - Trang: <không xác định nếu không có thông tin>

  QUAN TRỌNG:
  - Chỉ sử dụng thông tin từ metadata.
  - Nếu không có thông tin trang thì ghi: "không xác định".
  - Không sử dụng định dạng khác.

  Ví dụ output:

  - Nguồn: example.pdf
  - Trang: không xác định
  """
  template = f"""{system_prompt}

  context: {context}

  câu hỏi: {query}

  hãy trả lời câu hỏi dựa trên context và metadata
  """
  return template
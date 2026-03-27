import re
from typing import List
from pathlib import Path
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_community.chat_message_histories import SQLChatMessageHistory
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine

@tool
def tool_qua_bong_vang():
    """Danh sách top Ballon d'Or"""

    ranking = [
        {"rank":1,"player":"Ousmane Dembele (chiến thắng)"},
        {"rank":2,"player":"Lamine Yamal"},
        {"rank":3,"player":"Vitinha"},
        {"rank":4,"player":"Mohamed Salah"},
        {"rank":5,"player":"Raphinha"},
        {"rank":6,"player":"Achraf Hakimi"},
        {"rank":7,"player":"Kylian Mbappe"},
        {"rank":8,"player":"Cole Palmer"},
        {"rank":9,"player":"Gianluigi Donnarumma"},
        {"rank":10,"player":"Nuno Mendes"},
    ]

    return ranking

@tool
def tool_anime():
    """Anime nổi bật"""

    ranking = [
        {"rank":1,"anime":"Solo Leveling","note":"Anime of the Year"},
        {"rank":2,"anime":"Frieren","note":"Best Drama"},
        {"rank":3,"anime":"Demon Slayer","note":"Best Continuing Series"},
        {"rank":4,"anime":"Apothecary Diaries","note":"Voice acting award"},
        {"rank":5,"anime":"Look Back","note":"Film of the Year"},
        {"rank":6,"anime":"Dandadan","note":"Best Opening"},
        {"rank":7,"anime":"Attack on Titan","note":"Global Impact Award"},
    ]

    return ranking

@tool
def get_oscar_nominations_2026():
    """Danh sách giải Oscar 2026"""

    ranking = [
        {"rank":1,"award":"Best Picture","winner":"One Battle After Another"},
        {"rank":2,"award":"Best Director","winner":"Paul Thomas Anderson"},
        {"rank":3,"award":"Best Actor","winner":"Michael B. Jordan"},
        {"rank":4,"award":"Best Actress","winner":"Jessie Buckley"},
        {"rank":5,"award":"Best Supporting Actor","winner":"Sean Penn"},
        {"rank":6,"award":"Best Supporting Actress","winner":"Amy Madigan"},
        {"rank":7,"award":"Best Animated Film","winner":"KPop Demon Hunters"},
        {"rank":8,"award":"Best International Film","winner":"Sentimental Value"},
        {"rank":9,"award":"Best Documentary","winner":"Mr. Nobody Against Putin"},
        {"rank":10,"award":"Best Visual Effects","winner":"Avatar: Fire and Ash"},
    ]

    return ranking


class AgentService:

    def __init__(self, provider):
        self.provider = provider
        self.url = self._create_file(".data", "chat_history.db")
        self.engine = create_engine(f"sqlite:///{self.url}")

    async def write_narrative(self, description: str):
        instruction = """
        Bạn là một tác giả tiểu thuyết tài năng và cũng là nhà lập kế hoạch xuất sắc.

        NHIỆM VỤ:
        Dựa trên mô tả câu chuyện của người dùng, hãy thực hiện tuần tự 2 bước trong cùng một phản hồi:

        BƯỚC 1: LẬP KẾ HOẠCH (PLANNING)
        - Phân tích ý tưởng
        - Tạo Outline chi tiết gồm: Bối cảnh, Nhân vật, Mâu thuẫn, Các bước phát triển, Cao trào, Kết thúc.
        - Định dạng: Danh sách đánh số rõ ràng.

        BƯỚC 2: VIẾT TRUYỆN (WRITING)
        - Dựa ngay vào Outline ở Bước 1.
        - Viết nội dung truyện ngắn hoàn chỉnh, có cảm xúc, mạch lạc.
        - Đảm bảo văn phong tiểu thuyết hấp dẫn.

        QUY TẮC QUAN TRỌNG:
        - KHÔNG dừng lại sau khi viết Outline. Hãy viết luôn cả phần truyện.
        - Sử dụng dấu phân cách rõ ràng giữa phần Outline và phần Truyện (ví dụ: "--- PHẦN TRUYỆN ---").
        - Ngôn ngữ: Tiếng Việt.
        - Trả lời ngắn gọn.
        - Truyện tối đa 500 từ
        """

        agent_novel = create_agent(
            model=self.provider, 
            system_prompt=instruction
        )

        user_input = f"Tạo outline và viết truyện dựa trên mô tả sau: {description}"

        stream = agent_novel.astream({
            "messages": [HumanMessage(content=user_input)]
        })
        async for event in stream:
            if "model" in event:
                for msg in event["model"]["messages"]:
                    if not msg.content:
                        continue
                    yield msg.content
        ...

    async def take_note(self, session_id: str, query: str):
        system_prompt = """
        Bạn là AI ghi chú.

        Quy tắc:

        1. Nếu người dùng viết:
        [GHI CHÚ]: nội dung
        -> trả lời: đã lưu ghi chú thành công

        2. Nếu người dùng viết:
        [HỎI]: câu hỏi
        -> tìm thông tin trong các ghi chú trước đó.

        3. Nếu tìm thấy thông tin phù hợp
        -> trả lời bằng nội dung ghi chú.

        4. Nếu không có thông tin
        -> trả lời: không có thông tin

        5. Nếu người dùng không dùng [GHI CHÚ] hoặc [HỎI]
        -> trả lời: Vui lòng hãy ghi chú bằng cú pháp [GHI CHÚ]: <nội dung ghi chú>
        """

        agent = create_agent(
            model=self.provider,
            system_prompt=system_prompt
        )

        agent_with_history = RunnableWithMessageHistory(
            agent,
            self._get_history,
            input_messages_key="messages"
        )

        result = agent_with_history.invoke(
            {
                "messages": [
                    HumanMessage(content=query)
                ]
            },
            config={
                "configurable": {
                    "session_id": session_id
                }
            }
        )

        return result["messages"][-1].content
    def _get_history(self, session_id: str):
        print("SESSION:", session_id)
        return SQLChatMessageHistory(
            session_id=session_id,
            connection=self.engine
        ) 
    def _create_file(self, path: str, name: str):
        full_path = Path(path).resolve() / name
        full_path.parent.mkdir(parents=True, exist_ok=True)
        if not full_path.exists():
            try:
                full_path.touch()
                print(f"File created: {full_path}")
            except OSError as e:
                raise ValueError(f"Cannot create file: {e}")
        return str(full_path)

    async def search_multi_domain(self, query: str):
        agent = create_agent(
            model=self.provider,
            tools=[
                tool_qua_bong_vang,
                tool_anime,
                get_oscar_nominations_2026
            ],
            system_prompt="""
                Bạn là trợ lý AI hỗ trợ tìm kiếm đa lĩnh vực.

                Nếu câu hỏi liên quan:
                - Oscar → dùng tool get_oscar_nominations_2026
                - Anime → dùng tool tool_anime
                - Quả bóng vàng → dùng tool tool_qua_bong_vang

                Nếu không liên quan đến các lĩnh vực trên → trả lời: "Xin lỗi, tôi không thể trả lời câu hỏi này."
                Ngôn ngữ: Tiếng Việt
                """
        )

        stream = agent.astream(
            {"messages": [{"role": "user", "content": query}]}
        )

        async for event in stream:

            if "model" in event:
                for msg in event["model"]["messages"]:

                    if not msg.content:
                        continue

                    yield msg.content

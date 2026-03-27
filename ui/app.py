from turtle import mode
import streamlit as st
import requests
import httpx


class App:

    BASE_URL = "http://localhost:8080/api/v1"

    def __init__(self):
        self.page = st.sidebar.selectbox(
            "Select page",
            ["Chatbot", "Game Awards QA", "Create Blog", "Convert to JSON", "Novel Agent", "Take note", "Multi-domain Search"]
        )
    def run(self):
        if self.page == "Chatbot":
            self.chatbot_component()
        elif self.page == "Game Awards QA":
            self.game_awards_qa_component()
        elif self.page == "Create Blog":
            self.create_blog_component()
        elif self.page == "Convert to JSON":
            self.convert_video_description_to_json_component()
        elif self.page == "Novel Agent":
            self.novel_agent_component()
        elif self.page == "Take note":
            self.take_note_component()
        elif self.page == "Multi-domain Search":
            self.multi_domain_search_component()

    def chatbot_component(self):

        st.header("Chatbot")

        prompt_type = st.selectbox("Mode", ["none", "stream"])
        message = st.text_input("Prompt:")

        if st.button("Send", type="primary"):

            url = f"{self.BASE_URL}/bai_tap/chat_with_ai?prompt_type={prompt_type}"

            payload = {
                "message": message
            }
            with st.spinner("loading..."):
                try:
                    if prompt_type == "none":
                        response = self.api_service(url, payload)
                        st.write(response)
                    else:
                        response = self.api_stream_service(url, payload)
                        st.write_stream(response)
                except Exception as e:
                    st.error(f"Error: {e}")

    def game_awards_qa_component(self):
        
        st.header("🎮 Game Awards QA 2026")
        
        question = st.text_input("Ask about Game Awards:")
        
        if st.button("Ask", type="primary"):
            try:
                url = f"{self.BASE_URL}/bai_tap/game_awards_QA"
                
                payload = {
                    "question": question
                }
                
                with st.spinner("Getting answer..."):
                    response = self.api_stream_service(url, payload)
                    st.write_stream(response)
                    
            except Exception as e:
                st.error(f"Error: {e}")

    def create_blog_component(self):
        
        st.header("✍️ Create Blog with AI")
        
        title = st.text_input("Blog title:")
        
        if st.button("Generate Blog", type="primary"):
            try:
                url = f"{self.BASE_URL}/bai_tap/create_blog_with_ai"
                
                payload = {
                    "title": title,
                }
                
                with st.spinner("Generating blog..."):
                    response = self.api_stream_service(url, payload)
                    st.success("Generate successfully!")
                    st.write_stream(response)
                    
            except Exception as e:
                st.error(f"Error: {e}")

    def convert_video_description_to_json_component(self):
        st.header("🎬 Format Video Description to JSON")
        
        channel_name = st.text_input("Channel Name:")
        video_title = st.text_input("Video Title:")
        views = st.number_input("Views:", min_value=0, value=0)
        upload_date = st.date_input("Upload Date:")
        tags = st.text_input("Tags (comma separated):")
        is_short = st.checkbox("Is YouTube Shorts?")
        description = f"""
        Channel Name: {channel_name}
        Video Title: {video_title}
        Views: {views}
        Upload Date: {upload_date}
        Tags: {tags}
        Is YouTube Shorts: {is_short}
        """.strip()
        
        if st.button("Format to JSON", type="primary"):
            try:
                url = f"{self.BASE_URL}/bai_tap/format_video"
                
                payload = {
                    "description": description
                }
                
                with st.spinner("Formatting to JSON..."):
                    response = self.api_service(url, payload)
                    st.success("Formatted successfully!")
                    st.json(response)
                    
            except Exception as e:
                st.error(f"Error: {e}")

    def take_note_component(self):
        st.header("📝 Take Note")
        
        # Mode selection
        mode = st.radio("Mode", ["📝 Note", "❓ Ask"], horizontal=True)

        button_text = ""
        query = ""
        spinner_text = ""

        if mode == "📝 Note":
            user_input = st.text_area(
                "Note content:", 
                height=200, 
                placeholder="<nội dung ghi chú>"
            )
            query = f"[GHI CHÚ]: {user_input}" 

            button_text = "Save Note"
            spinner_text = "Saving note..."
            
        elif mode == "❓ Ask":
            user_question = st.text_input("Question:", placeholder="<câu hỏi>")
            query = f"[HỎI]: {user_question}"
            button_text = "Ask"
            spinner_text = "Getting answer..."
        
        session_id = st.text_input("Session ID:", value="user_1")
        
        if st.button(button_text, type="primary"):
            try:
                url = f"{self.BASE_URL}/bai_tap/take_note"
                
                payload = {
                    "query": query,
                    "session_id": session_id
                }
                
                with st.spinner(spinner_text):
                    response = self.api_stream_service(url, payload)
                    st.write_stream(response)
                    
            except Exception as e:
                st.error(f"Error: {e}")

    def novel_agent_component(self):
        st.header("📖 Novel Agent")
        
        # Input fields
        title = st.text_input("Novel title:")
        genre = st.selectbox("Genre", ["Fantasy", "Romance", "Sci-Fi", "Mystery", "Horror"])
        style = st.selectbox("Writing Style", ["Classic", "Modern", "Poetic", "Simple"])
        
        if st.button("Generate Novel", type="primary"):
            try:
                params = {
                    "title": title,
                    "genre": genre,
                    "style": style
                }
                
                url = f"{self.BASE_URL}/bai_tap/novel_agent"
                
                payload = {
                    "description": f"Tiêu đề: {title}, Thể loại: {genre}, Phong cách: {style}",
                }  
                
                with st.spinner("Writing novel..."):
                    response = self.api_stream_service(url, payload)
                    st.write_stream(response)
                    
            except Exception as e:
                st.error(f"Error: {e}")

    def multi_domain_search_component(self):
        st.header("🔍 Tìm kiếm đa lĩnh vực")
        query = st.text_input("Nhập câu hỏi (Oscar, Anime, Ballon d'Or):", placeholder="Ví dụ: Ai thắng Oscar 2026?")
        if st.button("Search", type="primary"):
            try:
                url = f"{self.BASE_URL}/bai_tap/search_multi_domain"
                payload = {"query": query}
                with st.spinner("Searching..."):
                    response = self.api_stream_service(url, payload)
                    st.write_stream(response)
            except Exception as e:
                st.error(f"Error: {e}")

    # Service
    def api_service(self, url, payload):
        res = requests.post(
            url,
            json=payload,
            timeout=60
        )

        if res.status_code != 200:
            raise Exception(f"API error {res.status_code}")

        return res.json()

    def api_stream_service(self, url, payload):

        with httpx.stream(
            "POST",
            url,
            json=payload,
            timeout=75.0
        ) as r:

            for chunk in r.iter_raw():
                yield chunk.decode()

if __name__ == "__main__":
    App().run()
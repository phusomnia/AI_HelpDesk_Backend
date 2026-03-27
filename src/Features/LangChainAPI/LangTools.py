import asyncio
from typing import Any
from ddgs import DDGS
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright
import trafilatura

class LangTools:
    def __init__(self) -> None:
        pass

    async def duckduckgo_search(self, query: str, max_result = 5) -> Any:
        try:
            urls = []
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_result, region="vn-vi")
                if not results:
                    return "Ko tim thay ket qua"
                for result in results:
                    urls.append(result["href"])
            print(f"response: {urls}")
            return urls
        except Exception as e:
            print(e)

    async def ascrawl_web(self, url: str):
        text = await self.crawl_web(url)

        if not text:
            yield "No content found"
            return

        step = 100
        scale = 0.1
        for i in range(0, len(text), step):
            chunk = text[i:i+step]
            yield chunk
            await asyncio.sleep((step/1000) * scale)

    async def crawl_web(self, url: str):
        if not url or not url.startswith(('http://', 'https://')):
            return f"Invalid URL: {url}"

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=[
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--disable-extensions",
                "--disable-blink-features=AutomationControlled",
            ])
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await context.new_page()

            await page.goto(url, wait_until="domcontentloaded")

            html = await page.content()

            await browser.close()

        text = trafilatura.extract(html)
        return text

@tool
def add(a: int, b: int):
    """Cộng hai số nguyên."""
    return a + b
...

@tool
def test():
    """test tooling calling"""
    return "Test tooling calling"
...

@tool
def duckduckgo_search_tool(query: str, max_result = 5) -> Any:
    """
    Tìm kiếm thông tin trên internet qua DuckDuckGo. 
    """
    tool = LangTools()
    return tool.duckduckgo_search(query, max_result)
...

@tool("crawl_web")
async def crawl_web_tool(url: str):
    """
    Lấy thông thông tin trên internet qua url. 
    """
    tool = LangTools()
    return await tool.crawl_web(url)
...

@tool("ascrawl_web")
async def ascrawl_web_tool(url: str):
    """
    Lấy thông tin trên internet qua url và streaming trả về.
    """
    tool = LangTools()
    return await tool.ascrawl_web(url)
...



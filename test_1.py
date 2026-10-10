# -*- coding: utf-8 -*-
# 使用 OpenAI SDK + MCP SDK 调用阿里云百炼联网搜索（WebSearch）MCP 服务
import os
import asyncio
import json
from openai import OpenAI
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

async def main():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误：请设置环境变量 DASHSCOPE_API_KEY")
        return
    mcp_url = "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp"
    headers = {"Authorization": f"Bearer {api_key}"}
    # 1. 连接 MCP Server，获取可用工具列表
    async with streamablehttp_client(mcp_url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            # 转换为 OpenAI function calling 格式
            openai_tools = []
            for tool in tools_result.tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                    },
                })
            # 2. 调用 DashScope（OpenAI 兼容接口）
            client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            messages = [{"role": "user", "content": "搜索一下阿里云百炼MCP的最新进展"}]
            print("正在联网搜索...")
            print("=" * 50)
            # 3. 多轮工具调用循环
            while True:
                response = client.chat.completions.create(
                    model="qwen-max",
                    messages=messages,
                    tools=openai_tools or None,
                )
                choice = response.choices[0]
                msg = choice.message
                if not msg.tool_calls:
                    print(msg.content)
                    break
                messages.append(msg)
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    result = await session.call_tool(tc.function.name, args)
                    tool_content = ""
                    for block in result.content:
                        if hasattr(block, "text"):
                            tool_content += block.text
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_content,
                    })

if __name__ == "__main__":
    asyncio.run(main())
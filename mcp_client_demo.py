# -*- coding: utf-8 -*-
"""
作用：一个最小 MCP 客户端 —— 把 Agent 当"插头"，连上 MCP Server 发现并调用工具。
效果：验证"改工具不用改 Agent 代码"的解耦：客户端只认 MCP 协议，不认工具是谁实现的。
运行：python mcp_client_demo.py （服务端不用手动启动，客户端会自动拉起子进程）
"""
import asyncio  # 作用：MCP 客户端是异步 API。
import sys  # 作用：取当前解释器路径。
from pathlib import Path  # 作用：定位项目根目录。

from mcp import Client, StdioServerParameters  # 作用：MCP 2.x 高层客户端 + stdio 启动参数。

PROJECT_DIR = Path(__file__).resolve().parent  # 作用：本项目根目录；效果：被拉起的服务端子进程能找到 tools.py 与 .env。

SERVER_PARAMS = StdioServerParameters(  # 作用：告诉客户端"怎么把服务端拉起来"。
    command=sys.executable,  # 作用：用当前解释器；效果：不写死 "python"，避免用错虚拟环境。
    args=["mcp_server.py"],
    cwd=PROJECT_DIR,  # 作用：指定子进程工作目录；效果：load_dotenv() 才能找到项目根目录的 .env。
)

async def main() -> None:
    """作用：连服务端 → 列工具 → 调工具；效果：一次跑通即验证整条 MCP 链路。"""
    async with Client(SERVER_PARAMS) as client:  # 作用：拉起子进程并完成 MCP 握手；效果：退出时自动清理。
        tools = await client.list_tools()  # 作用：发现工具；效果：这一步就是"统一插座"的价值 —— 客户端事先不需要知道有哪些工具。
        print(f"发现 {len(tools.tools)} 个工具：")
        for t in tools.tools:
            props = list((t.input_schema or {}).get("properties", {}))  # 作用：取出参数名；效果：确认 schema 跨进程传过来了。
            print(f"  * {t.name} —— {t.description}")
            print(f"    参数：{props}")

        print("\n---- 调用 calc_reimbursement（跨进程）----")
        result = await client.call_tool(  # 作用：调工具；效果：真正执行发生在另一个进程里。
            "calc_reimbursement",
            {"city_tier": "一线", "nights": 3, "actual_amount": 1500},
        )
        for block in result.content:  # 作用：结果是一个内容块列表；效果：文本块取 .text。
            print(getattr(block, "text", ""))
        print("是否错误：", result.is_error)

        print("\n---- 调用 get_employee_info（跨进程）----")
        result = await client.call_tool("get_employee_info", {"employee_id": "E1002"})
        for block in result.content:
            print(getattr(block, "text", ""))

if __name__ == "__main__":
    asyncio.run(main())  # 作用：异步入口。
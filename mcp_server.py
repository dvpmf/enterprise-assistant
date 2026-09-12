# -*- coding: utf-8 -*-
"""
作用：把 tools.py 里的工具注册成一个 MCP Server（给工具定一个"统一插座"）。
效果：任何支持 MCP 的客户端（Claude Desktop / Cursor / 你自己的 Agent）都能"发现并调用"这些工具，
      而无需知道工具是怎么实现的 —— 这就是 N×M 适配降为 N+M。
运行：python mcp_server.py   （stdio 传输，通常由客户端自动拉起本进程）
"""
import sys  # 作用：往 stderr 写启动日志。

from mcp.server import MCPServer  # 作用：MCP 2.x 的服务端类；⚠️ v1 里叫 FastMCP，2.x 已改名。

from tools import TOOLS  # 作用：复用项目已有的 LangChain 工具定义；效果：一份定义，Agent 与 MCP 共用。

server = MCPServer(  # 作用：创建 MCP 服务；效果：name / instructions 会出现在客户端的服务信息里。
    name="enterprise-assistant-tools",
    instructions="企业智能助手工具集：企业知识库检索、差旅报销计算、员工信息查询。",
)

for lc_tool in TOOLS:  # 作用：把每个 LangChain 工具登记成 MCP 工具。
    server.add_tool(
        lc_tool.func,  # 作用：取工具背后的原始 Python 函数；效果：MCP 按函数签名自动生成 inputSchema（含 Annotated 里的参数说明）。
        name=lc_tool.name,
        description=lc_tool.description,
    )

if __name__ == "__main__":
    # ⚠️ 注意：stdio 传输下 stdout 就是 JSON-RPC 协议通道，绝不能往 stdout 打日志，
    #    否则会污染协议流、客户端直接解析失败。要打日志就用 stderr。
    print("MCP Server 启动中（stdio 传输）", file=sys.stderr, flush=True)  # 作用：启动提示；效果：写 stderr 不影响协议。
    server.run(transport="stdio")  # 作用：用标准输入输出通信；效果：客户端拉起本进程，通过 stdin/stdout 交换 JSON-RPC。
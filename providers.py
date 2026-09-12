# -*- coding: utf-8 -*-
"""
作用：模型层工厂 —— 统一创建「对话模型」与「向量化(embedding)模型」。
效果：业务代码只调用 get_chat_model() / get_embedding()；
      切换模型只需改 .env，业务代码一行都不用动。
"""
import os  # 作用：读取环境变量；效果：拿到 .env 里的 key、base_url、模型名。
from functools import lru_cache  # 作用：函数级缓存；效果：模型对象只创建一次，反复调用复用同一个。

from dotenv import load_dotenv  # 作用：把 .env 读进环境变量；效果：os.getenv 才能取到值。
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # 作用：OpenAI 兼容客户端；效果：一套代码连 DeepSeek 和百炼。
from langchain_ollama import ChatOllama  # 作用：本地模型客户端；效果：断网时的离线保底。

load_dotenv()  # 作用：加载项目根目录 .env；效果：本文件被 import 时配置就已就绪。

DEFAULT_PROVIDER = os.getenv("MODEL_PROVIDER", "deepseek")  # 作用：读默认用哪家模型；效果：改 .env 即可整体切换。

def _require_env(name: str) -> str:
    """作用：读取一个必须存在的环境变量；效果：缺失时第一时间报错，而不是等请求发出去才失败。

    参数：
        name: 环境变量名，字符串，例如 "DEEPSEEK_API_KEY"
    返回：
        str: 该环境变量的值
    异常：
        RuntimeError: 变量不存在或为空字符串时抛出，提示去 .env 补配置
    """
    value = os.getenv(name)  # 作用：按名字取值；效果：没配置就是 None。
    if not value:  # 作用：判断空值；效果：None 和 "" 都会被抓住。
        raise RuntimeError(f"缺少环境变量 {name}，请在项目根目录的 .env 中配置")  # 作用：快速失败；效果：报错直接告诉你缺哪个变量。
    return value  # 作用：把值交回给调用方；效果：调用方不用再判空。
def _build_deepseek() -> ChatOpenAI:
    """作用：创建 DeepSeek 对话模型（主链路）；效果：用 OpenAI 兼容协议打到 DeepSeek 服务。

    返回：
        ChatOpenAI: 可 invoke 的对话模型对象
    """
    reasoning_effort = os.getenv("REASONING_EFFORT", "").strip()  # 作用：读思考档位；效果：空字符串代表不启用思考模式。
    return ChatOpenAI(
        model=os.getenv("CHAT_MODEL", "deepseek-flash"),  # 作用：指定模型名；效果：改 .env 就能换型号。
        api_key=_require_env("DEEPSEEK_API_KEY"),  # 作用：传 DeepSeek 密钥；效果：不再依赖 OPENAI_API_KEY 这个名字。
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),  # 作用：把请求地址指向 DeepSeek；效果：同一个 ChatOpenAI 类即可对接任意兼容端点。
        temperature=0,  # 作用：把随机性降到最低；效果：RAG 场景答案更稳定、更贴着资料说。
        timeout=60,  # 作用：单次请求超时上限（秒）；效果：网络卡住时不会无限转圈。
        max_retries=1,  # 作用：SDK 层自动重试次数；效果：偶发抖动能救一次，真挂了尽快交给兜底模型。
        reasoning_effort=reasoning_effort or None,  # 作用：开关思考模式；效果：None 时该参数根本不发给服务端。
    )
def _build_qwen() -> ChatOpenAI:
    """作用：创建阿里云百炼 qwen 对话模型（容灾备用）；效果：同样走 OpenAI 兼容协议，只换 base_url 与 key。

    返回：
        ChatOpenAI: 可 invoke 的备用对话模型对象
    """
    return ChatOpenAI(
        model=os.getenv("FALLBACK_CHAT_MODEL", "qwen3.7-plus"),  # 作用：备用模型名；效果：DeepSeek 挂了由它顶上。
        api_key=_require_env("DASHSCOPE_API_KEY"),  # 作用：百炼密钥；效果：备用链路能独立建立连接。
        base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),  # 作用：指向百炼兼容端点。
        temperature=0,  # 作用：保证答案稳定。
        timeout=60,  # 作用：单次请求超时上限（秒）。
        max_retries=1,  # 作用：SDK 层自动重试。
    )

def _build_ollama() -> ChatOllama:
    """作用：创建本地 Ollama 对话模型（离线保底）；效果：完全不依赖外网，断网也能演示。

    返回：
        ChatOllama: 本地对话模型对象
    """
    return ChatOllama(
        model=os.getenv("OLLAMA_CHAT_MODEL", "qwen3.5:4b"),  # 作用：本地模型名；效果：用你已下载的 qwen3.5:4b。
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),  # 作用：Ollama 服务地址；效果：本机跑用默认值。
        reasoning=False,  # 作用：关闭思考模式；效果：答案进 content 而不是 thinking。
        temperature=0,  # 作用：保证答案稳定。
    )

@lru_cache(maxsize=None)
def get_chat_model(provider: str | None = None, with_fallback: bool = True):
    """作用：对话模型工厂；效果：返回一个可 invoke 的对话模型（主备链或单模型）。

    参数：
        provider: 可选字符串。强制指定 "deepseek" / "qwen" / "ollama"；不传则读 .env 里的 MODEL_PROVIDER
        with_fallback: 是否启用"主备降级"。默认 True；
                       ⚠️ 需要用 bind_tools() 时要传 False —— 主备链对象上没有 bind_tools 方法
    返回：
        Runnable: 调用 .invoke(提示词) 得到 AIMessage，取 .content 就是答案文本
    异常：
        ValueError: provider 名字不认识时抛出
    """
    name = (provider or DEFAULT_PROVIDER).lower()  # 作用：统一转小写；效果："DeepSeek" 与 "DEEPSEEK" 都能识别。

    if name == "deepseek":  # 作用：主链路。
        primary = _build_deepseek()  # 作用：构建主模型。
        if not with_fallback:  # 作用：只要裸模型；效果：给 bind_tools 用。
            return primary
        return primary.with_fallbacks([_build_qwen()])  # 作用：串主备链；效果：主模型抛异常时自动用备用模型重跑。
    if name == "qwen":  # 作用：只用百炼。
        return _build_qwen()
    if name == "ollama":  # 作用：只用本地。
        return _build_ollama()
    raise ValueError(f"未知的 MODEL_PROVIDER: {name!r}，可选：deepseek / qwen / ollama")  # 作用：拦住拼写错误。

@lru_cache(maxsize=None)
def get_embedding() -> OpenAIEmbeddings:
    """作用：embedding 工厂；效果：返回百炼向量化模型，把文本变成 1024 维向量。

    返回：
        OpenAIEmbeddings: 提供 embed_documents(列表) 与 embed_query(单条) 两个方法
    """
    return OpenAIEmbeddings(
        model=os.getenv("EMBED_MODEL", "qwen3.7-text-embedding"),  # 作用：向量模型名；效果：默认输出 1024 维。
        api_key=_require_env("DASHSCOPE_API_KEY"),  # 作用：百炼密钥；效果：DeepSeek 不提供 embedding，只能用百炼。
        base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),  # 作用：指向百炼兼容端点。
        check_embedding_ctx_length=False,  # 作用：关掉按 token 预切分；效果：直接发原文，避免非 OpenAI 服务报错（关键坑）。
        chunk_size=20,  # 作用：限制每批最多 20 条；效果：不超百炼单次 20 行上限，批量入库不被 400 拒绝。
        encoding_format="float",  # 作用：要求返回浮点数组；效果：绕开 base64 编解码差异。
        timeout=60,  # 作用：单次请求超时上限（秒）。
        max_retries=2,  # 作用：自动重试；效果：embedding 是批量操作，多给一次机会。
    )

if __name__ == "__main__":
    # 作用：本文件自检；效果：不联网也能验证三个 provider 与 embedding 都能被正确创建。
    print("当前默认 provider =", DEFAULT_PROVIDER)
    for name in ("deepseek", "qwen", "ollama"):
        print(f"{name:9s} ->", type(get_chat_model(name)).__name__)
    print("embedding  ->", get_embedding().model)
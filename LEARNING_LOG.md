# LEARNING_LOG · 排障日志

> 本项目是「企业智能助手（Enterprise AI Assistant）」的成长记录。
> 格式约定：**做了什么 → 遇到什么问题 → 怎么排查 → 怎么解决 → 懂了什么**。
> 重点记「排查过程」，因为面试官问的是思路，不是答案。

---

## 2026-09-07 · Day5（FastAPI 接口 + Streamlit 界面）

### 阶段
把命令行版 RAG 问答包装成 Web 服务：FastAPI 提供后端接口，Streamlit 提供浏览器界面，实现"前后端分离"。

### 做了什么
- 装包：fastapi + uvicorn（ASGI 服务器）+ streamlit。
- `minimal_api.py`：FastAPI 最小示例，理解"路由 + 启动"。
- `api.py`：把 Day4 的检索+拼接+生成逻辑包装成 `POST /ask` 接口，用 pydantic `AskRequest` 校验请求体必须有 `question` 字段。
- `test_api.py`：用 requests 发 POST 测接口，全链路验证通过（HTTP 客户端 → FastAPI → chroma 检索 → qwen3.5 → 返回）。
- `app.py`：Streamlit 聊天界面，用 `st.session_state.messages` 保存对话历史，`st.chat_message` 渲染气泡。
- 结果：浏览器 8501 端口对话界面跑通，问答准确带出处。

### 问题 A：uvicorn 启动找不到模块
- **现象**：`python -m uvicorn minimal_api:app --reload` 报 `Could not import module "minimal_api"`。
- **怎么排查**：看命令行提示符发现当前目录是 `D:\agent`，而文件在 `D:\agent\enterprise-assistant`。
- **怎么解决**：先 `cd D:\agent\enterprise-assistant` 再启动。uvicorn/streamlit 按"当前目录 + 模块名"定位文件，与运行脚本不同。
- **懂了什么**：启动服务器类工具前先确认 cwd；`minimal_api:app` = 文件名:应用变量名。

### 问题 B：uvicorn.run 加 reload 报错
- **现象**：代码里 `uvicorn.run(app, ..., reload=True)` 报 `You must pass the application as an import string to enable 'reload'`。
- **怎么解决**：改成字符串形式 `uvicorn.run("minimal_api:app", ...)`。
- **懂了什么**：reload 要监控文件变化，必须知道应用来自哪个文件（import string），传内存对象它无法定位文件。

### 问题 C：uvicorn 误报 test_api.py 变化
- **现象**：api.py 运行时监测到 test_api.py 新建触发 Reload，是**无害警告**，忽略即可。

### 问题 D：Streamlit 首次运行卡 Email 提示
- **现象**：首次 `streamlit run app.py` 停在 `Email:` 欢迎注册页。
- **怎么解决**：直接回车跳过即可，不填邮箱。

### 懂了什么
- **前后端分工**：FastAPI 管逻辑（后端），Streamlit 管界面（前端），界面通过 HTTP POST 调接口——自己是"服务器"，也是 Ollama 的"客户端"，处在链路中间层。
- **st.session_state**：Streamlit 每次交互重跑脚本，session_state 是跨重跑保存数据的"保险箱"，聊天记录靠它不丢。
- **pydantic BaseModel**：声明请求体结构，缺字段/类型错自动返回 422，省去手写校验。
- 两种启动方式：命令行 uvicorn 命令 vs 代码里 `uvicorn.run("模块:app")`。

---

## 2026-09-07 · Day4（完整 RAG 问答）

### 阶段
把检索 + 提示词拼接 + 模型生成拼成完整 RAG 问答，让助手"先查资料再回答"。

### 做了什么
- 新建 `rag_qa.py`：`retrieve`（检索）→ `build_prompt`（资料拼进提示词）→ `ask`（生成）三步走。
- 同时引入两位模型：`OllamaEmbeddings`（检索转向量）+ `ChatOllama`（负责回答）。
- `SYSTEM_PROMPT` 明确约束：只依据资料回答、没有就说"资料中未找到相关信息"、末尾注明出处。
- 运行方式：命令行循环问答，输入 exit 退出。

### 测试结果
- ✅ 问"出差住宿费一晚最高报多少" → 答"最高不超过 500 元" + 出处 reimbursement_policy.txt，准确。
- ⚠️ 问"打印机怎么用环保" → 答出"双面打印"，但把同段"禁止共享账号"也带出来了。
  - **原因分析**：命中的碎片是整个"办公规范"段（一段含 3 条），模型照单全收，缺"只挑相关说"的辨别力。
  - **优化方向**：缩小 chunk_size 让碎片更聚焦，或提示词加强"只回答与问题直接相关的内容"。

### 懂了什么
- RAG 三步分工：检索找资料（embedding 模型）、拼接装资料（`build_prompt` 是灵魂）、生成照资料回答（qwen3.5）。
- 模型不乱编不是因为它老实，而是"资料 + 提示词约束"压住了幻觉。
- 碎片粒度影响答案纯度：碎片越大 → 命中整段 → 答案越容易夹带无关信息。
- `llm.invoke()` 既能传 messages 列表（多轮记忆），也能传拼好的字符串（单轮带资料）——取决于场景。

---

## 2026-09-07 · Day3（RAG 建库第一步：文档切分）

### 阶段
用 LangChain 对员工手册、报销制度文档做切分，为后续向量化入库做准备。

### 做了什么
- 新建 `documents/` 目录，放入企业文档 employee_handbook.txt、reimbursement_policy.txt。
- 新建 `split_docs.py`：`TextLoader` 读文档 → `RecursiveCharacterTextSplitter` 切成知识碎片（chunk_size=200、chunk_overlap=50）。
- 运行结果：2 份文档 → 9 个碎片，每个碎片带来源标签，中文无乱码。

### 问题 A：DeprecationWarning（黄色警告，不影响运行）
- **现象**：导入 TextLoader 时提示 `langchain-community is being sunset and is no longer actively maintained`。
- **怎么排查**：读警告内容，得知是该包未来将被弃用，官方建议迁移到独立集成包（langchain_xxx 独立包）。
- **怎么解决**：功能当前正常，暂不处理；Day5 优化时再迁移，避免现在引入回归风险。
- **懂了什么**：弃用警告 ≠ 报错。它提示"未来的风险"，先识别、记下来，在重构时处理，不必打断当前开发。

### 问题 B（观察学习）：overlap 重叠效果验证
- **现象**：打印碎片头尾对比，发现碎片 2 结尾和碎片 3 开头都出现"4. 婚假：凭结婚证可休 3 天带薪婚假"。
- **怎么排查**：临时修改预览代码，分别打印每个碎片的开头 60 字与结尾 60 字，肉眼对比相邻碎片。
- **懂了什么**：
  - `chunk_overlap` = 相邻块共享一段文字，让恰好落在切分边界上的信息在前后两块都完整存在。
  - 若没有重叠，一条信息被切成两半 → 检索时无论搜到哪块都答不完整 → 模型只能瞎编。
  - 文档对象结构 = `page_content`（正文）+ `metadata`（来源标签），metadata 用于追溯碎片来自哪个文件。

### 问题 C：检索结果与预期不符（关键案例）
- **做了什么**：新建 `build_db.py`（向量化+存库）、`search_db.py`（相似度检索）。先写死问题测试通过后，改成 `input()` 支持手动输入问题。
- **现象 1**：问"请事假要提前几天申请"（具体问法）→ 命中请假制度碎片，答案准确。
- **现象 2**：问"如何报销"（宽泛问法）→ 前三名里有两块无关碎片（工作时间、婚假），真正的报销标准只排第 3。
- **怎么排查**：
  1. 对比两种问法的命中差异 → 发现**具体问法命中准、宽泛问法命中差**。
  2. 分析原因：embedding 是 274MB 小模型（nomic-embed-text），擅长"关键词相近"匹配，不擅长"意图相近"。
  3. 认识到：检索召回只是"给线索"，不负责"判断对错"。
- **怎么解决**（方向，非最终）：换更强 embedding（如 bge-m3）、调整切分参数、以及 Day4 让大模型从召回的多块中自行判断。
- **懂了什么**：
  - 用户提问越具体、关键词与文档重合越多，向量检索越准；宽泛/抽象问法是小 embedding 模型的短板。
  - 检索召回 ≠ 最终答案，召回靠 embedding，判断靠大模型——两者分工。
  - 持久化（persist_directory）让向量库落盘到 chroma_db/，search 时直接读取，无需重复向量化。

### 注意
- `TextLoader` 必须指定 `encoding="utf-8"`，否则 Windows 默认编码可能造成中文乱码。
- 所有"调大模型/处理文档"本质上都是把数据喂给模型，RAG 的第一步就是先把大文档切成能检索的小块。

---

## 2026-09-07 · Day2（chat.py 多轮记忆对话）

### 阶段
Python + requests 调本地 Ollama（qwen3.5:4b），实现带 system 人设的多轮记忆命令行对话。

### 做了什么
- 新建 `chat.py`：通过 `history` 列表累积 system/user/assistant 消息，每轮把完整历史发给 Ollama，实现"记忆"。
- 复用了 Day1 的底层方式：`requests.post` 发 JSON，不装任何框架。

### 问题 A：代码理解性 bug（手敲 4 坑）
程序能跑，但"记忆/人设"名存实亡。逐行对照原理后发现问题：

| # | 错误写法 | 正确写法 | 原因 |
|---|---------|---------|------|
| 1 | `from idlelib import history` | 删除 | 误加的无用导入 |
| 2 | payload 里加 `system_prompt` 字段 | 删除 | `/api/chat` 不认识该键，人设必须放 `messages` 的 system 消息 |
| 3 | `'content': 'SYSTEM_PROMPT'` | `'content': SYSTEM_PROMPT` | 带引号 = 发变量名本身；不带 = 引用变量内容 |
| 4 | 用户输入记成 `'role': 'system'` | 记为 `'role': 'user'` | role 决定消息身份，你的话必须是 user |

### 问题 B：多轮后偶发空回复（重点案例）
- **现象**：对话进行几轮后，模型偶尔返回空（`助手:` 后面什么都没有）。重启后再测，到第 3 轮左右复现。
- **怎么排查（关键过程）**：
  1. 先排除人为因素：干净重跑 3 句（不把"助手:xxx"当输入贴回去），空回复仍复现 → 判断是**必现 Bug**，不是误操作。
  2. 加临时调试行，打印**原始完整响应** `resp.json()`（不是只看最终文本）。
  3. 对比响应 JSON 发现：`message.content` 是空字符串 `''`，但 `message.thinking` 里有一大段英文草稿；`eval_count: 927` → **模型明明生成了 927 个 token，却全写进了 thinking，没落进 content**。
  4. 定位根因：qwen3.5:4b 默认开启"思考模式"（先打草稿再回答），小模型偶尔"想太多"，只出草稿就 `stop` 了。
- **怎么解决**：请求体加参数 `'think': False`，关闭思考模式。
- **怎么验证**：再跑 3 轮，每轮 `content` 都有正常文字，`thinking` 字段消失，身份记忆问答正确。
- **懂了什么**：
  - 模型返回的 JSON 里有 `content` 和 `thinking` 两个字段；`content` 才是程序该取的正式答案。
  - "空回复"排查方法论：**加打印看原始响应 → 区分是模型端空返回还是代码端没取到**。
  - 判断"偶发 vs 必现"：排除人为因素（如把模型输出又粘回输入污染历史）后再下结论。
  - 生产环境经验：对开启思考模式的模型，可显式传 `think: False` 保证 `content` 稳定，或用兜底逻辑（content 为空时从 thinking 取）。

---

## 2026-09-06 · Day1（main.py 单轮对话）

### 阶段
Python + requests 调本地 Ollama，实现单轮问答。

### 做了什么
- 新建 `main.py`：`requests.post` 调 `/api/chat`，让 qwen3.5:4b 回答一个问题。
- GitHub 仓库 `dvpmf/enterprise-assistant` 创建并推送首版。

### 问题与解决
- [待补充] KeyError 问题：_（回忆：当时取 `data[...]` 报 KeyError 的具体键、报错原文）_
- [待补充] 懒加载问题：_（回忆：Ollama 首次调用模型加载耗时/超时的具体表现与处理）_

### 懂了什么
- 所有"调大模型"本质 = 发 HTTP 请求 + body 塞 JSON + 收 JSON 响应。
- （待补充）

---

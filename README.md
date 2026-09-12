# 企业智能助手（Enterprise AI Assistant）

一个面向企业内部资料的问答助手：把公司文档传进知识库，员工用自然语言提问，助手先查资料再回答并标出出处；需要算钱或查信息时，会调用工具去算、去查，而不是凭记忆作答。

这是个人学习与求职作品，用来把 RAG、LangChain、Function Calling、LangGraph、MCP 这几条链路跑通，不是生产系统。

## 功能特性

### 知识库问答

- 多格式进库：txt、md、pdf、扫描件、图片。pdf 有文本层就直接抽文字，没有就渲染成图片走 OCR
- 两级检索：向量检索召回 10 条（粗筛），再用 rerank 精排保留 3 条（精排）。这是为了治宽泛问法命中无关碎片的老毛病
- 答案带出处：来源文件名、页码、来源类型、精排得分
- 用 LangChain 串起切分、向量化、检索、生成这几步

### Agent 工具调用

- 封装三个原子工具：知识库检索、差旅报销计算、员工信息查询
- 用官方 SDK 手写一遍 Function Calling，看清底层报文长什么样
- 用 LangGraph 做多工具编排：把[想 → 调工具 → 再想]做成状态图，能处理先查制度、再算金额、最后汇总这种多步任务
- Skill 封装：把提示词和允许使用的工具打包成技能包，切换技能时人设与可用工具跟着换
- MCP 服务：把工具注册成标准 MCP Server，任何支持 MCP 的客户端都能发现并调用

### 界面

- 聊天式提问，回答下方带可展开的[出处]面板
- 侧边栏可切换[知识库问答]和[Agent 工具调用]两种模式，后者会把这次实际调用了哪些工具列出来
- 侧边栏能直接上传文件入库

### 服务与部署

- FastAPI 提供四个接口，自带交互式文档
- 提供 Dockerfile 与 docker-compose 配置（api 与 app 两个容器）
- 对话模型走 DeepSeek 云 API，主模型失败时自动切备用模型
- 密钥走 .env，不进 git 也不打进镜像

## 技术栈

### 语言与运行时

- Python 3.14（本地开发）
- Python 3.12（容器基础镜像 python:3.12-slim-bookworm）

### Web 框架与服务

- FastAPI 0.141.1、Uvicorn 0.52.4、Pydantic 2.13.5、Streamlit 1.63.0

### AI 框架与编排

- LangChain 1.4.0（主框架）、langchain-core 1.6.2、langchain-openai 1.6.2
- langchain-text-splitters 1.1.2、langchain-chroma 1.1.0、langchain-ollama 1.1.0
- LangGraph 1.2.11

### 模型（都走云端 API）

- 对话主力 DeepSeek deepseek-flash，备用百炼 qwen3.7-plus
- 向量化百炼 qwen3.7-text-embedding（1024 维），精排百炼 qwen3.7-text-rerank

用两家厂商的原因很直接：DeepSeek 官方不提供 embedding 接口，向量化只能另找一家。

### 存储、解析、部署

- Chroma 1.5.9（向量库，本地持久化）
- PyMuPDF 1.28.2（读 pdf 与渲染页面）、RapidOCR-onnxruntime 1.2.3 与 onnxruntime 1.29.0（本地 OCR）、Pillow 12.3.0
- python-dotenv 1.2.3、requests 2.34.2、python-multipart 0.0.32、Docker 与 docker-compose
- MCP 官方 Python SDK 2.2.0（工具协议）

## 快速开始

### 前置要求

- Python 3.10 以上
- 两个 API Key：DeepSeek 的（做对话）、阿里云百炼的（做向量化、精排、备用对话）
- 要用 Docker 部署才需要 Docker 与 docker compose

### 1. 获取代码

```bash
git clone git@github.com:dvpmf/enterprise-assistant.git
cd enterprise-assistant
```

### 2. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate        # Windows 用 .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 配置密钥

```bash
cp .env.example .env
```

打开 .env 填上两个 Key，详见下面的配置说明。

### 4. 建向量库

```bash
python make_samples.py           # 可选：生成测试素材
python build_db.py               # 加载 documents/，切分，向量化，写入 chroma_db/
```

换了 embedding 模型必须重跑 build_db.py。不同模型算出来的向量不在同一个空间里，往旧库里追加新向量会让检索结果莫名其妙地不准，而且不会报错。

### 5. 启动

开两个终端，都要先切到项目目录：

```bash
python api.py                    # 终端一：FastAPI，监听 8000
streamlit run app.py             # 终端二：界面，监听 8501
```

### 6. 最简验证

```bash
curl http://127.0.0.1:8000/health
```

期望返回 {"status": "ok", "vectors": 12}。如果 vectors 是 0，说明库是空的，回去跑第 4 步。

再打开 http://127.0.0.1:8501 试三件事：

- 知识库模式问[出差住宿能报多少]，看答案下面有没有[出处（3 条）]折叠面板
- 切到 Agent 模式问[E1002 是谁？]，看有没有[调用的工具：get_employee_info]
- 侧边栏上传一个文件，看有没有[入库成功]提示

接口文档在 http://127.0.0.1:8000/docs，四个接口都能直接点着试。

### Docker 部署（可选）

```bash
docker compose up -d --build
```

会起 api（8000）和 app（8501）两个容器。两个容易踩的地方：

1. chroma_db 是挂载卷，指的是宿主机上 compose 文件所在目录里的那份。如果那边是空的，容器会自建空库，vectors 就是 0。容器里补建库用 `docker compose exec api python build_db.py`
2. .env 不在仓库里，重新 clone 后必须手动拷过去，否则容器起不来

## 配置说明

配置都放在项目根目录的 .env 里，仓库提供 .env.example 当模板：

```bash
cp .env.example .env
```

.env 同时写进了 .gitignore 和 .dockerignore：前者管不进 git，后者管不进镜像。

### 必填项

- DEEPSEEK_API_KEY：DeepSeek 平台的密钥，用来调对话模型。缺失时程序启动阶段就报错
- DASHSCOPE_API_KEY：阿里云百炼的密钥。向量化、精排、备用对话三处都靠它

### 服务地址（一般不用改）

- DEEPSEEK_BASE_URL：默认 https://api.deepseek.com
- DASHSCOPE_BASE_URL：默认 https://dashscope.aliyuncs.com/compatible-mode/v1，这是百炼的 OpenAI 兼容模式端点，接口结构和 OpenAI 一致，所以同一套 LangChain 代码能同时接两家

### 模型选择

- MODEL_PROVIDER：主 provider，默认 deepseek。可选 deepseek（主用 DeepSeek、qwen 兜底）、qwen（只用百炼，做对比实验）、ollama（只用本地，断网保底）
- CHAT_MODEL：主力对话模型，默认 deepseek-flash
- FALLBACK_CHAT_MODEL：备用对话模型，默认 qwen3.7-plus，主模型抛异常时自动顶上
- EMBED_MODEL：向量化模型，默认 qwen3.7-text-embedding，输出 1024 维
- RERANK_MODEL：精排模型，默认 qwen3.7-text-rerank
- REASONING_EFFORT：思考模式档位，可选 high / medium / low，留空表示不启用。如果接口报不支持该参数，删掉这一行即可，代码不用改

### 可选：界面与保底

- API_BASE：告诉界面上哪儿找后端，默认 http://127.0.0.1:8000；容器里由 compose 注入成 http://api:8000
- OLLAMA_CHAT_MODEL 与 OLLAMA_BASE_URL：只有 MODEL_PROVIDER=ollama 时才用
- DASHSCOPE_RERANK_URL：rerank 接口地址，代码里有默认值。注意 rerank 不在 OpenAI 兼容规范里（OpenAI 没有 rerank 接口），所以它走百炼原生路径，报文结构和 chat、embedding 都不一样

## 目录结构

```
enterprise-assistant/
├── documents/              知识库原文与测试素材（内容均为虚构）
├── chroma_db/              向量库，运行时生成（已 gitignore）
│
├── providers.py            模型层工厂：对话模型三选一 + 主备降级；向量模型走百炼
├── loaders.py              多格式摄入，含 OCR
├── retriever.py            检索层：向量库单例 + 粗筛 + 精排 + 拼提示词
├── rerank.py               调百炼做精排
├── build_db.py             重建向量库
├── make_samples.py         生成测试素材
├── test_cloud.py           验证两个云 API 通不通
│
├── tools.py                三个原子工具
├── fc_demo.py              手写 Function Calling 完整流程
├── graph.py                LangGraph 状态图
├── agent.py                对外函数 run_agent()
├── skills/                 Skill 封装：报销技能、入职技能、注册表
├── mcp_server.py           把工具注册成 MCP 服务
├── mcp_client_demo.py      MCP 客户端示例
│
├── api.py                  FastAPI 服务（/ask、/agent、/ingest、/health）
├── app.py                  Streamlit 界面
├── requirements.txt
├── .env.example            环境变量模板
├── Dockerfile / docker-compose.yml / .dockerignore
├── LEARNING_LOG.md         开发过程中的排障记录
└── README.md
```

## 接口文档

四个接口都写在 api.py 里，可以用 http://127.0.0.1:8000/docs 交互式试。

### POST /ask：知识库问答

流程是写死的：召回 top10，精排 top3，生成。稳定、快、省 token。

参数：question（字符串，必填）

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "出差住宿能报多少"}'
```

```json
{
  "answer": "按《公司差旅住宿标准（2026版）》，出差住宿报销上限为：一线城市 500 元/晚，二线城市 350 元/晚，其他城市 260 元/晚……",
  "sources": [
    {"source": "sample_policy.pdf", "page": 1, "type": "pdf-text", "score": 0.9998},
    {"source": "sample_scan.png", "page": null, "type": "image", "score": 0.9866},
    {"source": "reimbursement_policy.txt", "page": null, "type": "text", "score": 0.9756}
  ]
}
```

sources 每条包含：source（文件名）、page（页码，图片和 txt 返回 null）、type（解析路径：text / pdf-text / pdf-ocr / image）、score（精排得分，只能横向比同一次请求）

### POST /agent：Agent 问答

流程不写死，由模型自己决定要不要查、查几次、要不要算。更灵活，但比 /ask 慢也更贵。

参数：question（必填）、thread_id（可选，默认 default，同一个 id 的多轮对话共享上下文）

```bash
curl -X POST http://127.0.0.1:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"question": "公司出差住宿标准是多少？我住 3 晚花了 1500，能报多少？", "thread_id": "u1"}'
```

```json
{
  "answer": "……",
  "tools_used": ["search_knowledge_base", "calc_reimbursement"],
  "messages": 6
}
```

tools_used 是这次实际调用过的工具名（按调用顺序），messages 是消息总数，能看出几步完成。

### POST /ingest：上传文件入库

请求是 multipart/form-data，字段名 file。支持 .txt、.md、.pdf、.png、.jpg、.jpeg、.bmp、.tif、.tiff，大小上限 20 MB。

```bash
curl -X POST http://127.0.0.1:8000/ingest -F "file=@documents/sample_scan.png"
```

```json
{"source": "sample_scan.png", "chunks": 1, "message": "入库成功"}
```

接口是幂等的：同名文件会先删掉旧记录再写入，重复上传不会产生重复条目。如果确实覆盖了，message 会变成入库成功（已覆盖同名旧记录 1 条）。

错误情况：类型不支持返回 400，超过 20 MB 返回 413，一个文字都没解析出来返回 422。

去重是按文件名判断的，不是按内容。两个内容不同但同名的文件会互相覆盖。

### GET /health：健康检查

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status": "ok", "vectors": 12}
```

向量库连不上时返回 503 并带上原因。

## 架构设计

### 分层

```
摄入层   loaders.py      按扩展名分发：txt/md 直读；pdf 有文本层直接抽、没有就渲染成图走 OCR；图片直接 OCR
                         统一产出 Document(正文 + metadata{source, page, type})

切分层   LangChain 的 RecursiveCharacterTextSplitter(chunk_size=200, overlap=50)

向量层   providers.get_embedding()  →  百炼 qwen3.7-text-embedding（1024 维）

存储层   Chroma（本地持久化到 chroma_db/）

检索层   retriever.py    粗筛 top10  →  rerank 精排 top3  →  拼提示词

模型层   providers.get_chat_model()  →  DeepSeek（主）/ 百炼 qwen（备）

编排层   graph.py + agent.py    LangGraph 状态图
         fc_demo.py             裸写 Function Calling（摸清底层用）

协议层   mcp_server.py / mcp_client_demo.py    工具的 MCP 暴露与调用

服务层   api.py          /ask、/agent、/ingest、/health

界面     app.py          Streamlit 聊天界面

部署     Dockerfile / docker-compose.yml        api + app 两个容器
```

### 两条调用链路

/ask（固定流程）：

```
提问 → 向量粗筛 10 条 → rerank 精排 3 条 → 拼提示词 → DeepSeek 生成 → 答案 + 出处
```

/agent（模型自主编排）：

```
提问 → agent 节点：模型决定要不要调工具
         不调 → 直接给答案，走到 END
         要调 → 转到 tools 节点执行工具 → 结果回传 → 回到 agent 节点继续想
```

第二条是个循环，靠模型不再要求调工具来退出。设了 recursion_limit 防止它反复调同一个工具不撒手。

入库和问答必须用同一个 embedding 模型，两个方向在同一套向量空间里，检索才有意义。

## 常见问题与排障

完整排查过程记在 LEARNING_LOG.md，这里只列高频的几条。

### 报 ModuleNotFoundError，或提示缺少环境变量

十有八九是当前目录不对。uvicorn 和 streamlit 按[当前目录 + 模块名]找文件，load_dotenv() 也要靠当前目录找到 .env。先 cd 到项目根目录。

### /health 返回 vectors 是 0

向量库是空的。三种可能：从没跑过 build_db.py、documents/ 是空的、或者在 Docker 里而挂载的宿主机目录是空的。跑一次 build_db.py 即可。

### 检索结果不准，或出现重复条目

不准：先怀疑换了 embedding 但没重建库。两套向量不在同一空间，会错但不报错。

重复：如果重复条目的分数完全一样，说明向量也完全一样，就是同一份内容存了两份。/ingest 现在做了同名覆盖不会再产生新的重复，但库里已有的脏数据要重跑 build_db.py 清掉。

### 容器里 OCR 代码报 ImportError: libGL.so.1

rapidocr-onnxruntime 依赖 opencv-python（不是 headless 版），需要系统图形库，而 slim 镜像默认没有。Dockerfile 里已装 libgl1 和 libglib2.0-0，改基础镜像时记得带上这层。

### 容器里界面打开是 404

docker-compose.yml 里给 app 容器的 API_BASE 没配对。容器之间要用服务名互访，应该是 http://api:8000，不是 127.0.0.1（容器里的 127.0.0.1 指容器自己）。

### docker compose up 全是 CACHED，但代码是新的

CACHED 的含义是和上次构建一样，不是构建得好。全缓存命中要先怀疑我要改的东西到底有没有被带进来。判断方法：看基础镜像名对不对、构建步骤数对不对（新 Dockerfile 是 6 步，老的是 5 步）。

### MCP 客户端报协议解析失败

stdio 传输下 stdout 就是 JSON-RPC 协议通道，不能往里 print。日志要写 stderr：`print("服务端启动", file=sys.stderr, flush=True)`

## 开发规范

### 提交规范

一个文件一次提交，不攒到最后：新增一个文件是一次提交，改一个已有文件也是一次提交。

提交信息用中文带前缀：feat（新功能）、fix（修 bug）、docs（文档）、chore（依赖配置清理）、refactor（重构）。

提交前先 git status 确认 .env 不在待提交列表里，并且用 `git add 具体文件名` 而不是 `git add .`。

### 分支策略

单人项目直接在 main 上开发。分支的意义是隔离多人并行开发或隔离高风险改动；这个项目只有一个人、每次改动都小且有验证，开分支只会增加合并成本。将来要多人协作再按功能开分支。

### 配置管理

- 密钥只放 .env，代码里绝不写死；新增配置项时同步更新 .env.example
- 代码只读环境变量，不读 .env 文件本身（靠 load_dotenv() 启动时注入），这样同一份代码在本机、容器、服务器上都能跑
- .gitignore 和 .dockerignore 是两套独立门禁，两个都要加 .env

### 代码与文档

- 每行代码后面标作用与效果，函数和模块写 docstring
- 排障记录按[现象 → 怎么排查 → 怎么解决 → 学到了什么]四段式写
- 文档不用表格，靠标题分层和列表表达

## 开发历程

阶段一 · 对话基础
先用 requests 直接调本地 Ollama，搞清楚调大模型就是发 HTTP 请求加塞 JSON；然后做多轮记忆，把历史累积起来每轮全发；再用 LangChain 重写一遍（ChatOllama 加 Message 对象加 invoke），体会框架封装了什么。

阶段一 · RAG 问答与 Web 化
用 LangChain 跑通完整 RAG：文档切分、向量化入库、相似度检索、拼提示词生成；再用 FastAPI 包成 POST /ask，用 Streamlit 做聊天界面。

容器化部署
写 Dockerfile 和 docker-compose，代码改成读环境变量，实现配置与代码分离。

阶段三 · 多格式检索与云端模型
新增模型层工厂、多格式摄入（含 OCR）、Rerank 精排；对话模型从本地切到 DeepSeek 云 API，向量化和精排走阿里云百炼；检索升级成召回 10 条再精排 3 条；换 embedding 后重建向量库。

阶段四 · Agent 工具调用
手写一遍 Function Calling 摸清底层报文；用 LangGraph 做多工具编排；把报销和入职两个场景封装成 Skill；用 MCP 把工具服务化；接口新增 /agent；界面支持模式切换、出处展示和文件上传。

## 说明

这是我在找工作期间做的作品，目的是把 RAG、LangChain、Function Calling、LangGraph、MCP 这几条链路完整走一遍并跑通。

documents/ 里的员工手册、报销制度、PDF 与图片素材，以及工具里用到的员工信息，内容全部是虚构的，不含任何真实企业数据。

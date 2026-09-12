# LEARNING_LOG · 排障日志

这是[企业智能助手]开发过程中的问题记录。

记法：做了什么 → 遇到什么现象 → 怎么排查 → 怎么解决 → 学到了什么。重点记排查过程，因为被问到的时候考察的往往是思路，不是答案。

按开发顺序排列，最早的在最上面。

---

## 单轮对话与项目起点

### 做了什么

新建 main.py，用 requests.post 调 Ollama 的 /api/chat，让 qwen3.5:4b 回答一个问题；在 GitHub 上建仓库并推送首版。

### 问题 1：ModuleNotFoundError: No module named 'requests'

- 现象：直接运行 main.py 就报找不到 requests。
- 原因：requests 是第三方库，不是标准库，装了 Python 不等于装了它。
- 解决：pip install requests。

### 问题 2：取结果时 KeyError: 'messages'

- 现象：从响应里取值时报 KeyError。
- 原因：我以为返回结构里是复数的 messages（因为发请求时写的就是 messages），但单个响应里的字段是单数 message。
- 解决：改成 resp.json()["message"]["content"]。
- 学到了什么：发出去的字段名和收回来的字段名不一定一样。拿不准就把完整 JSON 打印出来看一眼，比猜快。

### 问题 3：content 是空的，done_reason 显示 load

- 现象：请求返回了，但答案内容为空。
- 原因：Ollama 是懒加载模型，第一次调用某个模型时它要先把模型读进内存，这一次请求不会真正生成内容。
- 解决：先手动 ollama run 预热一次。

### 懂了什么

所有调大模型的本质就是发 HTTP 请求、body 里塞 JSON、收 JSON 响应。requests.post 传 json= 会自动序列化并加 Content-Type 头，raise_for_status() 用来检查响应码是不是 2xx。

---

## 多轮记忆对话

### 做了什么

新建 chat.py，用 history 列表累积 system / user / assistant 消息，每轮把完整历史发给 Ollama，实现记忆。

### 问题 1：代码能跑，但记忆和人设名存实亡

逐行对照原理后，发现手敲时写错了四处：

1. 多了一行 from idlelib import history，误加的无用导入
2. 在 payload 里加了 system_prompt 字段，/api/chat 不认识这个键，人设必须放在 messages 里的 system 消息中
3. 写成 'content': 'SYSTEM_PROMPT'，带引号发过去的是变量名本身，不带引号才是变量内容
4. 把用户输入记成了 'role': 'system'，用户说的话必须是 user

### 问题 2：多轮后偶发空回复（重点案例）

- 现象：对话几轮后模型偶尔返回空。重启后再测，大约第 3 轮复现。
- 怎么排查：
  1. 先排除人为因素，干净重跑 3 句，空回复仍复现，判断是必现 bug 而不是误操作
  2. 加临时调试行，打印原始完整响应 resp.json()，而不是只看最终文本
  3. 对比发现 message.content 是空字符串，但 message.thinking 里有一大段草稿，eval_count 是 927，也就是模型生成了 927 个 token 却全写进了 thinking
  4. 定位根因：qwen3.5:4b 默认开启思考模式（先打草稿再回答），小模型偶尔想太多，只输出草稿就 stop 了
- 解决：请求体加 'think': False 关闭思考模式。再跑 3 轮，content 都有正常文字。
- 学到了什么：模型返回的 JSON 里 content 才是正式答案；排查空回复的方法是加打印看原始响应，区分是模型端返回空还是代码端没取到；判断偶发还是必现，要先排除人为因素（比如把模型输出又贴回输入污染了历史）。

---

## 文档切分与向量库

### 做了什么

新建 documents/ 放入两份企业文档；新建 split_docs.py，用 TextLoader 读文档、RecursiveCharacterTextSplitter 切块（chunk_size 200、overlap 50），2 份文档切成 9 个碎片；再写 build_db.py 入库、search_db.py 检索。

### 问题 1：DeprecationWarning（黄色警告，不影响运行）

导入 TextLoader 时提示 langchain-community 将被弃用。功能当前正常，暂不处理，避免引入回归风险。弃用警告不等于报错，先记下来，等重构时再处理。

### 问题 2：overlap 重叠效果验证

打印碎片头尾对比时，发现相邻两块都出现了同一条婚假规定。这说明 chunk_overlap 让相邻块共享一段文字，作用是让恰好落在切分边界上的信息在前后两块都完整存在；如果没有重叠，一条信息被切成两半，检索时无论搜到哪块都答不完整。

### 问题 3：检索结果与预期不符（关键案例）

- 现象：问[请事假要提前几天]这种具体问法命中很准；问[如何报销]这种宽泛问法，前三名里有两块无关碎片，真正的报销标准只排第 3。
- 怎么排查：对比两种问法的命中差异，发现具体问法准、宽泛问法差；原因是本地 embedding 是小模型，擅长关键词相近的匹配，不擅长意图相近；检索召回只是给线索，不负责判断对错。
- 后来的处理：这个问题在阶段三通过换更强的 embedding 加 rerank 精排解决。
- 学到了什么：提问越具体、关键词与文档重合越多，向量检索越准；召回不等于最终答案，召回靠 embedding，判断靠大模型。

### 注意

TextLoader 必须指定 encoding="utf-8"，否则 Windows 默认编码可能造成中文乱码。

---

## 完整 RAG 问答

### 做了什么

新建 rag_qa.py：retrieve 检索 → build_prompt 把资料拼进提示词 → 生成。同时引入两个模型，一个负责转向量，一个负责回答。SYSTEM_PROMPT 里明确约束：只依据资料回答、没有就说资料中未找到、末尾注明出处。

### 测试结果

问报销标准答得准确并带出处；但问[打印机怎么用环保]时，把同一段里的[禁止共享账号]也带出来了。原因是命中的是整个办公规范段（一段含 3 条），模型照单全收。后来在阶段三通过两级检索改善。

### 懂了什么

RAG 三步分工是检索找资料、拼接装资料、生成照资料回答，其中 build_prompt 是灵魂。模型不乱编不是因为它老实，而是资料加提示词约束压住了幻觉。碎片粒度影响答案纯度，碎片越大越容易夹带无关内容。

---

## FastAPI 接口与 Streamlit 界面

### 做了什么

用 minimal_api.py 理解路由与启动；api.py 把检索、拼接、生成包装成 POST /ask，用 pydantic 校验 question 字段；test_api.py 做接口测试；app.py 用 Streamlit 做聊天界面，靠 st.session_state 保存历史。

### 问题 1：uvicorn 启动找不到模块

- 现象：报 Could not import module "minimal_api"。
- 怎么排查：看命令行提示符，发现当前目录是上一级，而文件在项目目录里。
- 解决：先 cd 到项目目录再启动。uvicorn 和 streamlit 都是按[当前目录加模块名]找文件的，和直接运行脚本不一样。

### 问题 2：uvicorn.run 加了 reload 报错

报 You must pass the application as an import string。原因是 reload 要监控文件变化，必须知道应用来自哪个文件，传内存里的对象它没法定位。改成字符串形式 uvicorn.run("minimal_api:app") 即可。

### 懂了什么

FastAPI 管逻辑、Streamlit 管界面，界面通过 HTTP POST 调接口。Streamlit 每次交互都会重跑整个脚本，所以聊天记录必须靠 session_state 这种跨重跑的保险箱来存。pydantic 声明请求体结构后，缺字段或类型错会自动返回 422。

---

## Linux 与 Docker 容器化部署

### 做了什么

在 VMware 的 Ubuntu Server 上，从 Windows 用 SSH 远程操作。新增 requirements.txt、.dockerignore、Dockerfile、docker-compose.yml，并把代码改成读环境变量，实现配置与代码分离。

### 问题 1：Docker Hub 拉不到镜像

- 现象：docker run hello-world 报 connect: connection refused。
- 怎么排查：先 curl 百度返回 200，证明虚拟机本身能上网，排除整体断网；再看报错是访问 docker.io 失败，锁定只有 Docker Hub 被墙。
- 解决：配 /etc/docker/daemon.json 的 registry-mirrors，重启 docker。
- 学到了什么：遇到 connection refused 要先区分是网络不通还是目标被墙。

### 问题 2：容器连不上 Windows 上的 Ollama

- 现象：容器能起来，但调模型失败。netstat 看到 Ollama 只监听 127.0.0.1。
- 解决：设 OLLAMA_HOST=0.0.0.0 并放行防火墙入站端口。
- 学到了什么：127.0.0.1 表示只有本机能连，0.0.0.0 表示所有网卡都收。容器里的 127.0.0.1 指的是容器自己，不是宿主机，所以 base_url 必须显式写宿主机 IP。

### 问题 3：Windows 上 setx 设了环境变量，重启也不生效

原因：setx 写的是注册表，但 explorer.exe 在开机时已经把当时的环境读进内存了，从开始菜单或托盘启动的程序继承的是旧环境。改法是在新开的 cmd 窗口里用 set 设临时变量，再从那个窗口启动进程。对应到 Linux 的同类坑是：改了 shell 配置要重新 source 或重新登录。

### 问题 4：docker compose 命令不存在

docker.io 包不含 compose 插件，要单独装 docker-compose-v2。另外 v1（带连字符的 docker-compose）已淘汰，现在用 v2（空格的 docker compose）。

### 问题 5：docker exec 加管道会卡死终端

exec 默认分配伪终端，和管道冲突，加 -T 参数即可；或者改用 docker inspect 从外部看配置。写自动化脚本时容易踩。

### 问题 6（重点案例）：Ollama 加载模型崩溃加 CUDA error

- 现象：POST /ask 返回 500，容器日志最后一行是 CUDA error: shared object initialization failed。
- 怎么排查（分层排除法）：
  1. 是 500 而不是连接失败，说明网络通的，属于服务端异常
  2. 读堆栈，崩在加载对话模型那一步，而 embedding 和检索都已经成功，范围收窄到模型加载这一环
  3. 查进程和端口，发现 Ollama 装在非默认路径，而且桌面版和命令行 server 同时在跑
  4. 翻服务器日志对齐时间线，发现同一个模型一次加载失败、另一次加载成功，判定是间歇性故障，不是代码或容器的问题
  5. 查上游 issue 确认：Windows 加 CUDA 下 llama-server 偶发分配锁页内存失败，静默降级后第一个内核调用就崩；社区反馈是桌面版正常、命令行启动会触发
- 解决：杀掉所有 ollama 进程，改回官方桌面版启动。
- 附带一个坑：崩溃的实例会让 /api/tags 返回空模型列表，制造出[模型被删了]的假象，一度把排查带偏。
- 学到了什么：排障要分层，网络、服务端、调用链、上游组件、已知 issue，每层都用证据把范围缩小；把日志和时间线对齐是判断偶发还是必现的关键；能划清故障边界（容器没责任，是宿主机显卡加 Ollama 的已知缺陷），有时候比修好它更重要。

### 懂了什么（Docker 的核心概念）

镜像是安装包或模板，容器是跑起来的实例，类似类和对象。构建分层缓存要求把 COPY requirements.txt 和 COPY . . 分开写，改代码时才不用重装依赖。端口映射是宿主端口对容器端口，EXPOSE 只是声明。卷挂载把数据放在宿主上，和代码镜像分离。Compose 会建专用网络和内置 DNS，所以 app 容器能用服务名直接访问 api 容器。

---

## 多格式检索与云端模型

### 做了什么

把模型从本地换成云端：对话用 DeepSeek 的 deepseek-flash 做主力、百炼 qwen3.7-plus 做备用，向量化用百炼 qwen3.7-text-embedding，检索结果再用百炼 qwen3.7-text-rerank 精排。为此新建 providers.py 把创建模型收进工厂函数，业务代码只调 get_chat_model() 和 get_embedding()，换模型只改 .env，并套了一层主备降级。

让知识库支持多格式：新建 loaders.py，txt、md、pdf、图片都能进库，pdf 有文本层直接抽、没有就渲染成图走 OCR；另写 make_samples.py 造测试素材。

最后重构检索与接口：build_db.py 改成先删库再重建，api.py 换成新 provider，检索升级成召回 10 条再精排 3 条，并新增 /ingest 与 /health。

### 问题 1：换了 embedding 模型，旧向量库必须删掉重建

embedding 决定的是向量空间。旧库的向量是 nomic-embed-text 算出来的，新问题是 qwen3.7 算出来的，两者不在同一个坐标系里，相似度没有任何意义。维度不同只是表面，就算维度一样也不能混用。解决方法是先 rmtree 删目录再重建，建完立刻用检索自测一次。忘了这一步的表现是检索结果莫名其妙地不准，但不会报错。

### 问题 2：宽泛问法命中无关碎片，用两级检索解决

向量检索把问题和文档分别编码成向量再算距离，这个过程信息有损，所以问题越宽泛距离就越不可靠。改成两级：先召回 10 条，再把这 10 条连同原问题交给 rerank 打分，只留 3 条。rerank 是把问题和候选片段拼在一起送进模型，判断更准。常见的企业级链路是召回、重排、生成，召回负责快、宁多勿漏，重排负责准、筛掉无关的。

### 问题 3：encoding_format 的警告不是报错

每次跑都打印黄色警告说 encoding_format 不是默认参数、被转移到了 model_kwargs。翻源码发现这个类确实没声明这个字段，而 model_kwargs 会被原样拼进请求体，所以参数照样生效，只是走了条非正式的路。LangChain 官方也建议接非 OpenAI 服务时设成 float，所以保留不动。Warning 和 Error 是两回事，看到警告先读它说了什么再决定。

### 问题 4：.dockerignore 里漏了 .env，密钥会被打进镜像

原 .dockerignore 里没有 .env，而 Dockerfile 有 COPY . .，会把构建目录下的东西全部复制进镜像，密钥就永久留在镜像层里了。解决是 .dockerignore 加上 .env，运行时配置改由 compose 的 env_file 注入。要点：.gitignore 和 .dockerignore 是两套独立门禁，管住了 git 不等于管住了镜像；密钥要运行时注入，不能构建时打进去。

### 问题 5：slim 镜像里没有 libGL，OCR 依赖会导入失败

查依赖列表时发现 rapidocr-onnxruntime 依赖的是 opencv-python 而不是 headless 版，它需要系统里的图形库，而 slim 镜像默认不装，结果是本地跑得好好的、进容器一 import cv2 就报 ImportError。解决是 Dockerfile 里加一层 apt 装 libgl1 和 libglib2.0-0，并把基础镜像钉成 slim-bookworm 避免 Debian 版本漂移。ImportError 里出现 .so 文件名，基本可以断定缺的是系统级动态库。

### 问题 6：容器里的 chroma_db 不是开发机上那个

compose 里挂载的 ./chroma_db 指的是宿主机上 compose 文件所在目录里的那份，不是开发机上建的库。判断方法很简单：/health 返回的向量条数是 0，就说明那份是空的、容器自建了一个空库。卷挂载是用宿主的目录替换容器里的目录，所以要清楚宿主那一侧到底是什么，跨机器部署时[我本地是好的]不成立。

### 问题 7：虚拟机上的容器跑的还是旧代码

构建成功、容器也 Up，但行为不对。逐行看构建日志找到三个证据：基础镜像名是旧的、构建步骤只有 5 步（新 Dockerfile 是 6 步）、几乎每一步都是 CACHED。结论是虚拟机上的代码根本没更新。要点：看构建日志要读细节，不能只看构建成功；基础镜像名、步骤数、缓存命中情况都能反推出用的到底是哪份代码。

### 问题 8：Docker 构建全 CACHED，但代码其实是旧的

和上一条是同一件事，单独记是因为这一点反直觉：CACHED 的含义是和上次构建一样，不是构建得好。看到全缓存命中，要先怀疑我要改的东西到底有没有被带进来。

### 问题 9：虚拟机打不开，报[获取该虚拟机的所有权失败]

- 怎么排查：先看进程列表确认没有 vmware-vmx.exe，说明没有虚拟机在运行；再看虚拟机目录，发现还留着 .lck 锁目录，里面的锁名对应的进程早就不存在了。
- 原因：VMware 用 .lck 目录当排他锁，开机时创建、正常关机时删除。上次没正常关机，锁留了下来，VMware 就以为还有程序在用它。
- 解决：确认进程真的不存在之后，把 .lck 目录删掉。
- 学到了什么：这和 Linux 的 .pid 文件、flock 是同一个思路，用文件当占位牌表示资源被占用，清理顺序必须是先确认持有者进程死了再删锁。如果虚拟机真在运行就删锁，两个进程同时写同一个虚拟磁盘，可能把磁盘写坏。
- 附带一个坑：重开机时提示[此虚拟机似乎正在使用中]，要选[我已移动该虚拟机]。选[我已复制]会重新生成网卡 MAC 地址，IP 变了 SSH 就连不上。

---

## Agent 工具调用

### 做了什么

新建 tools.py 封装三个工具（知识库检索、报销计算、员工信息）；fc_demo.py 用官方 SDK 直连 DeepSeek 手写一遍 Function Calling，看清底层报文；graph.py 用 LangGraph 做成状态图，agent.py 包成 run_agent() 对外用；api.py 新增 POST /agent。最后把报销和入职两个场景打成 Skill，再用 MCP 把工具暴露成标准服务。

### 问题 1：api.py 和 tools.py 差点循环导入

- 现象：工具里要用检索逻辑，而这套逻辑写在 api.py 里，直接 from api import 也能跑。
- 为什么不能这么写：api.py 后面要 import agent，agent 会 import graph，graph 会 import tools，tools 又要 import api，这就成环了。
- 解决：把检索代码抽成 retriever.py，两边都从它导入，环就断了。
- 学到了什么：抽模块很多时候不是为了让代码好看，而是为了打断依赖环。判断标准是这段逻辑有没有两个以上不相干的调用方。

### 问题 2：模型给的参数是字符串，回传还必须配对 id

- 现象：直接把 call.function.arguments 当字典用，报类型错误。
- 怎么排查：把原始响应打印出来，发现它是一个 JSON 格式的字符串。
- 解决：用 json.loads() 转成字典。回传结果时 role 必须是 tool，并且要带上对应的 tool_call_id，配错了接口直接报 400。
- 学到了什么：Function Calling 的完整链路是用 JSON Schema 声明工具、清单随请求发给模型、模型返回 tool_calls、本地执行、结果带 id 回传、模型生成最终答案。模型只负责说调哪个和参数是什么，真正执行的是我自己的代码。

### 问题 3：@tool 默认不会读 docstring 里的参数说明

翻 langchain-core 源码发现 tool() 装饰器有个 parse_docstring 参数，默认是 False，也就是默认只把整个 docstring 当工具描述，不解析里面的参数说明；但它会把 Annotated 里写的说明取出来当参数描述。所以参数说明要写成 Annotated[str, "要查询的问题"]，并在自检代码里打印 args_schema 确认说明确实进了 schema。参数说明写得准不准，直接影响模型能不能选对工具、填对参数。

### 问题 4：主备链模型上没有 bind_tools 方法

get_chat_model() 返回的不是 ChatOpenAI 而是包了主备降级的 RunnableWithFallbacks，bind_tools 是 BaseChatModel 的方法，它自然没有。解决是给工厂加一个 with_fallback 开关拿裸模型，更好的做法是先 bind_tools 再 with_fallbacks，这样工具能力和容灾都能保住。学到两点：封装一层就少一批能力；遇到方法不存在，第一反应应该是先打印对象的真实类型。另外要清楚 with_fallbacks 兜的是调用期异常，兜不住请求成功但返回内容为空这种情况。

### 问题 5：MCP 2.x 里 FastMCP 已经改名了

装包时发现多了个依赖叫 mcp-types，感觉 API 可能和常见教程不一样，于是直接去读源码：导出的类是 MCPServer，而 mcp/server/fastmcp.py 里只有一行 raise，提示 FastMCP 已改名为 MCPServer、附带迁移文档链接。正确写法是 from mcp.server import MCPServer，而且 @server.tool() 必须带括号。客户端则是用 Client 加 StdioServerParameters。教训是网上的教程可能是旧版本的，装完包先读源码确认 API 比照抄博客可靠。

### 问题 6：stdio 传输下不能往 stdout 打日志

在 mcp_server.py 里顺手写了句 print，结果客户端直接报协议解析失败。原因是 stdio 传输就是用进程的 stdin 和 stdout 传 JSON-RPC 消息的，stdout 是协议通道，不是给人看的控制台，往里 print 等于往协议流里插垃圾数据。日志要写到 stderr。凡是用标准输入输出通信的程序，日志都不能走 stdout。

### 问题 7：同一个文件传两次，检索结果里出现了重复条目（重点案例）

- 现象：问[出差住宿能报多少]，三条出处里同一个文件出现了两次，而且分数完全一样；同时 /health 显示的向量条数比建库后多了一条。
- 怎么排查：
  1. 一开始怀疑检索不准，但[两条分数一模一样]很反常，两份不同资料不可能分毫不差。于是倒推：分数一样说明向量一样，那大概率是同一份内容存了两份
  2. 回看做过什么：验证 /ingest 时上传过一次这个文件，而它在建库读 documents/ 时已经入过一次库了
  3. 去看 /ingest 的代码，最后一行是 add_documents，只追加、从不检查有没有同名记录
  4. 再看条数：建库后 12 条，那次上传后变成 13 条，正好对上
- 后果：不只是多一条结果。top3 只有三个名额，被重复的那条占掉一个，本该排第三的资料被挤了出去。对比很明显：清库前问这个问题答案里没提文档冲突，清库之后答案主动指出了两份文档住宿标准不一致。
- 解决：先重跑 build_db.py 清掉脏数据；再改 /ingest，入库前先按文件名查有没有旧记录，有就先删再写，让接口变成幂等的。
- 验证：把造成重复的文件再传一次，返回里出现[已覆盖同名旧记录 1 条]，条数稳定在 12 没变。
- 学到了什么：写接口先问一句这个操作重复执行会怎样，重复执行结果一致才叫幂等；先删后写的顺序有讲究，删除要放在解析成功之后，否则上传坏文件会把库里原有内容清掉；这次去重是按文件名不是按内容，要按内容去重得把哈希写进 metadata；排障要从反常的细节入手，这次破案的关键就是[两条分数完全一样]这个不合逻辑的现象。

### 实测结果

- loaders.py 跑一遍：5 个 Document，覆盖 text、pdf-text、pdf-ocr、image 四种来源
- test_cloud.py：DeepSeek 回话约 1.65 秒；百炼 embedding 约 0.50 秒、1024 维
- POST /ask：约 3.99 秒，返回答案加 3 条带精排得分的出处
- POST /agent：复合问题约 7.28 秒，调了 2 次检索加 3 次计算共 5 次工具调用
- 入库幂等验证：同一文件重复上传，返回[已覆盖同名旧记录 1 条]，条数稳定在 12
- MCP 解耦验证：改服务端数值，客户端零改动，输出自动变化

---

## 界面优化

### 做了什么

原来的 app.py 只把答案显示出来，返回的 sources 一个都没用，/agent 和 /ingest 在界面上也完全不可见。改完之后：回答下方多了可展开的出处面板；侧边栏能切换知识库问答和 Agent 工具调用两种模式，后者会把实际调用的工具列出来；侧边栏还能直接上传文件入库。

### 问题 1：改了界面，但容器里的界面会全 404

旧版界面只调 /ask，所以环境变量传的是带 /ask 后缀的完整地址。新界面要调三个接口，如果变量里还带着 /ask 后缀就没法拼出另外两个地址。解决是把它改成基地址 API_BASE，由代码去拼路径，同时要改 docker-compose.yml 里给 app 容器注入的那个变量。这就是典型的改一处漏一处，凡是引用了这个变量的地方都要跟着改，改名之前先全局搜一遍。

### 问题 2：Streamlit 脚本不方便在沙箱里验证

界面靠浏览器渲染，而运行环境的沙箱开不了浏览器，没法直观看到效果。替代办法是把 app.py 当普通模块导入，直接调里面的函数打真实请求来验证；接口是否通、渲染函数是否抛异常都能覆盖到。但这种验证只覆盖逻辑层，不覆盖像素级渲染，要看真实界面最终还是得自己跑 streamlit run 用浏览器打开。知道验证手段的边界在哪，比验证本身更重要。

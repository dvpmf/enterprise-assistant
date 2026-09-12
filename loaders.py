# -*- coding: utf-8 -*-
"""
作用：多格式文档加载器 —— 把 txt / md / pdf / 图片 统一转成 LangChain 的 Document 列表。
效果：无论输入什么格式，输出都是 List[Document]，上层（切分、向量化）不用关心原始格式。
"""
from pathlib import Path  # 作用：面向对象路径；效果：比字符串拼路径安全。

import pymupdf  # 作用：PDF 读写；效果：抽文本层、渲染页面成图。
from langchain_core.documents import Document  # 作用：LangChain 的统一文档容器；效果：全项目只认这一种结构。

# ---- 扩展名 → 处理分支 的映射表 ----
TEXT_SUFFIXES = {".txt", ".md"}  # 作用：纯文本类；效果：直接读文件。
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}  # 作用：图片类；效果：走 OCR。
PDF_SUFFIXES = {".pdf"}  # 作用：PDF 类；效果：先试文本层，不行再渲染+OCR。

MIN_TEXT_CHARS = 20  # 作用：判定 PDF 一页"有没有文字"的阈值；效果：少于 20 字符就当扫描件处理。

_ocr_engine = None  # 作用：全局缓存 OCR 引擎；效果：整个进程只初始化一次。

def _get_ocr():
    """作用：懒加载 OCR 引擎；效果：只有真遇到图片时才初始化，import 本模块时不加载模型。

    返回：
        RapidOCR: OCR 引擎实例
    """
    global _ocr_engine  # 作用：声明要改的是模块级变量；效果：不加这行，函数内赋值只会创建局部变量。
    if _ocr_engine is None:  # 作用：只初始化一次。
        from rapidocr_onnxruntime import RapidOCR  # 作用：延迟导入；效果：不用 OCR 时省内存和启动时间。
        _ocr_engine = RapidOCR()  # 作用：初始化引擎；效果：首次运行要加载 ONNX 模型，会明显卡一下。
    return _ocr_engine

def _ocr_to_text(img_source) -> str:
    """作用：对图片做 OCR 识别；效果：返回识别出的纯文本（按行拼接）。

    参数：
        img_source: 图片路径（str / Path）或图片字节（bytes）
    返回：
        str: 识别出的文字；一个都没认出来时返回空字符串
    """
    result, _ = _get_ocr()(img_source)  # 作用：执行识别；效果：result 形如 [[框坐标, "文字", "置信度"], ...]，识别失败时为 None。
    if not result:  # 作用：兜住"没认出字"；效果：None 直接返回空串，避免下面报错。
        return ""
    return "\n".join(item[1] for item in result)  # 作用：只取每项第 2 个元素（文字）；效果：按行拼成整段文本。

def load_text(path: Path) -> list[Document]:
    """作用：加载纯文本文件（txt / md）；效果：整个文件当作一个 Document。

    参数：
        path: 文件路径
    返回：
        list[Document]: 只含一个元素的列表
    """
    text = path.read_text(encoding="utf-8")  # 作用：读全文并指定 UTF-8；效果：避免 Windows 默认 GBK 让中文乱码。
    return [Document(page_content=text, metadata={"source": path.name, "type": "text"})]  # 作用：包装成 Document；效果：source 只存文件名，方便在答案里标注出处。

def load_image(path: Path) -> list[Document]:
    """作用：加载图片；效果：OCR 把像素转成文字，再包装成 Document。

    参数：
        path: 图片路径
    返回：
        list[Document]: 识别成功返回一个元素，识别不出内容返回空列表
    """
    text = _ocr_to_text(str(path))  # 作用：OCR；效果：文本向量模型看不懂像素，必须先变成文字。
    if not text.strip():  # 作用：防空文档；效果：识别失败就不要往向量库里塞垃圾。
        print(f"[警告] {path.name} 未识别出文字，已跳过")
        return []
    return [Document(page_content=text, metadata={"source": path.name, "type": "image"})]

def load_pdf(path: Path) -> list[Document]:
    """作用：加载 PDF；效果：逐页处理 —— 有文本层就直读，没有就把该页渲染成图走 OCR。

    参数：
        path: PDF 文件路径
    返回：
        list[Document]: 每页一个 Document
    """
    docs: list[Document] = []  # 作用：收集结果。
    doc = pymupdf.open(path)  # 作用：打开 PDF；效果：拿到可遍历的页面对象。
    try:
        for page_index, page in enumerate(doc):  # 作用：逐页遍历；效果：page_index 从 0 开始计数。
            text = page.get_text("text").strip()  # 作用：抽文本层；效果：扫描件这一句拿到的是空字符串。
            if len(text) >= MIN_TEXT_CHARS:  # 作用：判断本页是不是"真有字"。
                source_type = "pdf-text"  # 作用：标记来源类型；效果：便于排障时知道走的哪条路。
            else:
                pix = page.get_pixmap(dpi=200)  # 作用：把本页渲染成位图；效果：dpi=200 兼顾清晰度与速度。
                text = _ocr_to_text(pix.tobytes("png")).strip()  # 作用：转成 PNG 字节再 OCR；效果：RapidOCR 直接支持 bytes 入参。
                source_type = "pdf-ocr"
            if not text:  # 作用：两条路都没拿到字；效果：跳过空白页。
                print(f"[提示] {path.name} 第 {page_index + 1} 页无内容，已跳过")
                continue
            docs.append(Document(
                page_content=text,  # 作用：正文。
                metadata={"source": path.name, "page": page_index + 1, "type": source_type},  # 作用：记录出处与页码；效果：回答时能标注"某某文件第几页"。
            ))
    finally:
        doc.close()  # 作用：无论中途是否报错都关闭文件；效果：避免文件句柄泄漏。
    return docs

def load_file(path: Path) -> list[Document]:
    """作用：单文件分发器；效果：按扩展名自动选分支，不支持的格式给提示并返回空列表。

    参数：
        path: 文件路径
    返回：
        list[Document]: 该文件产出的 Document 列表
    """
    suffix = path.suffix.lower()  # 作用：取扩展名转小写；效果：.PDF 和 .pdf 都能识别。
    if suffix in TEXT_SUFFIXES:
        return load_text(path)
    if suffix in PDF_SUFFIXES:
        return load_pdf(path)
    if suffix in IMAGE_SUFFIXES:
        return load_image(path)
    print(f"[跳过] 不支持的格式：{path.name}")
    return []

def load_directory(directory: str | Path = "documents") -> list[Document]:
    """作用：批量加载目录下所有支持的文档；效果：返回合并后的 Document 列表。

    参数：
        directory: 目录路径，默认 documents
    返回：
        list[Document]: 所有文件产出的 Document
    """
    dir_path = Path(directory)  # 作用：转成 Path 对象；效果：兼容传字符串或 Path。
    docs: list[Document] = []  # 作用：收集结果。
    for path in sorted(dir_path.iterdir()):  # 作用：遍历目录下所有条目；效果：sorted 让顺序稳定，方便复现结果。
        if not path.is_file():  # 作用：跳过子目录。
            continue
        docs.extend(load_file(path))  # 作用：合并结果；效果：extend 是逐个追加，不是把列表套进列表。
    return docs

if __name__ == "__main__":
    # 作用：自检；效果：把 documents/ 全量加载一遍，打印每个 Document 的来源、类型、字数。
    all_docs = load_directory()  # 作用：默认加载 documents 目录。
    print(f"\n共加载 {len(all_docs)} 个 Document：")
    for doc in all_docs:
        meta = doc.metadata  # 作用：取出元数据；效果：下面取字段更短。
        print(f"  [{meta.get('type', '-'):9s}] {meta.get('source')} "
              f"page={meta.get('page', '-')} 字数={len(doc.page_content)}")  # 作用：格式化打印；效果：一眼看出哪条分支被走了。
    if all_docs:  # 作用：防空列表；效果：没加载到文件时不打印正文。
        print("\n---- 第一个 Document 前 100 字 ----")
        print(all_docs[0].page_content[:100])  # 作用：切片取前 100 字；效果：确认内容没乱码。
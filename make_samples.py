# -*- coding: utf-8 -*-
"""
作用：生成测试素材（PDF + 图片），让 loaders.py 有东西可测。
效果：在 documents/ 下生成三个文件：
      ① sample_policy.pdf —— 带文本层的正常 PDF（走 PyMuPDF 直读）
      ② sample_scan.png   —— 纯图片（走 OCR）
      ③ sample_scan.pdf   —— 无文本层的 PDF（渲染成图后走 OCR）
"""
from pathlib import Path  # 作用：面向对象的路径操作；效果：比字符串拼路径安全，跨系统通用。

import pymupdf  # 作用：PDF 读写库；效果：能建 PDF、写文字、插图片。
from PIL import Image, ImageDraw, ImageFont  # 作用：图像库；效果：能画一张带文字的图当"扫描件"。

DOC_DIR = Path("documents")  # 作用：素材目录；效果：所有产出都放这里。
SAMPLE_TEXT = """公司差旅住宿标准（2026 版）

一、一线城市：住宿标准 500 元/晚，出差补贴 120 元/天。
二、二线城市：住宿标准 350 元/晚，出差补贴 80 元/天。
三、其他城市：住宿标准 260 元/晚，出差补贴 60 元/天。
四、住宿费需凭正规发票报销，超额部分由个人自行承担。
"""  # 作用：素材正文；效果：内容与"报销"相关，方便后面用"如何报销"做检索测试。

def make_text_pdf() -> Path:
    """作用：生成带文本层的 PDF；效果：loaders 可以直接抽出文字，无需 OCR。

    返回：
        Path: 生成的文件路径
    """
    path = DOC_DIR / "sample_policy.pdf"  # 作用：目标路径；效果：documents/sample_policy.pdf。
    doc = pymupdf.open()  # 作用：新建一个空 PDF 文档；效果：不传文件名就是内存里的新文档。
    page = doc.new_page()  # 作用：新建一页；效果：默认 A4 大小。
    y = 72  # 作用：光标纵坐标；效果：控制每行文字往下排。
    for line in SAMPLE_TEXT.splitlines():  # 作用：逐行写；效果：避免文字全挤成一坨。
        page.insert_text((50, y), line, fontsize=12, fontname="china-s")  # 作用：写一行字；效果：china-s 是 PyMuPDF 内置中文字体，不依赖系统字体。
        y += 20  # 作用：下移一行；效果：行距 20 磅。
    doc.save(path)  # 作用：落盘；效果：生成真实文件。
    doc.close()  # 作用：释放资源；效果：避免文件被占用。
    return path

def make_scan_png() -> Path:
    """作用：生成一张"扫描件风格"的图片；效果：用来测试图片 → OCR 这条分支。

    返回：
        Path: 生成的文件路径
    """
    path = DOC_DIR / "sample_scan.png"  # 作用：目标路径。
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 28)  # 作用：加载微软雅黑；效果：PIL 默认字体不支持中文，必须指定中文字体文件。
    img = Image.new("RGB", (900, 460), "white")  # 作用：造一张白底图；效果：RGB 三通道，宽 900 高 460。
    draw = ImageDraw.Draw(img)  # 作用：拿画笔；效果：可以在图上写字。
    y = 30  # 作用：起始纵坐标。
    for line in SAMPLE_TEXT.splitlines():  # 作用：逐行画字。
        draw.text((30, y), line, fill="black", font=font)  # 作用：写一行黑字。
        y += 42  # 作用：下移一行；效果：行距 42 像素。
    img.save(path)  # 作用：落盘。
    return path

def make_scan_pdf(png_path: Path) -> Path:
    """作用：把图片塞进 PDF，生成"没有文本层"的扫描件 PDF；效果：逼 loaders 走 OCR 兜底分支。

    参数：
        png_path: 上一步生成的图片路径
    返回：
        Path: 生成的文件路径
    """
    path = DOC_DIR / "sample_scan.pdf"  # 作用：目标路径。
    img = Image.open(png_path)  # 作用：打开图片；效果：拿到宽高尺寸。
    doc = pymupdf.open()  # 作用：新建空 PDF。
    page = doc.new_page(width=img.width, height=img.height)  # 作用：建一页与图片同尺寸的页面。
    page.insert_image(pymupdf.Rect(0, 0, img.width, img.height), filename=str(png_path))  # 作用：把图片铺满整页；效果：页面上只有像素、没有文字对象。
    doc.save(path)  # 作用：落盘。
    doc.close()  # 作用：释放资源。
    return path

if __name__ == "__main__":
    DOC_DIR.mkdir(exist_ok=True)  # 作用：确保目录存在；效果：exist_ok=True 表示已存在也不报错。
    paths = [make_text_pdf(), make_scan_png()]  # 作用：先生成 PDF 和 PNG。
    paths.append(make_scan_pdf(paths[1]))  # 作用：再用 PNG 造出扫描件 PDF。
    for p in paths:  # 作用：遍历结果。
        print(f"已生成：{p}  ({p.stat().st_size / 1024:.1f} KB)")  # 作用：打印文件与大小；效果：st_size 是字节数，除以 1024 得 KB。
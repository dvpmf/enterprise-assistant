# -*- coding: utf-8 -*-
"""作用：把 skills 目录变成一个可直接执行的"包"，效果：python -m skills 即可运行演示。"""
from skills import demo  # 作用：引入演示函数。

if __name__ == "__main__":
    demo()
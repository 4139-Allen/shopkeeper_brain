"""
导入流程节点基类
定义统一的节点接口规范，提供通用功能
"""

from abc import ABC





# 节点基类定义：ABC 抽象基类
class BaseNode(ABC):
    """
    导入流程节点 的基类
    所有节点都基础此基类，并实现 process 方法
    基类提供 统一的日志、任务追踪和错误处理
    """

    name: str   #子类必须赋值

    





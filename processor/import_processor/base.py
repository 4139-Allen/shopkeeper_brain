"""
导入流程节点基类
定义统一的节点接口规范，提供通用功能
"""

from abc import ABC, abstractmethod
from typing import TypeVar
import logging
from knowledge.processor.import_processor.config import ImportConfig, get_config
from knowledge.processor.import_processor.state import ImportGraphState

T = TypeVar("T")

# 节点基类定义：ABC 抽象基类
class BaseNode(ABC):
    """
    导入流程节点 的基类
    所有节点都基础此基类，并实现 process 方法
    基类提供 统一的日志、任务追踪和错误处理
    """

    name: str = None  #子类赋值

    def __init__(self, config: ImportConfig | None = None):
        """
        初始化节点，为每一个子节点配置属性
        Args:
            config: 配置对象，默认使用全局配置
        """
        self.config = config if config is not None else get_config()
        self.logger = logging.getLogger(f"import.{self.name}")

    def __call__(self, state:ImportGraphState) -> ImportGraphState:
        """
        节点执行入口
        langgraph 调用节点时会调用此方法，实现类能像方法一样调用
        附带提供统一的日志输出、任务追踪和异常处理
        Args:
            state: 图状态字典
        Returns:
            更新后的状态字典
        Raises:
            ImportProcessError: 节点执行失败时抛出
        """
        task_id = state.get("task_id", "")
        self.logger.info(f"----开始执行 {self.name} 节点----" + (f" task_id={task_id}" if task_id else ""))
        try:
            result = self.process(state)
            self.logger.info(f"----{self.name} 节点执行完成----")
            return result
        except Exception as e:
            self.logger.error(f"{self.name} 节点执行失败: {e}", exc_info=True)
            raise






    @abstractmethod
    def process(self, state:ImportGraphState):
        """
        节点核心处理逻辑
        """
        pass

    def log_step(self, step_name: str, message: str = ""):
        """
        记录步骤日志

        Args:
            step_name: 步骤名称
            message: 附加信息
        """
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)

#配置日志格式
def setup_logging(level: int = logging.INFO):
    """
    配置导入流程日志
    Args:
        level: 日志级别
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )



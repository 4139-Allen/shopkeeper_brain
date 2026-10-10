"""
查询流程节点基类
定义统一的节点接口规范，提供通用功能
"""

from abc import ABC, abstractmethod
import logging
from knowledge.processor.query_process.config import QueryConfig, get_config
from knowledge.processor.query_process.exceptions import QueryProcessError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.utils.task_util import add_running_task, add_done_task, get_task_status, get_done_task_list, \
    get_running_task_list


# 节点基类定义：ABC 抽象基类
class BaseNode(ABC):
    """
    导入流程节点 的基类
    所有节点都基础此基类，并实现 process 方法
    基类提供 统一的日志、任务追踪和错误处理
    """

    name: str = None  #子类赋值

    def __init__(self, config: QueryConfig | None = None):
        """
        初始化节点，为每一个子节点配置属性
        Args:
            config: 配置对象，默认使用全局配置
        """
        self.config = config if config is not None else get_config()
        self.logger = logging.getLogger(f"query.{self.name}")

    def __call__(self, state:QueryGraphState) -> QueryGraphState:
        """
        节点执行入口
        langgraph 调用节点时会调用此方法，实现类能像方法一样调用
        附带提供统一的日志输出、任务追踪和异常处理
        Args:
            state: 图状态字典
        Returns:
            更新后的状态字典
        Raises:
            QueryProcessError: 节点执行失败时抛出。
        """
        is_stream = state.get('is_stream')
        task_id = state.get('task_id')

        try:
            self.logger.info(f"--- {self.name} 开始 ---")

            if task_id:
                add_running_task(task_id, self.name)  # 当前正准备执行的节点加入
                # 如果是流式
                if is_stream:
                    self._push_progress(task_id)

            result = self.process(state)
            if task_id:
                add_done_task(task_id, self.name)  # 当前正准备执行的节点加入
                # 如果是流式
                if is_stream:
                    self._push_progress(task_id)

            self.logger.info(f"--- {self.name} 完成 ---")
            return result
        except Exception as e:
            self.logger.error(f"{self.name} 执行失败: {e}")
            raise QueryProcessError(
                message=str(e),
                node_name=self.name,
                cause=e
            )


    @abstractmethod
    def process(self, state:QueryGraphState):
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

    @staticmethod
    def _push_progress(task_id):
        """
        推送节点的进度(全量推所有进度)
        Args:
            task_id: 任务id
        """
        push_sse_event(task_id=task_id,
                       event=SSEEvent.PROGRESS,
                       data={
                           "status": get_task_status(task_id),
                           "done_list": get_done_task_list(task_id),
                           "running_list": get_running_task_list(task_id),
                       })

#配置日志格式
def setup_logging(level: int = logging.INFO):
    """
    配置导入流程日志
    Args:
        level: 日志级别,默认 INFO。
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )



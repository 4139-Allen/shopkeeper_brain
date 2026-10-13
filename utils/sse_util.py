"""
SSE 流式工具

生产者在工作线程（LangGraph 同步图靠 run_in_executor 跑），消费者在事件循环
（HTTP 响应生成器）。两者通过 asyncio.Queue 桥接，工作线程侧用
loop.call_soon_threadsafe 投递，保证线程安全。
"""
import asyncio
import dataclasses
import json
import logging
from typing import Dict, Any, AsyncGenerator

logger = logging.getLogger(__name__)


class SSEEvent:
    PROGRESS = "progress"  # 任务节点进度
    DELTA = "delta"        # LLM 流式输出增量
    FINAL = "final"        # 最终完整答案（同时是流的终止信号）


@dataclasses.dataclass
class _SSEStream:
    """一条 SSE 流的桥接对象：记录创建它的 loop，供工作线程安全投递"""
    loop: asyncio.AbstractEventLoop
    queue: "asyncio.Queue"


# Key: task_id, Value: _SSEStream
# 生命周期 = 一次 HTTP 请求（生产者与消费者同处一个请求），不再是跨请求状态
_streams: Dict[str, _SSEStream] = {}


def create_sse_queue(task_id: str, maxsize: int = 512) -> asyncio.Queue:
    """
    创建并注册一条 SSE 流。

    注意：必须在事件循环内调用（loop.call_soon_threadsafe 需要拿到当前 loop），
    所以要在 async 路由里创建，不能丢到工作线程里建。

    :param task_id: 任务id
    :param maxsize: 队列上限，满时丢弃并告警（防止生产端无限堆积）
    :return: 该任务的 asyncio.Queue
    """
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
    _streams[task_id] = _SSEStream(loop, q)
    return q


def remove_sse_queue(task_id: str) -> None:
    """移除指定任务的 SSE 流（不存在则忽略）"""
    _streams.pop(task_id, None)


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """打包 SSE 消息格式"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _safe_put(q: "asyncio.Queue", item: Dict[str, Any]) -> None:
    """
    在事件循环线程内投递事件。
    队满时丢弃并告警——不能让它抛异常，否则会在 loop 里变成未捕获异常。
    """
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        logger.warning("SSE 队列已满，丢弃一条事件")


def push_sse_event(task_id: str, event: str, data: Dict[str, Any]) -> None:
    """
    通过 task_id 推送事件到 SSE 流。

    调用方在工作线程里（节点代码），所以必须用 call_soon_threadsafe 跨线程投递，
    不能用 asyncio.Queue 的直接写入。
    """
    # 1. 获取 SSE 流
    stream = _streams.get(task_id)

    # 2. 流不存在（请求已结束/已清理）——显式记录，避免事件被静默丢弃
    if stream is None:
        logger.warning(f"SSE 流不存在，事件被丢弃 task_id={task_id} event={event}")
        return

    # 3. 线程安全投递到事件循环
    stream.loop.call_soon_threadsafe(_safe_put, stream.queue, {"event": event, "data": data})


async def sse_generator(task_id: str) -> AsyncGenerator:
    """
    流式输出结果的消费者
    1. 从 SSE 队列中获取事件（纯事件驱动：挂起等待，无轮询、无超时）
    2. 封装队列中的数据以及事件类型为 sse 协议的数据包格式
    3. 将封装好的数据包 yield 出去；收到 FINAL 即终止

    客户端断开会由 Starlette 取消本生成器（await queue.get 抛 CancelledError），
    因此不再需要主动轮询 request.is_disconnected()。

    :param task_id: 任务id
    :return: AsyncGenerator
    """
    # 1. 根据任务id 获取流对象
    stream = _streams.get(task_id)

    # 2. 校验
    if stream is None:
        logger.warning(f"SSE 流不存在，无法建立流 task_id={task_id}")
        return

    # 3. 消费队列
    try:
        while True:
            # 3.1 挂起等待事件（不占线程、不空转）
            msg = await stream.queue.get()

            # 3.2 获取事件类型 / 事件数据
            event_type = msg.get("event")
            event_data = msg.get("data")

            # 3.3 打包并 yield
            yield _sse_pack(event_type, event_data)

            # 3.4 FINAL 是最后一条事件，推完即止
            if event_type == SSEEvent.FINAL:
                return
    except (ConnectionResetError, BrokenPipeError):
        # 客户端中断 关闭了窗口或者浏览器
        return
    except asyncio.CancelledError:
        # 服务端中断 协程被取消 重新抛出，让外层知道它被成功取消
        raise
    finally:
        remove_sse_queue(task_id)

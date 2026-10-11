from pydantic import BaseModel, Field



class UploadResponse(BaseModel):
    """文件上传响应 POST / upload 返回"""
    # ... 是 Python 内置的 Ellipsis(省略号对象),在 Pydantic 里是一个约定俗成的特殊标记,含义是:这个字段必填,没有默认值。
    # 等价 message: str 只不过少了fastapi的doc文档说明description
    message: str = Field(..., description="响应消息")
    task_id: str = Field(..., description="任务ID")

class TaskStatusResponse(BaseModel):
    """任务状态响应  GET   /status/{task_id} 返回"""
    status: str = Field(..., description="任务状态")
    done_list: list[str] = Field(..., description="已完成的节点列表")
    running_list: list[str] = Field(..., description="正在运行的节点列表")
    durations: dict[str, float] = Field(..., description="各个节点耗时(秒)")





from pydantic import BaseModel, Field


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    status: str = Field(..., description="任务状态")
    done_list: list[str] = Field(..., description="已完成的节点")
    running_list: str = Field(..., description="正在运行的节点")


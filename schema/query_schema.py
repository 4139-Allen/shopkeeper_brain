from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str = Field(..., description="查询内容")
    session_id: str | None = Field(..., description="会话ID")
    is_stream: bool = Field(..., description="是否流式输出")

class QueryResponse(BaseModel):
    """查询响应模型"""
    message: str = Field(..., description="查询返回的内容")
    session_id: str = Field(..., description="会话ID")
    task_id: str = Field(..., description="任务ID")
    answer: str = Field(..., description="最终答案")

class StreamSubmitResponse(BaseModel):
    """流式提交响应模型"""
    message: str = Field(..., description="查询返回的流式内容")
    session_id: str = Field(..., description="会话ID")
    task_id: str = Field(..., description="任务ID")





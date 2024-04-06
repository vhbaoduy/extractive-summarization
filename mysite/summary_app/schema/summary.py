from pydantic import BaseModel


class SummaryRequest(BaseModel):
    content: str

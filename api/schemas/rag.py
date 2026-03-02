from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    question: str
    answer: str

class FullPipelineRequest(BaseModel):
    text: str
    question: str

class FullPipelineResponse(BaseModel):
    stored_disease: str | None
    question: str
    answer: str
    error: str | None = None

    
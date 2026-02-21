from pydantic import BaseModel,Field

class MedicalEntities(BaseModel):
    disease: str = Field(description="Name of the disease or medical condition")
    symptoms: list[str] = Field(description="List of associated symptoms")
    causes: list[str] = Field(description="List of causes or risk factors")
    treatments: list[str] = Field(description="List of recommended treatments")
    severity: str = Field(description="Severity : low | medium | high")
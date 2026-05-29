from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class ClauseType(str, Enum):
    penalitate = "penalitate"
    obligatie = "obligatie"
    drept = "drept"
    forta_majora = "forta_majora"
    confidentialitate = "confidentialitate"
    reziliere = "reziliere"
    date_personale = "date_personale"
    altele = "altele"


class RiskLevel(str, Enum):
    RIDICAT = "RIDICAT"
    MEDIU = "MEDIU"
    SCAZUT = "SCAZUT"
    CONFORM = "CONFORM"
    NECUNOSCUT = "NECUNOSCUT"


class PartyDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = ""
    cui_cnp: Optional[str] = None
    address: Optional[str] = None


class SectionDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    start_page: int = Field(ge=1)


class ClauseDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    section: str
    text: str
    page: int = Field(ge=1)
    type: ClauseType = ClauseType.altele


class DocumentMetadataDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = "Document juridic"
    page_count: int = Field(default=0, ge=0)
    parties: list[PartyDTO] = Field(default_factory=list)
    signing_date: Optional[str] = None
    effective_date: Optional[str] = None
    value: Optional[str] = None
    duration: Optional[str] = None


class ParsedDocumentDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metadata: DocumentMetadataDTO
    sections: list[SectionDTO] = Field(default_factory=list)
    clauses: list[ClauseDTO] = Field(default_factory=list)


class RetrievalResultDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    source: str
    score: float = Field(ge=0.0, le=1.0)


class RiskAssessmentDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clause_id: str
    risk_level: RiskLevel
    issues: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    context_was_empty: bool = False


class RecommendationDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clause_id: str
    original_text: str
    reformulated_text: str = ""
    explanation: str = ""
    sources: list[str] = Field(default_factory=list)
    candidates: Optional[list[str]] = None

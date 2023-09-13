from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List


class PersonIntel(BaseModel):
    summary: str = Field(description="Summary of the project")
    facts: List[str] = Field(description="Interesting facts about person")

    def to_dict(self):
        return {"summary": self.summary, "facts": self.facts}


person_intel_parser: PydanticOutputParser = PydanticOutputParser(
    pydantic_object=PersonIntel
)
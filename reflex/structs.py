from dataclasses import dataclass

@dataclass(frozen=True)
class Sample:
    head: str
    context: str
    tail: str
    question: str
    template: str


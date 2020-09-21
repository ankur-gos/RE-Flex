from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    head: str
    context: str
    tail: str
    question: str
    template: str


@dataclass(frozen=True)
class Triplet:
    head: str
    predicate: str
    tail: str
    context: str


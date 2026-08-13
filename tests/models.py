from pydantic import BaseModel


class User(BaseModel):
    id: int
    name: str
    age: int
    city: str
    score: float | None = None

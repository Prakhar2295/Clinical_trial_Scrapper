from pydantic import BaseModel, Field


class AddCategoryRequest(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of the new document category to add",
    )
    description: str = Field(
        default="",
        max_length=500,
        description="Optional description of this category",
    )


class AddCategoryResponse(BaseModel):
    status: str
    name: str
    all_categories: list[str]
    message: str


class ListCategoriesResponse(BaseModel):
    total: int
    categories: list[str]

from fastapi import APIRouter, HTTPException

from app.core.category_registry import category_registry
from app.schemas.category import AddCategoryRequest, AddCategoryResponse, ListCategoriesResponse

router = APIRouter()


@router.post("/categories", response_model=AddCategoryResponse)
def add_category(request: AddCategoryRequest) -> AddCategoryResponse:
    try:
        added_name = category_registry.add(request.name)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to add category: {exc}") from exc

    all_categories = category_registry.all()
    return AddCategoryResponse(
        status="created",
        name=added_name,
        all_categories=all_categories,
        message=f"Category '{added_name}' added successfully. Total categories: {len(all_categories)}",
    )


@router.get("/categories", response_model=ListCategoriesResponse)
def list_categories() -> ListCategoriesResponse:
    categories = category_registry.all()
    return ListCategoriesResponse(total=len(categories), categories=categories)

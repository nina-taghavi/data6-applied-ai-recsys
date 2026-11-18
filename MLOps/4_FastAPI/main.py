from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Mini Exercise 4: FastAPI Store & ML Predictor (+Upgrade)",
    description="Official FastAPI example from Slide 41 upgraded with CRUD endpoints, health check, and Iris flower prediction.",
    version="2.0.0",
)


# 1. Pydantic Data Validation Models (Slide 41)
class Item(BaseModel):
    name: str = Field(..., examples=["Sunflower Seed Packet"])
    price: float = Field(..., gt=0, examples=[4.99])
    is_offer: bool | None = Field(default=None, examples=[True])


class FlowerInput(BaseModel):
    sepal_length: float = Field(..., gt=0, examples=[5.1])
    sepal_width: float = Field(..., gt=0, examples=[3.5])
    petal_length: float = Field(..., gt=0, examples=[1.4])
    petal_width: float = Field(..., gt=0, examples=[0.2])


# In-memory store for the "+Upgrade" part of Mini Exercise 4
items_db: dict[int, Item] = {
    1: Item(name="Rose Bouquet", price=19.99, is_offer=True),
    2: Item(name="Tulip Bulb Pack", price=8.50, is_offer=False),
}


# 2. Base Endpoints from Slide 41
@app.get("/")
def read_root():
    """Root greeting endpoint from Slide 41."""
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    """
    Get an item by ID (path parameter) with an optional query parameter `q`.
    Upgraded to look up items in `items_db` if present.
    """
    stored_item = items_db.get(item_id)
    return {
        "item_id": item_id,
        "q": q,
        "item": stored_item,
    }


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    """
    Update or save an item by ID using Pydantic body validation (Slide 41).
    """
    items_db[item_id] = item
    return {"item_name": item.name, "item_id": item_id, "saved_item": item}


# 3. "+Upgrade" Endpoints (Slide 39 & Slide 42)
@app.get("/health")
def health_check():
    """Typical MLOps health endpoint (Slide 39) to check if the service is up."""
    return {"status": "healthy", "items_in_store": len(items_db)}


@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    """Upgraded endpoint to delete an item from the in-memory store."""
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    deleted = items_db.pop(item_id)
    return {"deleted_id": item_id, "deleted_item": deleted}


@app.post("/predict")
def predict_flower(features: FlowerInput):
    """
    Typical MLOps prediction endpoint (Slide 39): POST /predict.
    Uses a simple rule on petal measurements to classify Iris species.
    """
    if features.petal_length < 2.5:
        species = "setosa"
    elif features.petal_width < 1.8:
        species = "versicolor"
    else:
        species = "virginica"

    return {
        "predicted_species": species,
        "input_features": features,
    }

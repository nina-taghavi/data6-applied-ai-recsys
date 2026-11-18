# 🚀 Part 4: Model Deployment with FastAPI (Mini Exercise 4)

> **Course Reference:** `08 MLOps.pdf` — **Slides 37–42 (Mini Exercise 4: Run the FastAPI Example + Upgrade on Your Laptop)**  
> **Official Docs:** 🔗 [FastAPI Official Tutorial](https://fastapi.tiangolo.com/#example) & 🔗 [Interactive Swagger UI (`http://127.0.0.1:8000/docs`)](http://127.0.0.1:8000/docs)

---

## 💡 1. What Is the Problem? (In Simple Words)

You trained a great Machine Learning model in Python. Now, how does a mobile app, a website, or a customer actually **use** your model?
You can't ask customers to install Python and open a Jupyter Notebook!

**The Solution (Slides 38–40):** We wrap the model inside a **REST API** using **FastAPI**.
* A website or user sends input data to a URL (like `POST /predict`).
* Your FastAPI server checks that the data is valid (using **Pydantic**), runs the model, and sends back a JSON prediction in milliseconds!

### 3 Ways to Deploy a Model (Slide 38)

| Deployment Pattern | How It Works | Everyday Example | Pros & Cons |
| :--- | :--- | :--- | :--- |
| **1. Batch Inference** | Runs offline on a schedule (e.g., every night at 2 AM) on a huge table of data. | Monthly customer churn report; daily fraud report. | ✅ Simple and cheap.<br/>❌ Not instant. |
| **2. Real-Time Inference** | Waits for a user request and replies **immediately** (in milliseconds) via a **REST API**. | Chatbots, live recommender systems, uploading a flower photo to classify. | ✅ Instant answers.<br/>⚠️ Needs an API server (like **FastAPI**). |
| **3. Streaming** | Continuously processes a non-stop live stream of events. | Live credit-card tap fraud blocking; factory sensor anomaly alerts. | ✅ Handles high-speed live streams.<br/>⚠️ Most complex infrastructure. |

---

## 🗂️ 2. Files in This Folder (`4_FastAPI/`)

| File | What It Does |
| :--- | :--- |
| **[`main.py`](main.py)** | Implements the exact **Slide 41 FastAPI example** (`GET /`, `GET /items/{item_id}`, `PUT /items/{item_id}`) **plus the Slide 42 `+Upgrade`** (`GET /health`, `DELETE /items/{item_id}`, and an ML `POST /predict` endpoint for Iris flowers). |
| **[`test_main.py`](test_main.py)** | Automated **pytest** unit tests (Slide 55) that test every endpoint in `main.py`. |
| **`README.md`** | This simple study guide. |

---

## 🔍 3. Code Explanation (`main.py` — Slide 41 + Upgrade)

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# 1. Pydantic Model = Automatic Input Guard (Slide 41)
class Item(BaseModel):
    name: str
    price: float
    is_offer: bool | None = None

# 2. GET Endpoint (Read data)
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}

# 3. PUT Endpoint (Send/Update JSON data)
@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}
```

| Concept from Slide 41 | Simple Explanation |
| :--- | :--- |
| **`Uvicorn`** | The fast web server engine that listens on port `8000` and hands incoming requests to your FastAPI Python code. |
| **`Pydantic (BaseModel)`** | The **bouncer at the door**. If `item_id` is supposed to be an `int` or `price` is supposed to be a `float`, and someone sends `"price": "banana"`, Pydantic automatically blocks it and returns a clear error message! |
| **Path vs. Query Parameter** | In `GET /items/5?q=flower`:<br/>• `5` is inside the URL path (`item_id: int`).<br/>• `q=flower` comes after the `?` (`q: str \| None = None`). |
| **Our `+Upgrade` Additions** | 1. **In-Memory Store (`items_db`)**: `PUT /items/{id}` actually saves the item in memory so `GET /items/{id}` returns it!<br/>2. **`GET /health` (Slide 39)**: Standard MLOps health check endpoint.<br/>3. **`POST /predict` (Slide 39)**: Accepts 4 flower measurements (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) and predicts the Iris flower species! |

---

## 🚀 4. How to Run Mini Exercise 4 on Your Laptop (Slides 41–42)

### Step 1: Start the FastAPI Server
Open your terminal, activate `.venv`, and run the server in development mode (auto-reloads whenever you save `main.py`):

```bash
cd /Users/nooshin/Uni/Data/data6/data6/MLOps/4_FastAPI
source ../../.venv/bin/activate

# Start with Uvicorn (Development mode with auto-reload)
uvicorn main:app --reload --port 8000
```

### Step 2: Open the Automatic Interactive Docs (Slide 42)
While the server is running, open this link in your browser:
👉 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

FastAPI automatically generates an interactive webpage (**Swagger UI**) where you can click **"Try it out"** and **"Execute"** on every endpoint (`GET /`, `GET /items/{item_id}`, `PUT /items/{item_id}`, `GET /health`, `POST /predict`) right in your browser!

### Step 3: Run the Automated Unit Tests (Slide 55)
```bash
python -c "import test_main; test_main.test_read_root(); test_main.test_health_check(); test_main.test_read_and_update_item(); test_main.test_predict_flower(); print('All tests passed!')"
```

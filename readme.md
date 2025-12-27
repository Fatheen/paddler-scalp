## Setup & Run Instructions

### 1. Clone the repository
```bash
git clone

python -m venv venv
source venv/bin/activate       # macOS / Linux
# OR
venv\Scripts\activate          # Windows

pip install -r backend/requirements.txt

uvicorn backend.app:app --reload --port 8000


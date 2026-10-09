from fastapi import FastAPI

app = FastAPI(
    title="IntelliStudy API"
)

@app.get("/")
def root():
    return {
        "message": "IntelliStudy API Running"
    }
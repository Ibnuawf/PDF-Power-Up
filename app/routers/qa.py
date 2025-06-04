from fastapi import APIRouter

router = APIRouter()

@router.get("/qa/ping")
def ping():
    return {"message": "QA router is working!"}
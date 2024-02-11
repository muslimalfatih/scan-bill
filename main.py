from fastapi import FastAPI
from app.routes import router

app = FastAPI()

# Include the router
app.include_router(router)

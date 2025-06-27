from pydantic import BaseModel, EmailStr
from typing import Optional
from pymongo import MongoClient
import os

class User(BaseModel):
    username: str
    email: EmailStr
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

def get_database_client():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("MONGO_URI environment variable not set.")
    client = MongoClient(mongo_uri)
    return client

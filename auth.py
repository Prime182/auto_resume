from passlib.context import CryptContext
from users import UserInDB, get_database_client

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(username: str):
    client = get_database_client()
    db = client["ai_resume"]
    users_collection = db["user"]
    user_dict = users_collection.find_one({"username": username})
    client.close()
    if user_dict:
        return UserInDB(**user_dict)
    return None

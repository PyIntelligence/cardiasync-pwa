import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-cardiasync')
    DATABASE_URL = os.getenv('DATABASE_URL')
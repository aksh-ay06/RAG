import uuid 
from datetime import datetime, timezone
from sqlalchemy import JSON,Boolean, Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import UUID
from src.db.interfaces.postgresql import Base
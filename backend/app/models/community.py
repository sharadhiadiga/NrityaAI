from datetime import datetime
from sqlalchemy import DateTime
from sqlalchemy import Column, Integer, String
from app.database import Base

class Channel(Base):
    __tablename__ = "channels"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)

class UserChannel(Base):
    __tablename__ = "user_channels"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    channel_id = Column(Integer, nullable=False)

class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(String, nullable=False)
    user_id = Column(Integer, nullable=False)
    channel_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Comment(Base):
    __tablename__ = "comments"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(String, nullable=False)
    user_id = Column(Integer, nullable=False)
    post_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Like(Base):
    __tablename__ = "likes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    post_id = Column(Integer, nullable=False)
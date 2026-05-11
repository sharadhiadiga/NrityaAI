from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.community import Channel
from app.models.community import UserChannel
from app.models.community import Post
from app.models.community import Comment
from app.models.community import Like

router = APIRouter()

# DB connection
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Create Channel
@router.post("/channels")
def create_channel(name: str, description: str = "", db: Session = Depends(get_db)):
    new_channel = Channel(name=name, description=description)
    db.add(new_channel)
    db.commit()
    db.refresh(new_channel)
    return new_channel


# Get all channels
@router.get("/channels")
def get_channels(db: Session = Depends(get_db)):
    return db.query(Channel).all()

#join channel
@router.post("/join")
def join_channel(user_id: int, channel_id: int, db: Session = Depends(get_db)):
    join = UserChannel(user_id=user_id, channel_id=channel_id)
    db.add(join)
    db.commit()
    return {"message": "User joined channel"}

#leave channel
@router.post("/leave")
def leave_channel(user_id: int, channel_id: int, db: Session = Depends(get_db)):
    db.query(UserChannel).filter(
        UserChannel.user_id == user_id,
        UserChannel.channel_id == channel_id
    ).delete()
    
    db.commit()
    return {"message": "User left channel"}

#create posts
@router.post("/post")
def create_post(content: str, user_id: int, channel_id: int, db: Session = Depends(get_db)):
    new_post = Post(content=content, user_id=user_id, channel_id=channel_id)
    db.add(new_post)
    db.commit()
    db.refresh(new_post)
    return new_post

#get posts by channel
@router.get("/posts/{channel_id}")
def get_posts(
    channel_id: int,
    skip: int = 0,
    limit: int = 5,
    db: Session = Depends(get_db)
):
    return db.query(Post)\
        .filter(Post.channel_id == channel_id)\
        .order_by(Post.created_at.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()

#create comments
@router.post("/comment")
def create_comment(content: str, user_id: int, post_id: int, db: Session = Depends(get_db)):
    new_comment = Comment(content=content, user_id=user_id, post_id=post_id)
    db.add(new_comment)
    db.commit()
    db.refresh(new_comment)
    return new_comment

#get comments by post
@router.get("/comments/{post_id}")
def get_comments(post_id: int, db: Session = Depends(get_db)):
    return db.query(Comment).filter(Comment.post_id == post_id).all()

#like a post
@router.post("/like")
def like_post(user_id: int, post_id: int, db: Session = Depends(get_db)):
    like = Like(user_id=user_id, post_id=post_id)
    db.add(like)
    db.commit()
    return {"message": "Post liked"}

#unlike a post
@router.post("/unlike")
def unlike_post(user_id: int, post_id: int, db: Session = Depends(get_db)):
    db.query(Like).filter(
        Like.user_id == user_id,
        Like.post_id == post_id
    ).delete()

    db.commit()
    return {"message": "Post unliked"}

#count likes
@router.get("/likes/{post_id}")
def get_likes(post_id: int, db: Session = Depends(get_db)):
    count = db.query(Like).filter(Like.post_id == post_id).count()
    return {"likes": count}

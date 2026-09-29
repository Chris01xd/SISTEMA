from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.authentication.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
)
from app.database.session import get_db
from app.models.user import Role, User
from app.schemas.auth import LoginRequest, Token, UserCreate, UserOut

router = APIRouter()


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, db: Session = Depends(get_db)) -> UserOut:
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    role = db.query(Role).filter(Role.name == payload.role_name).first()
    if role is None:
        default_permissions = (
            {"*": True}
            if payload.role_name == "admin"
            else {"simulation:run": True, "reports:generate": True}
        )
        role = Role(name=payload.role_name, permissions=default_permissions)
        db.add(role)
        db.commit()
        db.refresh(role)

    user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
        role_id=role.id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=Token)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> Token:
    user = db.query(User).filter(User.email == payload.email).first()
    if user is None or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return Token(
        access_token=create_access_token(user.email),
        refresh_token=create_refresh_token(user.email),
    )

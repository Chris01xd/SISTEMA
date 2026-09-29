"""FastAPI dependencies for authentication and RBAC."""
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.authentication.security import decode_token
from app.database.session import get_db
from app.models.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_token(token)
    if payload is None or "sub" not in payload:
        raise credentials_exception

    user = db.query(User).filter(User.email == payload["sub"]).first()
    if user is None or not user.is_active:
        raise credentials_exception
    return user


def require_permission(permission: str):
    def _dependency(user: User = Depends(get_current_user)) -> User:
        perms = (user.role.permissions or {}) if user.role else {}
        if not perms.get(permission, False) and not perms.get("*", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing permission: {permission}",
            )
        return user

    return _dependency


def require_any_permission(*permissions: str):
    """Like require_permission, but passes if the user holds ANY of the given
    keys (or the "*" wildcard) — used where more than one role is legitimately
    allowed to act (e.g. editing a CRISP-DM phase whose owning discipline
    varies by phase_number)."""

    def _dependency(user: User = Depends(get_current_user)) -> User:
        perms = (user.role.permissions or {}) if user.role else {}
        if perms.get("*", False) or any(perms.get(p, False) for p in permissions):
            return user
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Missing permission: one of {permissions}",
        )

    return _dependency


def has_permission(user: User, permission: str) -> bool:
    perms = (user.role.permissions or {}) if user.role else {}
    return bool(perms.get("*", False) or perms.get(permission, False))

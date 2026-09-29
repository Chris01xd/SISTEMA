from sqlalchemy import Boolean, ForeignKey, JSON, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPKMixin


class Role(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "roles"

    name: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    # e.g. {"simulation:run": true, "reports:generate": true, "twin:admin": true}
    permissions: Mapped[dict] = mapped_column(JSON, default=dict)

    users: Mapped[list["User"]] = relationship(back_populates="role")


class User(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str] = mapped_column(String(255), default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    role_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("roles.id"))
    role: Mapped["Role"] = relationship(back_populates="users")

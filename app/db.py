from datetime import UTC, datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Prediction(db.Model):
    __tablename__ = "predictions"
    id         = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, index=True,
                           default=lambda: datetime.now(UTC))
    filename   = db.Column(db.String(255), nullable=False)   # sanitized name only; no image stored
    label      = db.Column(db.String(32), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    p_real     = db.Column(db.Float, nullable=False)
    threshold  = db.Column(db.Float, nullable=False)
    model_name = db.Column(db.String(128), nullable=False)
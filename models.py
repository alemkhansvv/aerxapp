from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    hashed_password = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(120), nullable=True)
    address = db.Column(db.String(255), nullable=True)
    country = db.Column(db.String(120), nullable=True)
    city = db.Column(db.String(120), nullable=True)
    postal_code = db.Column(db.String(20), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    birthdate = db.Column(db.Date, nullable=True)

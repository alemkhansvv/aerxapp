from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User  # Импортируем db и User из models.py

# Регистрация нового пользователя
def register_user(email, password):
    password_hash = generate_password_hash(password)
    new_user = User(email=email, hashed_password=password_hash)
    db.session.add(new_user)
    db.session.commit()

# Функция для входа пользователя
def login_user(email, password):
    user = User.query.filter_by(email=email).first()
    if user and check_password_hash(user.hashed_password, password):
        return user
    return None

# Проверка, существует ли пользователь
def user_exists(email):
    return User.query.filter_by(email=email).first() is not None

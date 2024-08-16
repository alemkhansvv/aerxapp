import os
from flask import Flask
from flask_migrate import Migrate
from models import db  # Импортируем вашу базу данных из models.py
from app import app  # Убедитесь, что вы правильно импортируете ваше приложение

# Получаем значение DATABASE_URL из переменной окружения
database_url = os.getenv('DATABASE_URL', 'sqlite:///instance/database.db')

# Корректируем URL, если он начинается с postgres://
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
migrate = Migrate(app, db)

if __name__ == '__main__':
    app.run()

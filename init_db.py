import sqlite3

def init_db():
    conn = sqlite3.connect('instance/database.db')  # Подключение к правильному пути
    c = conn.cursor()

    # Создание таблицы пользователей
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()  # Сохранение изменений
    conn.close()  # Закрытие соединения

if __name__ == "__main__":
    init_db()
    print("Database initialized and users table created.")

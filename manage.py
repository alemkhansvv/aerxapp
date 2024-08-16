from flask import Flask
from flask_migrate import Migrate
from models import db

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'your_database_uri'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
migrate = Migrate(app, db)

if __name__ == '__main__':
    from flask.cli import with_appcontext
    import click

    @click.command(name='db_upgrade')
    @with_appcontext
    def db_upgrade():
        """Миграция базы данных."""
        from flask_migrate import upgrade
        upgrade()

    app.cli.add_command(db_upgrade)
    app.run()

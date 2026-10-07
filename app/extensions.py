from authlib.integrations.flask_client import OAuth
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import LoginManager
from flask_mail import Mail
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import CSRFProtect

db = SQLAlchemy()
login_manager = LoginManager()
migrate = Migrate()
oauth = OAuth()
mail = Mail()
csrf = CSRFProtect()

# In-memory storage: fine for a single dev/small-deploy process. If this ever runs
# behind multiple gunicorn workers, point storage_uri at a shared Redis instance
# instead, or each worker enforces its own separate limit.
limiter = Limiter(key_func=get_remote_address, storage_uri="memory://",
                  default_limits=["1000 per hour"])

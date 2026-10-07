"""Entrypoint for the SaaS app.

    python wsgi.py                  dev server, http://127.0.0.1:5000
    flask --app wsgi run             same, via the Flask CLI
    gunicorn -w 2 wsgi:app            production (Render/Railway)

Loads .env (if present) before the app factory reads its config from the
environment - see .env.example for what to set.
"""

from dotenv import load_dotenv

load_dotenv()

from app import create_app  # noqa: E402  (must follow load_dotenv())

app = create_app()

if __name__ == "__main__":
    import os
    app.run(debug=False, port=int(os.environ.get("PORT", 5000)))

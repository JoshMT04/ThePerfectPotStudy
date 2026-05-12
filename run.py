
from dotenv import load_dotenv
load_dotenv()
load_dotenv('.env.local', override=True)  # local overrides (e.g. REDIS_URL) — not committed

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run()
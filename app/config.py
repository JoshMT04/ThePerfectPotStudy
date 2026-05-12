# config.py
import os
import redis

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')

    # Pick Redis URL in order of preference
    redis_url = (
        os.environ.get('REDIS_TLS_URL')
        or os.environ.get('HEROKU_REDIS_OLIVE_URL')
        or os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    )

    # For Heroku Redis TLS (rediss://), disable cert verification via URL param.
    # redis-py 6.x removed ssl_cert_reqs as a direct kwarg; use query param instead.
    if redis_url.startswith('rediss://'):
        sep = '&' if '?' in redis_url else '?'
        redis_url = f"{redis_url}{sep}ssl_cert_reqs=none"

    # Single Redis connection
    redis_connection = redis.from_url(
        redis_url,
        max_connections=20,
        retry_on_timeout=True,
        socket_keepalive=True,
        socket_keepalive_options={},
        health_check_interval=30
    )

    # Flask-Session config
    SESSION_TYPE = 'redis'
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    SESSION_KEY_PREFIX = 'theperfectpot:'
    SESSION_REDIS = redis_connection
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour session timeout
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SECURE = True if os.environ.get('ENVIRONMENT') == 'production' else False

    # Flask-Caching config
    CACHE_TYPE = 'redis'
    CACHE_REDIS_URL = redis_url

    LANGUAGES = ['en', 'fr', 'zh', 'ja', 'it', 'es', 'pt']

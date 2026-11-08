# The whole demo in one container: FastAPI server + viewer + pipeline.
# Used by `docker compose up` (with PostgreSQL) and by deploy/cloudrun.sh (SQLite, public read-only mode).
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8000
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
# with DATABASE_URL (compose) seed PostgreSQL from the Excel pack; without it the server seeds SQLite by itself.
# A fresh clone without the built scene downloads the splat and builds it on the first start.
CMD ["sh", "-c", "if [ -n \"$DATABASE_URL\" ]; then python seed_db.py; fi && python server.py"]

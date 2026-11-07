# The whole demo in one container: FastAPI server + viewer + pipeline. Used by `docker compose up`.
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8000
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
# seed the evidence tables from the Excel pack, then start; the first start also downloads the splat and builds the scene
CMD ["sh", "-c", "python seed_db.py && python server.py"]

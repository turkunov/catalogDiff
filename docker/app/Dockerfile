FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace/app

# torch built for CUDA 12.6: the host driver (560) does not support newer CUDA.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126

# Dependencies are installed from a copy; at runtime the repository is mounted over /workspace/app.
COPY pyproject.toml ./
COPY pseudolabel ./pseudolabel
RUN pip install --no-cache-dir -e ".[encoder,dev]"

CMD ["python", "-m", "pseudolabel.app.main"]

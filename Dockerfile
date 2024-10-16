# Используем базовый образ с поддержкой Python 3.9 и OpenCV
FROM python:3.9-slim

# Устанавливаем зависимости системы для OpenCV и других библиотек
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ninja-build \
    && apt-get clean

# Устанавливаем зависимости проекта
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем Detectron2 из репозитория GitHub
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Копируем исходный код проекта внутрь контейнера
COPY . /app
WORKDIR /app

# Задаем команду по умолчанию для запуска Python-скрипта
ENTRYPOINT ["python", "code/detect_people.py"]

FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ninja-build \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
# (Note: Mounting volume in docker-compose for development)
COPY . .

# Default command
CMD ["python", "train.py", "--help"]

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev

# Copy the current directory
COPY . /app

# Set working directory
WORKDIR /app

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 7860

# Start the app using gunicorn
CMD ["gunicorn", "--workers", "2", "--timeout", "120", "--bind", "0.0.0.0:7860", "app:server"]

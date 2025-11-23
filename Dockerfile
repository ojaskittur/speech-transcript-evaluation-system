# Use Python 3.11 (matches your local setup)
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# 1. Install Java (System dependency for language-tool)
# We do this manually here because packages.txt is ignored in Docker spaces
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk-headless && \
    rm -rf /var/lib/apt/lists/*

# 2. Copy all your files into the container
COPY . .

# 3. Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 4. Create a specific user (Hugging Face requirement for security)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 5. Expose port 7860 (Hugging Face specifically listens on this port)
EXPOSE 7860

# 6. Run Streamlit pointing to that specific port
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
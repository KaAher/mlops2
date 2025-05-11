FROM python:3.9

WORKDIR /mlops_pro  # Set working directory inside the container
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY req_dev.txt /req_dev.txt
RUN pip install --upgrade pip && pip install -r /req_dev.txt  

# Copy all project files
COPY . .

# Ensure model is in the correct location
COPY models/trained.h5 /mlops_pro/models/trained.h5

# Copy frontpage.py
COPY webapp/pages/frontpage.py /mlops_pro/webapp/pages/frontpage.py

EXPOSE 8501  

CMD ["streamlit", "run", "/mlops_pro/webapp/pages/frontpage.py", "--server.port=8501", "--server.address=0.0.0.0"]
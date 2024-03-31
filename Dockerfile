FROM python:3.10.6-slim

# Install app
COPY . /usr/app
WORKDIR /usr/app

# Install dependencies
RUN pip install --upgrade pip && pip install -r Flask==2.3.2

# Run Battlesnake
CMD [ "python", "main.py" ]

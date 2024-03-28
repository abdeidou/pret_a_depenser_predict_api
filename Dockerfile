# Utilise l'image Python officielle et légère pour Python 3.12
FROM python:3.12

# Permet aux instructions et aux messages de journalisation d'apparaître immédiatement dans les logs Knative
ENV PYTHONUNBUFFERED True

# Copie le code local dans l'image du conteneur.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . .

# Installe les dépendances de production.
RUN pip install -r requirements.txt

# Exécute le service web au démarrage du conteneur avec le serveur web gunicorn
CMD exec gunicorn --bind :${PORT:-8080} --workers 1 --threads 8 --timeout 0 main:app

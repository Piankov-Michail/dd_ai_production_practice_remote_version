# dd_ai_production_practice_remote_version

### Create docker network: <br><br><pre>docker network create flowise-network</pre>

### Launch Flowise: <br><br><pre>docker run -d --network=flowise-network --name flowise -p 3000:3000 flowise</pre>

### [Agentflow](https://github.com/Piankov-Michail/dd_ai_production_practice_remote_version/blob/main/Agentflow.json) for importation <br>

### Set values NVIDIA_KEY, FLOWISE_URL and TELEGRAM_TOKEN in .env <br>
### Launch telegram-bot: <br><br> <pre>docker-compose build --pull</pre> <pre>docker-compose up -d</pre>

# dd_ai_production_practice_remote_version

### Launch Flowise: <br><br> <pre>docker build --no-cache -t flowise .</pre> <pre>docker run -d --network=host --name flowise</pre>

### [Agentflow](https://github.com/Piankov-Michail/dd_ai_production_practice_remote_version/blob/main/Agentflow.json) for importation <br>

### Set values NVIDIA_KEY, FLOWISE_URL and TELEGRAM_TOKEN in bot.py <br>
### Launch telegram-bot: <br><br> <pre>docker build -t ai-bot .</pre> <pre>docker run -d --rm --network=host ai-bot</pre>

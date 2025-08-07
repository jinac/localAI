# localAI
## Setup
### Step 1
Create and fill out '.env' file from env.example. If settings are fine, cp 'env.example' to '.env'.

### Step 2
Bring up docker container for ollama (to server models) and open-webui (for UX and everything else)
'docker-compose up'

(Optional): When you want to shutdown, you can invoke 'docker-compose down'

## Add models
After container is up, we can send commands to the ollama container to pull models.
For example, to get gpt-oss:20b, run the command:
'docker exec -it ollama ollama pull gpt-oss:20b'
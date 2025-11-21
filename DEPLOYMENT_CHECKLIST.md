# Railway Deployment Checklist

Quick reference checklist for deploying to Railway.

## Pre-Deployment

- [x] Dockerfile created
- [x] .dockerignore created
- [x] railway.json created
- [x] DEPLOYMENT.md created with instructions
- [x] Logging updated for container environment
- [x] .gitignore excludes sensitive data

## Railway Setup

- [ ] Create Railway account
- [ ] Connect GitHub repository
- [ ] Create new project from GitHub repo
- [ ] Railway auto-detects Dockerfile

## Environment Variables (Required)

- [ ] `DISCORD_TOKEN` - Discord bot token
- [ ] `DISCORD_CLIENT_ID` - Discord application client ID
- [ ] `BOT_OWNER_ID` - Your Discord user ID
- [ ] `AI_DEFAULT_PROVIDER` - "openai", "anthropic", or "gemini"
- [ ] At least one AI provider key:
  - [ ] `OPENAI_API_KEY` OR
  - [ ] `ANTHROPIC_API_KEY` OR
  - [ ] `GEMINI_API_KEY`

## Persistent Storage (Volumes)

- [ ] Mount volume: `/app/data` (SQLite + ChromaDB)
- [ ] Mount volume: `/root/.cache/huggingface` (ML models)

## Verification

- [ ] Build completes successfully
- [ ] Bot connects to Discord (check logs)
- [ ] ChromaDB initializes successfully
- [ ] Test command: `!help`
- [ ] Test RAG: `!ask <question>`
- [ ] Verify data persists after restart

## Post-Deployment

- [ ] Monitor logs for errors
- [ ] Check resource usage
- [ ] Set up monitoring/alerts if needed
- [ ] Document any custom configurations

## Troubleshooting

If issues occur, check:
1. Railway logs for error messages
2. All required environment variables are set
3. Volumes are mounted correctly
4. Discord token is valid
5. AI provider keys are valid

See `DEPLOYMENT.md` for detailed troubleshooting guide.


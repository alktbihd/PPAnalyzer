# Add GitHub Secret for Frontend Deployment

## Step 1: Add GitHub Secret

1. Go to: https://github.com/alktbihd/PPAnalyzer/settings/secrets/actions
2. Click **"New repository secret"**
3. **Name**: `AZURE_STATIC_WEB_APPS_API_TOKEN`
4. **Value**: `808d68cb81e6f088a3b306132699a0c607e83b3c9d0401d893d1e0b6d623f27003-6cf55001-d035-49b8-addb-0b4b2a1441f400311180b1a8df03`
5. Click **"Add secret"**

## Step 2: Trigger Deployment

After adding the secret, push any change to the `frontend/` directory to trigger deployment:

```bash
cd /Users/saeedalketbi/Desktop/Senior\ design\ 2/project/ppaudit
git add .
git commit -m "Trigger frontend deployment"
git push
```

## Frontend URLs

- **Production**: https://salmon-rock-0b1a8df03.3.azurestaticapps.net
- **Backend API**: https://ppanalyzer-backend.azurewebsites.net

The frontend is configured to use the production backend URL automatically.


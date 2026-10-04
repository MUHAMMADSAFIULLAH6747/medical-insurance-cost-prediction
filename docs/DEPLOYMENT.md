# Deployment Guide (Phase 8)

## A. Local app (already working)

```bash
cd medical-insurance-cost-prediction
pip install -r requirements.txt
python src/train_model.py
streamlit run app.py
```

Open: `http://localhost:8501`

## B. Publish on GitHub

1. Authenticate GitHub CLI:
   ```bash
   gh auth login
   ```
2. Create and push the repository:
   ```bash
   gh repo create medical-insurance-cost-prediction --public --source=. --remote=origin --push
   ```
3. Capture screenshots:
   - repository home → `docs/screenshots/01_github_repo.png`
   - rendered README → `docs/screenshots/02_readme.png`

## C. Deploy on Streamlit Community Cloud

1. Go to [https://share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app**
4. Select repository: `medical-insurance-cost-prediction`
5. Main file path: `app.py`
6. Click **Deploy**
7. Copy the public URL into `README.md`
8. Capture:
   - deployed app → already have `03_deployed_app.png` (update with cloud URL if needed)
   - working prediction → already have `04_working_prediction.png`

## D. Submission checklist

- [x] Problem, dataset, preprocessing, model, evaluation documented
- [x] Reproducible project structure
- [x] Trained model + metrics artifacts
- [x] Streamlit prediction app
- [x] GitHub repository published
- [x] Screenshots: GitHub repo, README, app, working prediction, structure
- [ ] Streamlit Community Cloud authorization + public `.streamlit.app` URL

One-click deploy (after signing into Streamlit with GitHub):
https://share.streamlit.io/deploy?repository=MUHAMMADSAFIULLAH6747%2Fmedical-insurance-cost-prediction&branch=main&mainModule=app.py

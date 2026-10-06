# Publishing this repo to GitHub

Follow these steps once. They turn this folder into a public GitHub repository.

## 1. Personalisation — already done

The repo is ready to publish as-is: the author name (Johann Malto), the LICENSE copyright, the
clone/remote URLs (`j-wulfe`), the report figures, and the deck title slides are all filled in.
Nothing needs editing before you push — just make sure your GitHub account username is
`j-wulfe` (if it differs, update the URLs in Sections 2–3 below).

## 2. Create the repository on GitHub

1. Go to https://github.com/new
2. Repository name: `fraud-detection` (or your choice)
3. Visibility: **Public**
4. Do **not** initialise with a README/License/.gitignore (this folder already has them).
5. Click **Create repository**. Copy the URL it shows, e.g.
   `https://github.com/j-wulfe/fraud-detection.git`

## 3. Push from your machine

From inside this folder:

```bash
git init
git add .
git commit -m "Initial commit: fraud-detection capstone"
git branch -M main
git remote add origin https://github.com/j-wulfe/fraud-detection.git
git push -u origin main
```

If `git push` asks for a password, use a **Personal Access Token** (GitHub → Settings →
Developer settings → Personal access tokens), not your account password.

## 4. Verify

- The dataset (`data/creditcard.csv`) and trained models (`models/*.joblib`) are **ignored by
  git on purpose** (see `.gitignore`) — your repo should show the `README.md` files in those
  folders but not the large artifacts. That is correct open-source practice.
- Check that `notebooks/`, `src/`, and `reports/` render on GitHub.

## 5. (Optional) Polish your GitHub profile

- Add a short repo **description** and **topics** (`machine-learning`, `fraud-detection`,
  `imbalanced-data`, `xgboost`) on the repo's main page.
- Pin the repo to your profile.
- A clear README (already included) is the single biggest factor in how the project reads.

---

**Deliverable for Step 7:** the public repo URL, which contains the final report
(`reports/Final_Report.md`) and the reproducible code (`src/` + `notebooks/`).

# GitHub Repository Setup Instructions

## Step 1: Create Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `FinQA-Mini-Project` (or your preferred name)
3. Description: "Explainable Quantitative Reasoning for Financial Reports using FinQA"
4. Choose **Public** or **Private** (your choice)
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## Step 2: Connect Local Repository to GitHub

After creating the repo on GitHub, run these commands:

```bash
# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/FinQA-Mini-Project.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 3: Add Collaborators

### Option A: Via GitHub Web Interface (Recommended)

1. Go to your repository on GitHub
2. Click on **Settings** tab
3. Click on **Collaborators** in the left sidebar
4. Click **Add people**
5. Add these usernames:
   - `aku134`
   - `0xlel0uch`
6. They will receive email invitations to collaborate

### Option B: Via GitHub CLI

If you have GitHub CLI installed:

```bash
gh repo add-collaborator YOUR_USERNAME/FinQA-Mini-Project aku134
gh repo add-collaborator YOUR_USERNAME/FinQA-Mini-Project 0xlel0uch
```

## Step 4: Verify Setup

After your teammates accept the invitations, they can clone the repo:

```bash
git clone https://github.com/YOUR_USERNAME/FinQA-Mini-Project.git
cd FinQA-Mini-Project
```

## Current Repository Contents

✅ **Documentation:**
- `IMPLEMENTATION_PLAN.md` - Detailed 6-phase implementation plan
- `TECHNICAL_GUIDE.md` - Technical specifications
- `PROJECT_SUMMARY.md` - Quick reference guide
- `README.md` - Project overview

✅ **Starter Code:**
- `src/data_loader.py` - Dataset loading utilities
- `src/executor.py` - Program execution engine
- `src/evaluate.py` - Evaluation metrics

✅ **Project Infrastructure:**
- `requirements.txt` - Dependencies
- `setup.sh` - Setup script
- `.gitignore` - Git ignore rules

## Next Steps for Team

1. **Clone the repository** (for teammates)
2. **Set up environment**: Run `./setup.sh` or follow README
3. **Download dataset**: Clone FinQA repo and copy dataset files
4. **Start implementation**: Follow `IMPLEMENTATION_PLAN.md`
5. **Divide work**: Assign phases to team members

## Suggested Work Division

- **Team Member 1**: Retriever module (Phase 2.1)
- **Team Member 2**: Generator module (Phase 2.2)
- **Team Member 3**: Specialist model (Phase 3) + ICL model (Phase 4)
- **All**: Evaluation and comparison (Phase 5), Documentation (Phase 6)

## Git Workflow Recommendations

1. Create feature branches: `git checkout -b feature/retriever-module`
2. Commit frequently with clear messages
3. Push to GitHub regularly
4. Create pull requests for major features
5. Review each other's code before merging

## Important Notes

- The dataset files (`data/*.json`) are in `.gitignore` - each team member needs to download them separately
- Model checkpoints are also ignored - use GitHub Releases or external storage for large files
- Keep the repository organized and follow the project structure


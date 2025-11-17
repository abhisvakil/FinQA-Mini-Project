#!/bin/bash

# GitHub Repository Setup Script
# This script helps you connect your local repo to GitHub

echo "=========================================="
echo "GitHub Repository Setup"
echo "=========================================="
echo ""

# Check if remote already exists
if git remote | grep -q "^origin$"; then
    echo "⚠️  Remote 'origin' already exists."
    echo "Current remote URL:"
    git remote get-url origin
    echo ""
    read -p "Do you want to update it? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting..."
        exit 1
    fi
    git remote remove origin
fi

# Get repository details
echo "Please provide the following information:"
echo ""
read -p "Your GitHub username: " GITHUB_USERNAME
read -p "Repository name (default: FinQA-Mini-Project): " REPO_NAME
REPO_NAME=${REPO_NAME:-FinQA-Mini-Project}

echo ""
echo "Repository will be created at:"
echo "https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
echo ""

read -p "Have you created the repository on GitHub? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Please create the repository first:"
    echo "1. Go to: https://github.com/new"
    echo "2. Repository name: ${REPO_NAME}"
    echo "3. Description: Explainable Quantitative Reasoning for Financial Reports using FinQA"
    echo "4. Choose Public or Private"
    echo "5. DO NOT initialize with README, .gitignore, or license"
    echo "6. Click 'Create repository'"
    echo ""
    read -p "Press Enter when you've created the repository..."
fi

# Add remote
echo ""
echo "Adding remote origin..."
git remote add origin https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git

# Push to GitHub
echo "Pushing to GitHub..."
git branch -M main
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo ""
    echo "Next steps:"
    echo "1. Go to: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}/settings/access"
    echo "2. Click 'Add people' under 'Collaborators'"
    echo "3. Add these usernames:"
    echo "   - aku134"
    echo "   - 0xlel0uch"
    echo "4. They will receive email invitations"
    echo ""
else
    echo ""
    echo "❌ Failed to push to GitHub."
    echo "Please check:"
    echo "1. Repository exists on GitHub"
    echo "2. You have push access"
    echo "3. Your GitHub credentials are configured"
    echo ""
fi


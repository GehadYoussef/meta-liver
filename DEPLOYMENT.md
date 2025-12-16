# Meta Liver - Streamlit Cloud Deployment Guide

## 🚀 Deploy to Streamlit Cloud (Free)

### Step 1: Push to GitHub

Your app is ready to deploy! Follow these steps:

1. **Create a new repository on GitHub:**
   - Go to https://github.com/new
   - Repository name: `meta-liver` (or any name you prefer)
   - Description: "Interactive Liver Genomics Analysis Platform"
   - Make it **Public** (required for free Streamlit Cloud)
   - Click "Create repository"

2. **Push your code to GitHub:**

```bash
cd /home/ubuntu/meta_liver

# Add remote origin (replace YOUR_USERNAME with GehadYoussef)
git remote add origin https://github.com/GehadYoussef/meta-liver.git

# Rename branch to main (Streamlit Cloud prefers main)
git branch -M main

# Push to GitHub
git push -u origin main
```

You'll be prompted for GitHub credentials. Use:
- **Username**: GehadYoussef
- **Password**: Your GitHub personal access token (create one at https://github.com/settings/tokens)

### Step 2: Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud:**
   - Visit https://streamlit.io/cloud
   - Click "Sign up" or "Sign in" with your GitHub account
   - Authorize Streamlit to access your repositories

2. **Create a new app:**
   - Click "New app"
   - Select:
     - **Repository**: GehadYoussef/meta-liver
     - **Branch**: main
     - **Main file path**: streamlit_app.py
   - Click "Deploy"

3. **Wait for deployment:**
   - Streamlit will build and deploy your app
   - Takes 2-5 minutes for first deployment
   - You'll get a permanent URL like: `https://meta-liver-gehadyoussef.streamlit.app`

### Step 3: Share Your App

Once deployed, you'll have a permanent public URL:
```
https://meta-liver-gehadyoussef.streamlit.app
```

Share this URL with anyone to access your Meta Liver app!

---

## 📊 Important: Data Files

**⚠️ NOTE**: The data files are NOT included in the GitHub repository (they're too large and in .gitignore).

### Solution: Upload Data to Streamlit Cloud

After deployment, you need to add your data files:

1. **Option A: Use Streamlit Secrets (Recommended)**
   - Go to your app settings on Streamlit Cloud
   - Click "Secrets" 
   - Add your data files or paths

2. **Option B: Host Data Externally**
   - Upload data to cloud storage (Google Drive, AWS S3, etc.)
   - Modify `streamlit_app.py` to load from URL instead of local path

3. **Option C: Include Data in Repository**
   - Remove data files from `.gitignore`
   - Push data to GitHub (if < 100MB total)
   - Data will be included in deployment

### Modify App for Cloud Deployment

If using external data, update the data loading path in `streamlit_app.py`:

```python
# Change from:
data_dir = Path("/home/ubuntu/upload")

# To:
data_dir = Path("./data")  # or URL to cloud storage
```

---

## 🔄 Automatic Updates

After initial deployment, any changes you push to GitHub will automatically update your live app:

```bash
# Make changes to streamlit_app.py
# Commit and push
git add streamlit_app.py
git commit -m "Update: Add new feature"
git push origin main

# Your app updates automatically within 1-2 minutes!
```

---

## 💰 Streamlit Cloud Pricing

- **Free Tier**: 
  - 1 public app
  - Unlimited viewers
  - 1GB storage
  - Limited compute
  - Perfect for this use case!

- **Pro Tier** (if needed):
  - Multiple apps
  - Custom domains
  - Priority support
  - $10/month

---

## 🆘 Troubleshooting

### App won't load
- Check GitHub repository is public
- Verify `streamlit_app.py` exists in root directory
- Check logs in Streamlit Cloud dashboard

### Data files not found
- Upload data files to cloud storage
- Update data loading paths in app
- Or include data in GitHub repository

### Slow performance
- Streamlit Cloud free tier has limited resources
- Upgrade to Pro tier for better performance
- Or optimize data loading with caching

---

## 📝 Quick Checklist

- [ ] Create GitHub repository
- [ ] Push code to GitHub
- [ ] Sign up for Streamlit Cloud
- [ ] Connect GitHub account
- [ ] Deploy app
- [ ] Share permanent URL
- [ ] Upload data files (if needed)
- [ ] Test all features

---

## 🎯 Next Steps

1. **Create GitHub repository** with your username
2. **Push the code** using git commands above
3. **Deploy on Streamlit Cloud** (takes 5 minutes)
4. **Share the URL** with your team

Your Meta Liver app will be live permanently! 🎉

---

## 📞 Support

- **Streamlit Cloud Docs**: https://docs.streamlit.io/streamlit-cloud
- **GitHub Help**: https://docs.github.com
- **Streamlit Community**: https://discuss.streamlit.io


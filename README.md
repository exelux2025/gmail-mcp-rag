### Installation

1. **Create and activate a virtual environment** using `uv` (Python 3.9+ recommended, tested on 3.12.10):

   ```bash
   uv venv -p 3.12.10

   # On Windows:
   .venv\Scripts\activate

   # On macOS/Linux:
   source .venv/bin/activate
   ```

2. **Install project dependencies**:

   ```bash
   uv pip install -r pyproject.toml
   ```

3. **Set up environment variables**:

   Create a `.env` file with the following:

   ```env
   OPENAI_API_KEY="sk-pro-..."
   GOOGLE_CLIENT_ID="your-google-client-id"
   GOOGLE_CLIENT_SECRET="your-google-client-secret"
   ```

### Google Cloud Setup

1. **Create or select a Google Cloud project**
   - Open the Google Cloud Console and pick an existing project or click New Project.

2. **Enable the Gmail API**
   - In the left sidebar go to APIs & Services → Library, search for Gmail API, and click Enable.

3. **Configure the OAuth consent screen**
   - Navigate to APIs & Services → OAuth consent screen.
   - Under App Information, set:
     - App name: e.g., "Gmail‑Chroma RAG"
     - User support email: your email
   - Click Next.
   - Under User Type, choose External (for consumer Gmail) or Internal (for Workspace).
   - Click Next, add a Developer contact email, then Save and Continue.

4. **Add the Gmail scope**
   - On the consent screen page, open Data Access → Scopes.
   - Click Add or Remove Scopes, search for `https://www.googleapis.com/auth/gmail.readonly`, check it, and Save.

5. **Add test users**
   - In the Cloud Console, go to APIs & Services → OAuth consent screen.
   - In the left sidebar click Audience.
   - On the Audience page, scroll down past the "User Type" toggle. You'll see a Test users panel.
   - Click Add users, enter your Gmail address, then Save.

6. **Create OAuth client credentials**
   - Go to APIs & Services → Credentials.
   - Click Create Credentials → OAuth client ID.
   - Select Desktop app, name it (e.g., "Gmail→Chroma RAG"), and click Create.
   - Copy the displayed Client ID and Client secret.

### Running the App

To launch the main application:

```bash
streamlit run app.py
```

---
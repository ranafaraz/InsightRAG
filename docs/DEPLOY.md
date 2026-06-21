# Deployment notes

## Streamlit Community Cloud (free)
1. Push this repo to GitHub.
2. On https://share.streamlit.io, create an app pointing at `ui/streamlit_app.py`.
3. In **Advanced settings → Python deps**, it installs from `requirements.txt`.
   For the lightest free deployment, set these secrets/env so it stays offline:
   `EMBEDDING_BACKEND=hash`, `RERANK_BACKEND=lexical`, `LLM_BACKEND=stub`, `VECTOR_STORE=memory`.
   For real answers, set `LLM_BACKEND=openai` + `OPENAI_API_KEY` in **Secrets**.

## Hugging Face Spaces (free)
1. Create a new **Streamlit** Space.
2. Add this repo's files (or link the GitHub repo).
3. Add a `requirements.txt` (already present) and set the same env backends in **Settings → Variables/Secrets**.

## Recording the demo GIF
- Run `streamlit run ui/streamlit_app.py`, click **Load sample documents**, ask
  "What does InsightRAG use to rerank candidates?", then an off-topic question to show the refusal,
  then an injection prompt to show the block.
- Capture ~20–30s with [ScreenToGif](https://www.screentogif.com/) (Windows) and save to `docs/demo.gif`,
  then re-enable the image embed at the top of the README.

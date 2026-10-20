# Scopus Research Paper Recommender

A Streamlit app for exploring Scopus publication records and finding similar papers using TF-IDF text vectors and cosine similarity. The project also includes notebooks for extracting records and mapping Scopus subject codes.

## Project files

- `app.py` — Streamlit dashboard with recommendations, publication insights, and a data explorer.
- `info.csv` — publication dataset used by the dashboard.
- `Data Extracting Process.ipynb` — notebook for converting downloaded Scopus JSON records into a CSV dataset.
- `Web_scrapping_subjectcode & Additional Scopus API/API.ipynb` — notebook for querying the Scopus API.
- `Web_scrapping_subjectcode & Additional Scopus API/CodeSubject.ipynb` — notebook for collecting subject-code mappings.
- `Web_scrapping_subjectcode & Additional Scopus API/code_subject.csv` — subject-code data.

## Run the dashboard

Use Python 3. Install the dependencies and start Streamlit from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app reads `info.csv` from the project directory. On first run, it builds and saves the TF-IDF similarity model; those generated `.joblib` files are ignored by Git.

## Scopus API notebook

Set your Scopus API key in the environment before running cells that call the API. The notebook reads it from `SCOPUS_API_KEY`; do not paste API keys into the notebook or commit them.

```bash
export SCOPUS_API_KEY="your-api-key"
```


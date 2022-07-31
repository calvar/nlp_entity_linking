# nlp-entity-linking-lp-author

Repository for liveProject: NLP Entity Linking for Medical Transcripts


## Run from a command line

First, set up the Python environment:
```bash
python3 -m venv venv
source venv/bin/activate

python3 -m pip install -U pip
python3 -m pip install -e .
```

Then launch JupyterLab:
```bash
jupyter-lab
```

Which should open automagically in your browser.
Then launch a new notebook.


## Run from a container:

Alternatively, build a Docker container:
```bash
docker build --pull --rm -f "Dockerfile" -t debater:latest .
```

Then run it
```bash
docker run --rm -p 8888:8888 -it debater
```

Now open `http://localhost:8888` in your browser and launch a new
notebook.

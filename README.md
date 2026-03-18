# Steps to run

```
### PART 0 ###

# 0. Download the dataset from the link below - make sure in includes the audio etc.
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

# 1. Unzip the dataset and place the resulting Data folder in the base project directory

### PART 1 ###
# 0. Open a terminal and navigate to project folder

# 1. Create the virtual environment
python3 -m venv .venv

# 2. Activate it (Mac/Linux)
source .venv/bin/activate

# 3. Install the required packages
pip install -r requirements.txt

# 4. Complile the pipeline by running the following line
python pipeline_supervised.py --csv Data/features_30_sec.csv --method umap-unsupervised

# 5. Expose the backend API locally
uvicorn api_supervised:app --reload --port 8000


### PART 2 ###
# 0. Open a new terminal and navigate to the genremap2 folder within the project root

# 1. Install the needed packages
npm install

# 2. Activate the local server for frontend
npm run dev

# 3. Open the app in a browser
http://localhost:5173/
```

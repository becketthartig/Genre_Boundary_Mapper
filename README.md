# Steps to run

```
### PART 1 ###
# 0. Open a terminal and navigate to project folder

# 1. Create the virtual environment
python3 -m venv .venv

# 2. Activate it (Mac/Linux)
source .venv/bin/activate

# 3. Install the required packages
pip install -r requirements.txt

# 4. Expose the backend API locally
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

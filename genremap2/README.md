# GenreMap — Frontend

Dark studio-style UI for supervised genre visualization with audio playback.

## Local setup

```bash
cd genremap
npm install
npm run dev
```

Opens at http://localhost:5173

## Backend setup

The frontend requires the FastAPI backend running on port 8000.

```bash
# Generate results.json and pipeline.pkl from your CSV
python pipeline_supervised.py --csv Data/features_30_sec.csv

# Start the API (serves both data and audio files)
uvicorn api_supervised:app --reload --port 8000
```

The Vite dev server proxies all `/api/*` requests to `http://localhost:8000`,
so you never deal with CORS during development.

## Features

- **Audio playback on click**: Click any point to select it and hear the 30-second audio clip
  - Audio loops automatically while selected
  - Click another point to switch songs
  - Click the background to deselect and stop playback
  - "🔊 playing" indicator in the detail panel
  
- **Genre-based coloring**: Points colored by true genre labels from the CSV
  
- **PCA visualization**: 2D projection preserving as much variance as possible

- **Feature selection**: Use `--features` flag when running `pipeline_supervised.py` to choose which features to include in PCA

## Project structure

```
src/
  App.jsx              # Root — layout, data fetching, state
  App.module.css       # Main layout styles + global tooltip styles
  ScatterPlot.jsx      # D3 scatter plot with zoom/pan/hover/audio
  ClusterLegend.jsx    # Left sidebar genre list with hover highlight
  SongDetail.jsx       # Right panel song inspector
  UploadZone.jsx       # Drag-and-drop .wav uploader
  api.js               # Fetch wrapper for FastAPI endpoints
  index.css            # CSS variables, reset, global styles
```

## Audio file structure

The API expects audio files to be organized as:
```
Data/genres_original/{genre}/{filename}.wav
```

For example: `Data/genres_original/blues/blues.00008.wav`

The genre is extracted from the filename prefix, so this structure must match
what's in your CSV's filename column.

## Adding new visualizations

The viz selector in the header has placeholder slots. To add a new view:

1. Add an entry to `VISUALIZATIONS` in `App.jsx` with `available: true`
2. Add a new `activeViz === 'your-id'` branch in the canvas section
3. Create your new component (receives `songs` and `features` as props)

The `features` object from `/api/features` contains:
```json
{
  "all": [...],      // all 57 features from the CSV
  "active": [...]    // features currently used in PCA
}
```

Use this to populate the feature selector dropdown for cross-section views.

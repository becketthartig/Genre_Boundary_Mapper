import os
import tensorflow_datasets as tfds

# where to save
OUT_DIR = "Data/audio"
os.makedirs(OUT_DIR, exist_ok=True)

# load GTZAN
ds = tfds.load("gtzan", split="train", data_dir=None, shuffle_files=False)

for example in ds:
    audio = example["audio"].numpy()
    label = example["label"].numpy()
    fname = example["file_name"].numpy().decode("utf-8")

    # the file_name is like: blues/blues.00000.wav
    genre, filename = fname.split("/")
    save_path = os.path.join(OUT_DIR, filename)

    with open(save_path, "wb") as f:
        f.write(audio)

    print(f"saved: {save_path}")
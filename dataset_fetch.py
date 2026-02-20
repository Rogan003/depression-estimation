from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
from pathlib import Path

from common import merge_dataset_csv

api = KaggleApi()
api.authenticate()

all_files = []
page_token = None

while True:
    response = api.dataset_list_files(
        "saifzaman123445/daicwoz", #for example: "daehoyang/flickr2k"
        page_size=200,
        page_token=page_token
    )

    all_files.extend(response.files)

    # Check for next page
    if hasattr(response, 'nextPageToken') and response.nextPageToken:
        page_token = response.nextPageToken
    else:
        break

print(f"Total files retrieved: {len(all_files)}")

file_names = sorted([f.name for f in all_files if f.name.endswith('_AUDIO.wav') or "avec" in f.name.lower()])
print(file_names)
print(len(file_names))

output_dir=r"dataset"

for fname in file_names:
    print(f'downloading: {fname}')
    api.dataset_download_file(
        dataset="saifzaman123445/daicwoz",
        file_name=fname,
        path=output_dir,
        quiet=False
    )
    # The downloaded file will be named {original_name}.zip
    zip_path = Path(output_dir)/(Path(fname).name + ".zip")
    print(zip_path)

    if zip_path.exists():
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(output_dir)
        zip_path.unlink()  # Delete the zip file after extraction

merge_dataset_csv()
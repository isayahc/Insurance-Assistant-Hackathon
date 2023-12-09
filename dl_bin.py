import requests
from typing import Optional

def download_binary_file(url: str, file_path: Optional[str] = None) -> None:
    """
    Download a binary file from a given URL and save it to the specified path.

    :param url: URL of the binary file to be downloaded.
    :param file_path: Local path to save the file. If None, the file will be saved with its original name.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # If no specific file path is provided, extract the file name from the URL
        if file_path is None:
            file_path = url.split('/')[-1]

        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"File downloaded successfully: {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")

# Example usage
url = "https://llamahack.slack.com/files/U069A8NRB9T/F068ZTLK9KR/anthem_hsa_medical_insurance_benefit_booklet.pdf"
# download_binary_file(url)

import urllib.request
urllib.request.urlretrieve(url, "filename.pdf")

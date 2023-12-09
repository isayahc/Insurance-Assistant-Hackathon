import requests
import os
from typing import Optional
from urllib.parse import urlparse, unquote

def get_filename_from_url(url: str, cd: Optional[str]) -> str:
    """
    Extracts and returns the filename from the URL or content-disposition header.
    """
    if cd:
        fname = [x.strip() for x in cd.split(';') if x.strip().startswith('filename=')]
        if fname:
            return unquote(fname[0].split('=')[1].strip('"'))

    # Fallback to extracting filename from URL
    parsed_url = urlparse(url)
    return os.path.basename(parsed_url.path)

def download_file(url: str, save_dir: Optional[str] = None, save_name: Optional[str] = None) -> None:
    """
    Downloads a file from the given URL and saves it in the specified directory.
    If the directory does not exist, it will be created.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        filename = save_name if save_name else get_filename_from_url(url, response.headers.get('content-disposition'))

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, filename)
        else:
            file_path = filename

        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192): 
                if chunk:
                    file.write(chunk)

        print(f"File downloaded and saved as: {file_path}")

    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error: {err}")
    except Exception as e:
        print(f"Error: {e}")



if __name__ == "__main__":
    # Example Usage
    url = "https://llamahack.slack.com/files/U069A8NRB9T/F068ZTLK9KR/anthem_hsa_medical_insurance_benefit_booklet.pdf"
    download_file(url,"data")

import requests
from typing import Optional
from urllib.parse import unquote

def get_filename_from_cd(cd: Optional[str]) -> Optional[str]:
    """
    Get filename from content-disposition header.
    """
    if not cd:
        return None
    fname = [x.strip() for x in cd.split(';') if x.strip().startswith('filename=')]
    return unquote(fname[0].split('=')[1]) if fname else None

def download_file(url: str) -> None:
    """
    Download a file from a URL and save it with the server-provided name.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code

    filename = get_filename_from_cd(response.headers.get('content-disposition'))

    if filename:
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded and saved as: {filename}")
    else:
        print("Filename could not be determined from the server.")

if __name__ == "__main__":
    # Example Usage
    url = "YOUR_FILE_URL_HERE"
    download_file(url)

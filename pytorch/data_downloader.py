from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

#1TyUs6bmJYBR3mMH1dseTPsURu1fTHEU9
gauth = GoogleAuth()
# gauth.CommandLineAuth() # client_secrets.json need to be in the same directory as the script
gauth.LoadCredentialsFile(credentials_file="access.json")
drive = GoogleDrive(gauth)


def download_folder(folder_id, parent_path="."):
  file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
  for listed in file_list:
    if "fileExtension" not in listed:
      download_folder(listed['id'], parent_path + "/" + listed['title'])
      continue

    file_reference = drive.CreateFile({'id': listed['id']})
    print(f"Downloading {listed['title']}...")
    if not parent_path == "." and not os.path.exists(parent_path):
      os.makedirs(parent_path)

    file_reference.GetContentFile(parent_path + "/" + listed['title'])


if __name__ == '__main__':
  download_folder("1ZcYb4eH2jPndS5nkqQFcLHdGNM9dTF5C", parent_path="TabletGazeDataset")

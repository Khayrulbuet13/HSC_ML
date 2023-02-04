from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth

# For using listdir()
import os


# Below code does the authentication
# part of the code
gauth = GoogleAuth()

# Creates local webserver and auto
# handles authentication.
gauth.LocalWebserverAuth()	
drive = GoogleDrive(gauth)

# replace the value of this variable
# with the absolute path of the directory
path = r"/home/mdi220/simulations/Git_repository/HSC/HSC_khayrul/Output/TBCells_crops/BCells_Old_crops"

# iterating thought all the files/folder
# of the desired directory
for x in os.listdir(path):

	f = drive.CreateFile({'parents': [{'id': '1vdYlGV6Of03AO-v2GPVQ5KOjrBNJL6dY'}], 'title': x})
	print(x)
	f.SetContentFile(os.path.join(path, x))
	f.Upload()

	# Due to a known bug in pydrive if we
	# don't empty the variable used to
	# upload the files to Google Drive the
	# file stays open in memory and causes a
	# memory leak, therefore preventing its
	# deletion
	f = None


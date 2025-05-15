🌟 Overview: Purpose of the Code
This script creates a web-based interface using Streamlit to allow users to upload scanned OMR (Optical Mark Recognition) sheets and process them through the backend pipeline defined in main.py. It runs the image processing pipeline when the user clicks the "RUN" button.

There are also command-line instructions mentioned in the comment at the top:

python
Copy
Edit
"""
to run:
    streamlit run app.py

to create your own custom template:
    python main.py --setLayout
"""
streamlit run app.py — launches the web interface.

python main.py --setLayout — initiates a template layout configuration via command-line.

🔧 Imports and Setup
python
Copy
Edit
import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
from main import parse_args, entry_point_for_args
import uuid
import shutil
streamlit: Used for creating web apps with Python.

numpy and cv2 (OpenCV): Used for image processing.

os and shutil: Used for file and directory operations.

PIL.Image: For reading image uploads.

uuid: Generates unique folder names to isolate each run.

parse_args, entry_point_for_args: Imported from main.py, which contains the backend logic.

python
Copy
Edit
curr = os.getcwd()
args = parse_args()
test_dir = os.path.join(curr, "inputs", "new_omr")
curr stores the current working directory.

args stores parsed command-line-like arguments (from main.py).

test_dir is a base directory where uploaded images will be temporarily stored.

🧠 Function: run() — Web App Logic
This is the main function that defines and runs the Streamlit app.

1. App Title
python
Copy
Edit
st.title('OMR Reader')
Displays the web app title at the top.

2. File Upload
python
Copy
Edit
uploaded_files = st.sidebar.file_uploader("Choose image", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
Lets users upload one or more image files.

The file uploader appears in the sidebar.

3. Form to Trigger Processing
python
Copy
Edit
with st.form(key="myform", clear_on_submit=True):
    submitted = st.form_submit_button("RUN")
Displays a form with a "RUN" button.

When the user clicks "RUN", submitted becomes True, and processing starts.

4. Processing Uploaded Files
When the "RUN" button is clicked:

a. Create Unique Directory for the Run
python
Copy
Edit
unique_name = str(uuid.uuid4())
new_test_dir = os.path.join(test_dir, unique_name)
os.makedirs(new_test_dir)
Creates a unique folder inside inputs/new_omr/ to store the uploaded files for this session.

b. Save Uploaded Images to Disk
python
Copy
Edit
for up in uploaded_files:
    im = Image.open(up)
    im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(new_test_dir, up.name), im)
Each uploaded image is:

Opened using PIL.

Converted to OpenCV’s BGR format.

Saved to the newly created unique folder.

c. Set Input Path for Processing
python
Copy
Edit
args["input_paths"][0] = "inputs/new_omr/"+unique_name
Updates the args to point to the directory of uploaded images.

d. Copy Reference Files (if any)
python
Copy
Edit
for i in os.listdir(test_dir):
    if os.path.isfile(os.path.join(test_dir, i)):
        shutil.copy(os.path.join(test_dir, i), os.path.join(new_test_dir, i))
Copies any existing reference files in inputs/new_omr to the new folder. This may include layout templates or other required metadata.

e. Run the Backend Processor
python
Copy
Edit
entry_point_for_args(args)
Calls the main backend processor using the updated arguments.

f. Clean Up Temporary Files
python
Copy
Edit
shutil.rmtree(new_test_dir)
Deletes the temporary folder and its contents after processing to keep the system clean.

🚀 Final Execution Block
python
Copy
Edit
if __name__ == "__main__":
    run()
Ensures that the run() function is called only if this file (app.py) is executed directly, not imported.

✅ Summary
This script is a Streamlit-based web frontend that:

Lets users upload OMR sheet images.

Converts them to OpenCV format and saves them.

Runs the backend pipeline using the same logic from main.py.

Cleans up after each run to avoid file clutter.

It’s a user-friendly way to interact with an OMR processing system without needing to use the command line.

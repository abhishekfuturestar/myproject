"""
to run:
    streamlit run app.py

to create your own custom template:
    python main.py --setLayout

"""

import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
from main import parse_args, entry_point_for_args
import uuid
import shutil

curr = os.getcwd()
args = parse_args()
test_dir = os.path.join(curr, "inputs", "new_omr")


def run():

    st.title('OMR Reader')

    uploaded_files = st.sidebar.file_uploader("Choose image", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:

        with st.form(key="myform", clear_on_submit=True):

            submitted = st.form_submit_button("RUN")

            if submitted:
                unique_name = str(uuid.uuid4())
                new_test_dir = os.path.join(test_dir, unique_name)
                if not os.path.exists(new_test_dir):
                    os.makedirs(new_test_dir)

                for up in uploaded_files:
                    im = Image.open(up)
                    im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(new_test_dir, up.name), im)

                args["input_paths"][0] = "inputs/new_omr/"+unique_name
                for i in os.listdir(test_dir):
                    if os.path.isfile(os.path.join(test_dir, i)):
                        old_path = os.path.join(test_dir, i)
                        new_path = os.path.join(new_test_dir, i)
                        shutil.copy(old_path, new_path)
                entry_point_for_args(args)
                shutil.rmtree(new_test_dir)

if __name__ == "__main__":
    run()

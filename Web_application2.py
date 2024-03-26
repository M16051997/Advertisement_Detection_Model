import os
import streamlit as st
import cv2
from PIL import Image
from ultralytics import YOLO
import base64
import fitz
#os.chdir(r'D:\WebApplication_YOLO_AD_detectsystem\The Trail Image Folder')
# Define a function to apply custom CSS
pathimage = "https://raw.githubusercontent.com/M16051997/Advertisement_Detection_Model/main/rm314-adj-10.jpg" 

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background(pathimage)

st.title(':orange[Advertisement Detection Web App]')

# Define custom CSS to style the text area and text color
# Define custom CSS to style the text area with a black background and white text color
custom_css = """
<style>
    /* Add a border to the text area */
    .custom-text-area {
        border: 1px solid #000; /* You can adjust the border properties as needed */
        border-radius: 5px;
        padding: 10px;
        background-color: black; /* Black background color */
        color: white; /* White text color */
    }
</style>
"""

# Apply the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Instructions:
multi = """ Instructions:--

1. The Model Trained with English & Tamil NewsPapers.

2. Use any type of News paper wether PDF or Image file, the Model will automaticall Detect adds.

3. The model will take at a time whole newspaper but it is recommended to upload single page or image. It is very useful for us to count and verify the published ads.

4. The Model accuracy is around 80%. 
"""
st.markdown(multi)

# Function to convert PDF to images
def pdf_to_img(uploaded_file, img_path_prefix):
    # Save the uploaded file
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getvalue())

    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    if file_extension == ".pdf":
        pdf = fitz.open(uploaded_file.name)
        image_paths = []
        for page_number in range(pdf.page_count):
            page = pdf[page_number]
            # Convert the page to a pixmap
            pixmap = page.get_pixmap()

            # Convert the Pixmap to a Pillow Image
            img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)

            # Save the image as JPEG
            image_path = f"{img_path_prefix}_page_{page_number + 1}.jpeg"
            img.save(image_path)
            image_paths.append(image_path)

        pdf.close()
        return image_paths
    elif file_extension == ".jpeg" or file_extension == ".jpg":
        # If the uploaded file is already an image, return its path
        image_path = f"{img_path_prefix}_uploaded_image.jpeg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        return [image_path]
    else:
        st.error("Unsupported file format. Please upload a PDF or JPEG image.")
        return []

# Function to perform object detection
def perform_object_detection(image_path):
    # Load the YOLO model
    #model = YOLO(r"D:\ADS_Project_Deployment\Models\Detection Models\best31_1000_epochs.pt")
    model_url = "https://github.com/M16051997/Advertisement_Detection_Model/raw/main/best31_1000_epochs.pt"
    model = YOLO.from_pretrained(model_url)
    # Load and preprocess the image
    img = cv2.imread(image_path)

    results = model(img)

    detections = []  # Store tuples of bounding box and confidence
    # Access the detected objects and their properties
    if isinstance(results, list):
        for res in results:
            if res.boxes is not None:
                for det, confidence in zip(res.boxes.xyxy, res.boxes.conf):
                    x1, y1, x2, y2 = map(int, det[:4])
                    confidence_value = round(confidence.item(), 2)
                    detections.append(((x1, y1, x2, y2), confidence_value))
            else:
                print("No detections found in the current element.")
    else:
        print("No results found.")

    return detections


# Main function
def main():
    # st.title("Advertisement Detection Web App")

    # File upload
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "jpeg", "jpg"])

    if uploaded_file is not None:
        # Convert PDF to images or use the uploaded image directly
        image_paths = pdf_to_img(uploaded_file, "uploaded_image")

        if image_paths:
            # Perform object detection for each image
            for idx, image_path in enumerate(image_paths):
                st.image(image_path, caption=f"Page {idx + 1}", use_column_width=True)
                st.write(f"### Detected Advertisements - Page {idx + 1}")
                detections = perform_object_detection(image_path)

                # Iterate through the detections and extract the detected images
                for i, (detection, confidence) in enumerate(detections):
                    x1, y1, x2, y2 = detection
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert the coordinates to integers
                    # Crop the image using the bounding box coordinates
                    detected_image = cv2.imread(image_path)[y1:y2, x1:x2]
                    # Display the detected image
                    st.image(detected_image, caption=f"Detected Image {i + 1}", use_column_width=True)
                    st.write(f"Confidence: {confidence}")
                    
if __name__ == "__main__":
    main()

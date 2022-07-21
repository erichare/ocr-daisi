import matplotlib.pyplot as plt
import keras_ocr
import tensorflow as tf
import streamlit as st
import numpy as np
import io
import tempfile

from PIL import Image

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

def annotate_image(image: np.ndarray=None):
    '''
    Run the Keras OCR Algorithm on the given input image
    
    This function takes an image as either a PIL image, or a Numpy array,
    or a path to an image, and returns a new PIL image which includes
    bounding boxes and extracted text that was found in the image

    :param image np.ndarray: The image to analyze
    
    :return: Results of the OCR Algorithm
    '''

    def fig2img(fig):
        """Convert a Matplotlib figure to a PIL Image and return it"""
        import io
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img

    # If none is provided, we read the Daisi logo
    if image is None:
        image = Image.open("daisi.jpeg")
        image.load()

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, 'wb') as f:
        f.write(img_byte_arr) # where `stuff` is, y'know... stuff to write (a string)
    image = tmp.name

    # Get a set of three example images
    image = keras_ocr.tools.read(image)

    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples.
    prediction_groups = pipeline.recognize([image])
    fig, ax = plt.subplots()

    # Plot the predictions
    keras_ocr.tools.drawAnnotations(image=image, predictions=prediction_groups[0], ax=ax)

    return fig2img(fig)

if __name__ == "__main__":
    st.title("Optical Character Recognition with Daisies")
    st.write("This Daisi allows you to provide an image, and uses a Keras-based OCR model to perform OCR on the image.")
    
    with st.sidebar:
        uploaded_file = st.file_uploader("Choose an Image", type=["png","jpg","jpeg"])

    if not uploaded_file:
        file_name = "daisi.jpeg"
    else:
        file_name = uploaded_file.name

    with st.expander("Inference with PyDaisi", expanded=True):
        st.markdown(f"""
        ```python
        import pydaisi as pyd
        from PIL import Image

        ocr = pyd.Daisi("erichare/Optical Character Recognition")

        img = Image.open("{file_name}")
        img.load()

        result = ocr.annotate_image(img).value
        result.show()
        ```
        """)

    with st.spinner("Analyzing your image, please wait..."):
        final_result = annotate_image(uploaded_file)
        st.image(final_result, caption='Text Extracted from the Image')

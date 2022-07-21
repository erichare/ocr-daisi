import matplotlib.pyplot as plt
import keras_ocr
import tensorflow as tf

from PIL import Image

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

def annotate_image(image_url="https://upload.wikimedia.org/wikipedia/commons/b/bd/Army_Reserves_Recruitment_Banner_MOD_45156284.jpg"):

    def fig2img(fig):
        """Convert a Matplotlib figure to a PIL Image and return it"""
        import io
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img

    # Get a set of three example images
    image = keras_ocr.tools.read(image_url)

    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples.
    prediction_groups = pipeline.recognize([image])
    fig, ax = plt.subplots()

    # Plot the predictions
    keras_ocr.tools.drawAnnotations(image=image, predictions=prediction_groups[0], ax=ax)

    return fig2img(fig)

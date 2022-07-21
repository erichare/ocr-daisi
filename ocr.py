import matplotlib.pyplot as plt
import keras_ocr

from PIL import Image

def annotate_image():

    def fig2img(fig):
        """Convert a Matplotlib figure to a PIL Image and return it"""
        import io
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img

    # keras-ocr will automatically download pretrained
    # weights for the detector and recognizer.
    pipeline = keras_ocr.pipeline.Pipeline()

    # Get a set of three example images
    image = keras_ocr.tools.read('https://upload.wikimedia.org/wikipedia/commons/b/bd/Army_Reserves_Recruitment_Banner_MOD_45156284.jpg')

    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples.
    prediction_groups = pipeline.recognize([image])
    fig = plt.Figure()

    # Plot the predictions
    keras_ocr.tools.drawAnnotations(image=image, predictions=prediction_groups[0])
    
    return fig2img(fig)

# Optical Character Recognition (OCR) with Daisies

## How to Call

First, we simply load the PyDaisi package:

```python
import pydaisi as pyd
```

Next, we connect to the Daisi:

```python
optical_character_recognition = pyd.Daisi("erichare/Optical Character Recognition")
```

Next, we call the `annotate_image` function - Here, we can pass a PIL image object, a Numpy image, or, as in this example, if we pass `None`, it uses a default image (the Daisi logo):

```python
optical_character_recognition.annotate_image(image=None).value
```

## Running the Streamlit App

Or, we can automate everything by just [Running the Streamlit App](https://dev3.daisi.io/daisies/b661456f-ba50-457c-8092-5b6814b2c37f/app)

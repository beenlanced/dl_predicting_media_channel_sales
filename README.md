# Predicting Media Channel Sales Using Deep Learning with Tensorflow and Keras

<p>
  <img alt="Telecom Media Channels" src="imgs/media_channels.jpeg"/>
</p>

[img source: Pinterest](https://www.pinterest.com/pin/mess-media--731131320746574258/)

## Project Description

This project came out of the [LinkedIn Learning course: Deep Learning and Generative AI with Python: Data Processing and Analytics](https://www.linkedin.com/learning/deep-learning-and-generative-ai-data-prep-analysis-and-visualization-with-python)

The goal of the project is take raw marketing telecommunication data from a fictitious company and build a predictive model for media sales. Essentially, this is a typical machine learning project which gets an extra zhuzh by building the model using a deep learning/neural network model and making a web accessible application to provide access to the predicative model.

### What this Project Does Specifically

The project

- Loads and inspects the marketing telecommunication data
- Preprocesses/cleans the data
- Performs Exploratory Data Analysis
- Creates a deep learning network to build the prediction model
- Conducts analysis of the predictive model results
- Makes any improvements
- Makes the predicative model accessible via a web application

---

## Objective

The project contains the key elements:

- `Deep Learning` for neural networks building,
- `FastAPI` to render the app,
- `Git` (version control),
- `Httpx` to help make async HTTP requests
- `Jupyter` python coded notebooks,
- `Keras` to build autoencoder and layers,
- `Matplotlib` visualization of spaces,
- `Numpy` for arrays and numerical operations,
- `Python` the standard modules,
- `Pydantic` to define structure of incoming request body and validate input data,
- `Scikit-Learn` for PCA and TSNE modules and to get training and test datasets,
- `TensorFlow` to build autoencoder and layers

---

## Tech Stack

![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white)
![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)
![Matplotlib](https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?logo=matplotlib&logoColor=fff)
![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)

---

## Getting Started

Here are some instructions to help you set up this project locally.

---

## Installation Steps

The Python version used for this project is `Python 3.11` to be compatible with TensorFlow.

Follow the requirements for using TensorFlow [here](https://www.tensorflow.org/install/pip#macos)

use `uv pip install tensorflow`

- Make sure to use python versions `Python 3.9–3.12
- pip version 19.0 or higher for Linux (requires manylinux2014 support) and Windows. pip version 20.3 or higher for macOS.
- Windows Native Requires Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019

### Clone the Repo

1. Clone the repo (or download it as a zip file):

   ```bash
   git clone https://github.com/beenlanced/ai_learning_visualizing_latent_spaces.git
   ```

2. Create a virtual environment named `.venv` using `uv` Python version 3.11:

   ```bash
   uv venv --python=3.11
   ```

3. Activate the virtual environment: `.venv`

   On macOs and Linux:

   ```bash
   source .venv/bin/activate #mac
   ```

   On Windows:

   ```bash
    # In cmd.exe
    venv\Scripts\activate.bat
   ```

4. Install packages using `pyproject.toml` or (see special notes section)

   ```bash
   uv pip install -r pyproject.toml
   ```

   or alternatively with the `requirements.txt` file

   ```bash
   uv pip install -r requirements.txt
   ```

### Install the Jupyter Notebook `visualizing_latent_space.ipynb`

1. **Run the Project**

   - Run `visualizing_latent_space.ipynb` in Jupyter Notebook UI or in VS code.

   run the FastAPI application from your terminal:

   ```bash
   uvicorn main:app --reload
   ```

   then open a web browser and got to

   ```
   http://127.0.0.1:8000
   ```

---

## Special Notes

- Including`requirements.txt` file as there are unique considerations when using the `TensorFlow` library. Plus, just in case you run into issues wth the `pyproject.toml` file there is the ability to use the requirements file as well to build the virtual environment.

- Running this notebook took approximately 25 minutes to run using an Apple M4 Pro with macOS Sequoia. Depending on the processor(s) that you are running this time will vary.

---

### Final Words

Thanks for visiting.

Give the project a star (⭐) if you liked it or if it was helpful to you!

You've `beenlanced`! 😉

---

## Acknowledgements

I would like to extend my gratitude to all the individuals and organizations who helped in the development and success of this project. Your support, whether through contributions, inspiration, or encouragement, have been invaluable. Thank you.

Specifically, I would like to acknowledge:

As this work was a project for the [LinkedIn Learning course: Deep Learning and Generative AI with Python: Data Processing and Analytics](https://www.linkedin.com/learning/deep-learning-and-generative-ai-data-prep-analysis-and-visualization-with-python), a shout out to the folks at LinkedIn Learning.

- [Hema Kalyan Murapaka](https://www.linkedin.com/in/hemakalyan) and [Benito Martin](https://martindatasol.com/blog) for sharing their README.md templates upon which I have derived my README.md.

- The folks at Astral for their UV [documentation](https://docs.astral.sh/uv/)

---

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details

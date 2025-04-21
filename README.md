# traffic_control_mca_projectwork

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)





## Installation
To set up the project locally, follow these steps:

1. **Clone the repository:**

2. **Create a virtual environment:**

       python -m venv venv
       source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install dependencies:**

        pip install -r requirements.txt


## Usage

 ### Data Preparation

    python src/data_preprocessing.py
       
 ### Model Training

     python src/model_training.py

 ### Streamlit App

     streamlit run src/streamlit_app.py
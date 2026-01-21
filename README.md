# Real-Time Real Estate Price Prediction System

> Work in progress

Production-style end-to-end machine learning system for real estate price prediction built with **Python, scikit-learn, FastAPI, Docker, and Kafka**.  
The project demonstrates how to design, build and deploy a full ML pipeline including data preprocessing, model training, model serving, and real-time inference using a streaming architecture.

---

## Project Overview

The goal of this project is to build a **production-like machine learning system** that predicts real estate prices based on property characteristics such as:

- location (district, city, town),
- property type,
- size and area features,
- number of rooms,
- construction year,
- energy certificate,
- and available amenities (parking, garage, elevator, etc.).

The system is designed to work in a **real-time inference scenario** using Kafka as a simulated streaming data source and FastAPI as a model serving layer.

---

##  Machine Learning Pipeline

The ML pipeline includes:

- Data cleaning and preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Model training and evaluation  
- Model selection and serialization  
- Inference pipeline used by the API  

Models are trained using `scikit-learn` and evaluated using standard regression metrics.

---

##  System Architecture


- **Kafka Producer** simulates incoming real estate listings  
- **Kafka Consumer** reads messages and sends them to the ML API  
- **FastAPI service**:
  - validates input data using Pydantic
  - performs preprocessing
  - runs the trained ML model
  - returns the predicted price  

All components are orchestrated using **Docker Compose**.

+----------------+       +--------+       +-----------+       +--------------------+
| Kafka Producer | ----> | Kafka  | ----> | Consumer  | ----> | FastAPI ML Service |
+----------------+       +--------+       +-----------+       +--------------------+
                                                                  |
                                                                  v
                                                           +--------------+
                                                           | ML Model     |
                                                           +--------------+
                                                                  |
                                                                  v
                                                           +--------------+
                                                           | Prediction   |
                                                           +--------------+


---

##  Tech Stack

- Python
- Pandas, NumPy
- scikit-learn
- FastAPI
- Kafka
- Docker & Docker Compose
- (Optional) MLflow

---

##  Repository Structure

real-estate-ml-system/
├── data/ # Datasets (not committed)
├── notebooks/ # EDA and experiments
├── training/ # Training pipeline
├── model/ # Saved models (not committed)
├── api/ # FastAPI service
├── kafka/ # Producer and consumer
├── docs/ # Project documentation
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md


---

## How to Run (later)

```bash
docker-compose up --build
```
to test the API.

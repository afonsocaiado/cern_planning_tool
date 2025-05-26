# Task Prediction and Planning Tool for Complex Engineering Tasks

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Docker](https://img.shields.io/badge/docker-supported-brightgreen.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

A machine learning-powered planning tool developed in collaboration with CERN (European Organization for Nuclear Research) and FEUP (Faculty of Engineering of the University of Porto). This tool predicts and suggests activities and contributions for complex engineering tasks, helping to streamline planning processes in large-scale scientific projects.

## 🧑‍💻 Project Team

- **Afonso Caiado de Sousa** - Student from FEUP (Main developer)
- **João Carlos Pascoal Faria** - Thesis supervisor (FEUP)
- **João Pedro Mendes Moreira** - Thesis co-supervisor (FEUP)
- **Fernando Pedrosa** - CERN Thesis supervisor
- **Rodrigo Lanza** - CERN Thesis co-supervisor

## 🔍 Project Description

This repository contains a machine learning solution for predicting and planning complex engineering tasks, specifically developed for CERN's maintenance and operational needs. The tool analyzes historical data to suggest similar activities, predict required contributions, and optimize planning processes.

### Key Features

- **Similar Activity Identification**: Find activities with similar characteristics
- **Activity Suggestion**: Get recommendations for new activities based on context
- **Contribution Planning**: Predict necessary contributions for activities
- **Combined Suggestions**: Get comprehensive planning recommendations

## 🗂️ Repository Structure

- **[api/](api/)**: REST API implementation with endpoints for predictions and suggestions
  - Contains Docker configuration for easy deployment
  - Includes comprehensive documentation in the wiki folder
- **[data/](data/)**: CSV datasets used for training and testing the models
- **[src/](src/)**: Source code for all machine learning models and utilities
  - Supervised learning models
  - Clustering algorithms
  - Feature engineering
  - Evaluation tools

## 🚀 Getting Started

### Prerequisites

- Python 3.10
- Docker and Docker Compose

### Running the Application

For quick deployment:

1. Clone this repository
2. Navigate to the `api` directory
3. Run `docker-compose up --build`
4. Access the API at `http://localhost:5000/`

For detailed installation and usage instructions, see the [API README](api/README.md).

## 📚 Documentation

Comprehensive documentation is available in the [API wiki](api/wiki/) folder, including:

- API structure and endpoints
- Data maintenance procedures
- Model updating guidelines
- Future development plans

## 🔗 Related Publications

- [Link to thesis document - Coming soon](#)
- [Link to related conference paper - Coming soon](#)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
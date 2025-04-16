# Unmasking Influence: Analyzing the Spread of Misinformation in Online Social Networks

## Project Overview

This project examines how COVID-19 vaccine misinformation spreads through Reddit communities. By analyzing network structures in both misinformation-prone and factual information subreddits, we map information flow between users and communities, identify influential spreaders, and compare structural differences between networks.

### Research Questions

1. **Network Structure:** Do misinformation networks differ structurally from factual information networks?
2. **Influential Spreaders:** Who are the key users facilitating information spread?
3. **Information Flow:** How does information propagate through these networks?
4. **Cross-Pollination:** To what extent do users participate in both misinformation and factual communities?

## Data Collection

We used the Python Reddit API Wrapper (PRAW) to collect posts and comments from:

### Misinformation/Skeptical Viewpoint Subreddits:
- r/NoNewNormal
- r/Conspiracy
- r/DebateVaccines
- r/LockdownSkepticism
- r/ChurchOfCOVID

### Factual Information/Mainstream Viewpoint Subreddits:
- r/Coronavirus
- r/COVID19
- r/science
- r/medicine
- r/Health
- r/askscience

Data was collected focusing on COVID-19 vaccine discourse during the vaccine rollout period (December 2020 - March 2021).

## Network Construction

We built directed interaction networks where:
- Nodes represent Reddit users
- Edges represent comments, replies, and other interactions
- Network analysis was performed using NetworkX

## Key Findings

Our analysis revealed significant differences between misinformation and factual information networks:

1. **Structure:** Misinformation networks show more centralized structures with prominent hub users, while factual networks display more distributed influence.

2. **Cross-posters:** We identified 188 users who actively participate in both types of communities, serving as information bridges.

3. **Community Patterns:** Community detection revealed distinct clustering patterns between the two network types.

## Repository Structure

- `/reddit_data/`: Contains collected Reddit data
- `/results/`: Network analysis outputs and visualizations
- `reddit_scraper.py`: Script for collecting Reddit data using PRAW
- `main_analysis.py`: Main analysis script
- `network_metrics.py`: Functions for calculating network metrics
- `cross_posting_analysis.py`: Functions for analyzing cross-posting behavior
- `visualization.py`: Functions for generating network visualizations

## Installation

```bash
# Clone the repository
git clone https://github.com/ccannon062/CSC445-Final-Project.git
cd CSC445-Final-Project

# Create and activate virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

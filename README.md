# NFL Eliminator Simulator
This repository contains the code for a data science project that aims to solve the strategic challenge of an NFL Eliminator (or Survivor) pool. The project uses historical game and betting data to train machine learning models that predict game outcomes.

The core of the project is a simulation engine that projects the entire NFL season week-by-week. It leverages a Beam Search algorithm to efficiently search through a massive number of potential paths, identifying the sequences of picks with the highest probability of success. The ultimate goal is to provide data-driven weekly pick recommendations that balance short-term wins with long-term strategic value.

This project is detailed in a two-part Medium series:

 - Part 1: Data Exploration and Predictive Modeling ([Link](https://medium.com/@anzekravanja/using-data-science-to-survive-my-nfl-eliminator-pool-part-1-dcd706d7b671))
 - Part 2: The Simulation Engine and Results ([Link](https://medium.com/@anzekravanja/using-data-science-to-survive-my-nfl-eliminator-pool-part-2-fe296cc10418))

## Key Features:
Two Predictive Models: A model for current-week predictions (with betting spreads) and another for future-week predictions (without spreads).
Dynamic Season Simulation Engine: Projects team records and stats forward in time based on model predictions.
Beam Search Pathfinding: Efficiently finds the most probable 18-week paths through the season.
Weekly Pick Recommendation Logic: Aggregates results from thousands of simulations to suggest the most strategically sound pick for the current week.
# Assignment: Probabilistic Inference in Graphical Models

This assignment involves implementing probabilistic inference in graphical models using a junction tree algorithm. The provided Python script (`template.py`) contains the implementation of various components required for this task.

## Table of Contents
1. [Overview of the Code](#overview-of-the-code)
2. [Input and Output Formats](#input-and-output-formats)
3. [How to Run](#how-to-run)
4. [Key Concepts](#key-concepts)
5. [Dependencies](#dependencies)
6. [Notes](#notes)

## Overview of the Code

The code is structured into several classes and functions to handle different aspects of the inference process:

### 1. **Key Functions**
- **`multiply_potentials`**: Multiplies two potentials based on their overlapping variables.
- **`sum_out`**: Marginalizes out specific variables from a potential.
- **`forward_pass` and `backward_pass`**: Perform message passing in the junction tree for inference.

### 2. **`Inference` Class**
This class encapsulates the main logic for performing inference:
- **Initialization**: Reads input data, initializes variables, and constructs the moral graph.
- **`create_moral_graph`**: Constructs the moral graph from the input cliques.
- **`triangulate_and_get_cliques`**: Triangulates the graph and identifies maximal cliques.
- **`get_junction_tree`**: Constructs the junction tree using a maximum spanning tree approach.
- **`assign_potentials_to_cliques`**: Assigns and merges potentials to the cliques in the junction tree.
- **`get_z_value`**: Computes the partition function (Z-value) and normalizes marginals.
- **`compute_marginals`**: Computes the marginal probabilities for each variable.
- **`compute_top_k`**: Identifies the top-k most probable assignments.

### 3. **`Get_Input_and_Check_Output` Class**
This class handles input and output operations:
- **`get_output`**: Processes multiple test cases, performs inference, and generates results.
- **`write_output`**: Writes the output to a JSON file.

### 4. **Main Execution**
The script reads input from `TestCases.json`, performs inference for each test case, and writes the results to `Sample_Testcase_Output.json`.

## Input and Output Formats

### Input Format
The input is a JSON file containing multiple test cases. Each test case includes:
- `VariablesCount`: Number of variables.
- `Potentials_count`: Number of potentials.
- `Cliques and Potentials`: List of cliques and their associated potentials.
- `k value (in top k)`: The number of top assignments to compute.

### Output Format
The output is a JSON file containing:
- `Marginals`: Marginal probabilities for each variable.
- `Top_k_assignments`: The top-k most probable assignments with their probabilities.
- `Z_value`: The partition function value.

## How to Run

1. Place the input file (`TestCases.json`) in the same directory as the script.
2. Run the script:
   ```bash
   python template.py
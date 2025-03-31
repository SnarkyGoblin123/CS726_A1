import json
import itertools
import collections
import heapq
import functools
import math
from collections import defaultdict

def multiply_potentials(p_large, p_small, axes, clique_vars):
    from itertools import product
    
    new_potential = [0] * len(p_large)
    axes_indices = [clique_vars.index(var) for var in axes]
    num_vars = len(clique_vars)
    
    for idx, assignment in enumerate(product((0, 1), repeat=num_vars)):
        small_assignment = tuple(assignment[i] for i in axes_indices)
        small_idx = sum(b << i for i, b in enumerate(reversed(small_assignment)))
        new_potential[idx] = p_large[idx] * p_small[small_idx]

    return new_potential

def sum_out(potential, axes, clique_vars):
    axes_indices = [clique_vars.index(var) for var in axes] 
    num_vars = len(clique_vars)
    
    new_size = 2 ** (num_vars - len(axes_indices))
    new_potential = [0] * new_size

    for i, value in enumerate(potential):
       
        assignment = [(i >> j) & 1 for j in range(num_vars - 1, -1, -1)]

        reduced_assignment = [assignment[j] for j in range(num_vars) if j not in axes_indices]

        new_index = sum(val << (len(reduced_assignment) - 1 - idx) for idx, val in enumerate(reduced_assignment))
        new_potential[new_index] += value

    return new_potential

def forward_pass(junction_tree, cliques, cliques_set, potentials, messages, node, parent, visited):
    visited.add(node)

    for neighbor in junction_tree[node]:  
        if neighbor != parent and neighbor not in visited:  

            intersection = cliques_set[node].intersection(cliques_set[neighbor])  
            
            message = potentials[node]  

            if parent is not None:
                parent_intersection = cliques_set[node].intersection(cliques_set[parent])
                message = multiply_potentials(message, messages[(parent, node)], list(parent_intersection), cliques[node]) 

            shape = [2] * len(cliques[node])  
            axes_indices = [var for var in (cliques_set[node] - intersection)]

            message = sum_out(message, axes_indices, cliques[node])  
            messages[(node, neighbor)] = message  

            forward_pass(junction_tree, cliques, cliques_set, potentials, messages, neighbor, node, visited)  

def backward_pass(junction_tree, cliques, cliques_set, potentials, messages, node, parent, visited):
    visited.add(node)

    for neighbor in junction_tree[node]:  
        if neighbor != parent and neighbor not in visited:  
            backward_pass(junction_tree, cliques, cliques_set, potentials, messages, neighbor, node, visited)

    if parent is not None:

        message = potentials[node]

        for neighbor in junction_tree[node]:
            if neighbor != parent:
                message = multiply_potentials(message, messages[(neighbor, node)], list(cliques_set[node].intersection(cliques_set[neighbor])), cliques[node])

        intersection = cliques_set[node].intersection(cliques_set[parent])

        axes_indices = [var for var in (cliques_set[node] - intersection)]
        message = sum_out(message, axes_indices, cliques[node])

        messages[(node, parent)] = message

class Inference:
    def __init__(self, data):

        self.variables_count = data['VariablesCount']
        self.potentials_count = data['Potentials_count']
        self.cliques_and_potentials = data['Cliques and Potentials']
        self.k_value = data['k value (in top k)']
        self.cliques = []
        self.junction_tree = {}
        self.potentials = {}
        self.moral_graph = self.create_moral_graph()
        self.variable_domains = {i: [0, 1] for i in range(self.variables_count)}
        self.marginals = []
        self.z = 0

    def create_moral_graph(self):

        graph = {}

        for i in range(self.variables_count):
            graph[i] = set()

        for clique_data in self.cliques_and_potentials:
            clique = clique_data['cliques']

            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    graph[clique[i]].add(clique[j])
                    graph[clique[j]].add(clique[i])

        return graph

    def triangulate_and_get_cliques(self):

        graph = {node: set(neighbors) for node, neighbors in self.moral_graph.items()}
        elimination_order = self.min_fill_heuristic(graph)
        all_nodes = set(graph.keys())  

        for node in elimination_order:
            if node in graph: 
                neighbors = list(graph[node])
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        if neighbors[i] in graph and neighbors[j] in graph: 
                            graph[neighbors[i]].add(neighbors[j])
                            graph[neighbors[j]].add(neighbors[i])

                clique = set([node] + neighbors)

                is_maximal = True
                for other_node in all_nodes - clique:
                    can_extend = True

                    for clique_node in clique:
                        if other_node not in self.moral_graph[clique_node]:
                            can_extend = False
                            break
                    if can_extend:
                        is_maximal = False
                        break

                if is_maximal:
                    self.cliques.append(clique)

                del graph[node]
                for neighbor in neighbors:
                    if neighbor in graph: 
                        graph[neighbor].discard(node)
                        
        final_cliques = []
        for i in range(len(self.cliques)):
            is_subset = False

            for j in range(len(self.cliques)):
                if i != j and set(self.cliques[i]).issubset(set(self.cliques[j])):
                    is_subset = True
                    break

            if not is_subset:
                final_cliques.append(list(self.cliques[i]))

        self.cliques = final_cliques
        self.cliques = [list(clique) for clique in self.cliques]

    def min_fill_heuristic(self, graph):
        fill_in_counts = {}

        for node in graph:
            neighbors = list(graph[node])
            count = 0

            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] not in graph[neighbors[i]]:
                        count += 1
            fill_in_counts[node] = count

        return sorted(graph.keys(), key=lambda x: fill_in_counts[x])

    def get_junction_tree(self):

        jt_graph = {}

        for i in range(len(self.cliques)):
            jt_graph[i] = {}
            for j in range(len(self.cliques)):
                if i != j:
                    intersect = set(self.cliques[i]) & set(self.cliques[j])
                    weight = len(intersect)
                    if weight > 0:
                        jt_graph[i][j] = weight

        self.junction_tree = self.maximum_spanning_tree(jt_graph)

    def maximum_spanning_tree(self, graph):

        edges = []
        for node in graph:
            for neighbor, weight in graph[node].items():
                edges.append((weight, node, neighbor))
        edges.sort(reverse=True)

        parent = {node: node for node in graph}

        def find(node):
            if parent[node] == node:
                return node
            parent[node] = find(parent[node])
            return parent[node]

        def union(node1, node2):
            root1 = find(node1)
            root2 = find(node2)
            parent[root1] = root2

        mst = {}
        for node in graph:
            mst[node] = {}

        for weight, node1, node2 in edges:
            if find(node1) != find(node2):
                union(node1, node2)
                mst[node1][node2] = weight
                mst[node2][node1] = weight

        return mst
    
    def process_cliques(self):
        def reorder_potentials(clique, potential):
            sorted_clique = sorted(clique)  
            original_indices = {var: i for i, var in enumerate(clique)}  
            new_order = [original_indices[var] for var in sorted_clique]  

            num_entries = len(potential)
            num_vars = len(clique)

            reordered_potential = [0] * num_entries  

            for old_index in range(num_entries):
                old_assignment = list(map(int, format(old_index, f'0{num_vars}b')))  
                new_assignment = [old_assignment[new_order[i]] for i in range(num_vars)]  
                new_index = int("".join(map(str, new_assignment)), 2)  

                reordered_potential[new_index] = potential[old_index]

            return sorted_clique, reordered_potential

        new_cliques_and_potentials = []

        for entry in self.cliques_and_potentials: 
            clique = tuple(entry["cliques"])  
            potential = entry["potentials"]   

            sorted_clique, reordered_potential = reorder_potentials(list(clique), potential)

            new_cliques_and_potentials.append({
                "clique_size": len(sorted_clique),
                "cliques": sorted_clique,
                "potentials": reordered_potential
            })

        new_cliques_and_potentials.sort(key=lambda x: x["cliques"])  

        original_cliques = {
            tuple(entry["cliques"]): entry["potentials"]
            for entry in new_cliques_and_potentials
        }
        return new_cliques_and_potentials, original_cliques 

    def assign_potentials_to_cliques(self):
        def merge_duplicate_cliques(cliques_and_potentials):
            merged_cliques = {}
            for entry in cliques_and_potentials:
                clique = tuple(sorted(entry['cliques']))
                potential = entry['potentials']
                
                if clique not in merged_cliques:
                    merged_cliques[clique] = potential[:]
                else:
                    merged_cliques[clique] = [a * b for a, b in zip(merged_cliques[clique], potential)]
            
            return [{"clique_size": len(k), "cliques": list(k), "potentials": v} for k, v in merged_cliques.items()]
        
        self.cliques_and_potentials = merge_duplicate_cliques(self.cliques_and_potentials)

        def generate_binary_assignments(num_vars):
            return [list(map(int, format(i, f'0{num_vars}b'))) for i in range(2 ** num_vars)]

        def merge_two_potentials(subclique1, subpotential1, subclique2, subpotential2):
            merged_clique = sorted(set(subclique1) | set(subclique2)) 
            merged_potential_size = 2 ** len(merged_clique)
            merged_potential = [0] * merged_potential_size
            sub1_index = {var: i for i, var in enumerate(subclique1)}
            sub2_index = {var: i for i, var in enumerate(subclique2)}
            merged_index = {var: i for i, var in enumerate(merged_clique)}
            merged_assignments = generate_binary_assignments(len(merged_clique))

            for idx, merged_assignment in enumerate(merged_assignments):
                sub1_assignment = [merged_assignment[merged_index[var]] for var in subclique1]
                sub2_assignment = [merged_assignment[merged_index[var]] for var in subclique2]
                sub1_index_val = int("".join(map(str, sub1_assignment)), 2)
                sub2_index_val = int("".join(map(str, sub2_assignment)), 2)

                if sub1_index_val < len(subpotential1) and sub2_index_val < len(subpotential2):
                    merged_potential[idx] = subpotential1[sub1_index_val] * subpotential2[sub2_index_val]

            return merged_clique, merged_potential

        def merge_all_subcliques(subcliques, potentials):
            if not subcliques:
                return [], []

            merged_clique, merged_potential = subcliques[0], potentials[0]

            for i in range(1, len(subcliques)):
                merged_clique, merged_potential = merge_two_potentials(
                    merged_clique, merged_potential, subcliques[i], potentials[i]
                )

            return merged_clique, merged_potential

        result, original_cliques = self.process_cliques()

        clique_potential_map = {} 
        for i, clique in enumerate(sorted(self.cliques), start=1):  
                clique_tuple = tuple(sorted(clique))  

                overlapping_cliques = []

                for sub_clique in original_cliques.keys():
                    if set(sub_clique).issubset(set(clique_tuple)):  
                        overlapping_cliques.append((sub_clique, original_cliques[sub_clique]))

                if overlapping_cliques:
                    
                    subcliques, potentials = zip(*overlapping_cliques)

                    
                    merged_clique, merged_potential = merge_all_subcliques(subcliques, potentials)
                    clique_potential_map[clique_tuple] = merged_potential

                    
                    for sub_clique in subcliques:
                        original_cliques[sub_clique] = [1] * len(original_cliques[sub_clique])  

                else:
                    
                    clique_potential_map[clique_tuple] = [0] * (2 ** len(clique_tuple))

        junction_tree_index_map = {tuple(sorted(clique)): idx for idx, clique in enumerate(self.cliques)}
        modified_clique_potential_map = {
            junction_tree_index_map[tuple(sorted(clique))]: potential
            for clique, potential in clique_potential_map.items()
        }
        self.potentials = modified_clique_potential_map

    def get_z_value(self):
        self.comp_marginals()
        self.z = self.marginals[0][0] + self.marginals[0][1]
        self.marginals = [[val/self.z for val in marginal] for marginal in self.marginals]

        return self.z

    def comp_marginals(self):
        messages = {}

        junction_tree = self.junction_tree
        potentials = self.potentials
        cliques = self.cliques
        cliques_set = {i: set(clique) for i, clique in enumerate(cliques)}

        visited = set()
        forward_pass(junction_tree, cliques, cliques_set, potentials, messages, node=0, parent=None, visited=visited)
        visited = set()
        backward_pass(junction_tree, cliques, cliques_set, potentials, messages, node=0, parent=None, visited=visited)
        
        
        clique_potentials = {}

        for node,_ in enumerate(cliques):
            potential = potentials[node]
            for neighbor in junction_tree[node]:
                if (neighbor, node) in messages:
                    potential = multiply_potentials(potential, messages[(neighbor, node)], list(cliques_set[neighbor].intersection(cliques_set[node])), cliques[node])

            clique_potentials[node] = potential

        node_marginals = []
        visited = set()
        for i,clique in enumerate(cliques):
            for node in clique:
                if node not in visited:
                    remaining_vars = [v for v in clique if v != node]
                    marginal = sum_out(clique_potentials[i], remaining_vars, clique)
                    node_marginals.append([node,marginal])
                visited.add(node)
        
        self.marginals = [marginal for _, marginal in sorted(node_marginals, key=lambda x: x[0])]
    
    def compute_marginals(self):
        return self.marginals

    def compute_top_k(self):
        assignments_with_probs = []
        
        for assignment in itertools.product(*[list(self.variable_domains[i]) for i in range(self.variables_count)]):
            probability = 1.0
            
            for clique_index, potentials in self.potentials.items():
                clique = self.cliques[clique_index]
                variables_in_clique = clique if isinstance(clique, (list, tuple)) else [clique]
                
                var_to_index = {var: i for i, var in enumerate(variables_in_clique)}
                
                clique_assignment = [assignment[var] for var in variables_in_clique]
                
                assignment_index = sum(val * (2 ** i) for i, val in enumerate(reversed(clique_assignment)))
                
                probability *= potentials[assignment_index]
            
            assignments_with_probs.append((list(assignment), probability))
        
        sorted_assignments = sorted(assignments_with_probs, key=lambda x: x[1], reverse=True)

        top_k = [
            {'assignment': assignment, 'probability': probability / self.z}
            for assignment, probability in sorted_assignments[:self.k_value]
        ]
        
        return top_k



class Get_Input_and_Check_Output:
    def __init__(self, file_name):
        with open(file_name, 'r') as file:
            self.data = json.load(file)
    
    def get_output(self):
        n = len(self.data)
        output = []
        for i in range(n):
            inference = Inference(self.data[i]['Input'])
            inference.triangulate_and_get_cliques()
            inference.get_junction_tree()
            inference.assign_potentials_to_cliques()
            z_value = inference.get_z_value()
            marginals = inference.compute_marginals()
            top_k_assignments = inference.compute_top_k()
            output.append({
                'Marginals': marginals,
                'Top_k_assignments': top_k_assignments,
                'Z_value' : z_value
            })
        self.output = output

    def write_output(self, file_name):
        with open(file_name, 'w') as file:
            json.dump(self.output, file, indent=4)


if __name__ == '__main__':
    evaluator = Get_Input_and_Check_Output('TestCases.json')
    evaluator.get_output()
    evaluator.write_output('Sample_Testcase_Output.json')
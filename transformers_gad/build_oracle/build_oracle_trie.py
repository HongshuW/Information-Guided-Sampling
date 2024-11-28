from graphviz import Digraph
import torch
import pickle

class TrieNode:
    def __init__(self, token_id=None, token=None, raw_logit=None, raw_score=None):
        self.children = {}
        self.parent = None
        self.token_id = token_id
        self.token = token
        self.raw_logit = raw_logit
        self.raw_score = raw_score
        self.success_rate = 1
        # isEndOfWord is True if node represent EOS
        self.is_end_of_sequence = False
        self.is_start_of_sequence = False

    def __repr__(self):
        parent_token_id = 'None (Root Node)' if self.parent is None else self.parent.token_id
        return (f"TrieNode(token_id={self.token_id}, token='{self.token}', "
                f"raw_logit={self.raw_logit}, raw_score={self.raw_score}, children={list(self.children.keys())}, "
                f"parent={parent_token_id}, success rate={self.success_rate})") # TODO: add prefix



class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.root.is_start_of_sequence = True

    def insert(self, parent_node: TrieNode, child_node: TrieNode):
        # Insert child_node into parent_node's children dictionary
        if child_node.token_id not in parent_node.children:
            parent_node.children[child_node.token_id] = child_node
            child_node.parent = parent_node  # Set the parent of the child_node
            if child_node.token_id == 2: # TODO: replace to real EOS token
                child_node.is_end_of_sequence = True
            # update the success rate of the parent node
            return self.update_success_rate(parent_node)
        else:
            return 0

    def update_success_rate(self, node: TrieNode):
        if node and node.children:
            total_success_rate = sum(child.raw_logit * child.success_rate for child in node.children.values())
            updated_rate = node.success_rate - total_success_rate
            node.success_rate = total_success_rate
            if node.parent:
                return self.update_success_rate(node.parent)
            return updated_rate

    def search_last_parent(self, prefix: torch.LongTensor):
        found_parent = []
        current_parent = self.root
        for time_step, token_id in enumerate(prefix[0]): # still assume one batch
            token_id = token_id.item()
            if token_id in current_parent.children.keys():
                current_parent = current_parent.children[token_id]
                found_parent.append(current_parent.token_id)
            else:
                print(
                    f"last parent found is {found_parent}; current {token_id} not found in the trie at time step {time_step}")
                return None
        return current_parent

    def search_token_from_parent(self, parent_node, candidate_token_id):
        if parent_node is None:
            return None
        if candidate_token_id in parent_node.children.keys():
            return parent_node.children[candidate_token_id]
        else:
            return None

    def get_informativeness_for_candidate_token(self, parent_node, candidate_token_id):
        if parent_node is None:
            return 1
        if candidate_token_id in parent_node.children.keys():
            return 0
        else:
            return 1

    def get_success_rate_for_candidate_token(self, parent_node, candidate_token_id):
        if parent_node is None:
            return 1
        if candidate_token_id in parent_node.children.keys():
            return parent_node.children[candidate_token_id].success_rate
        else:
            return 1

    def search(self, sequence):
        node = self.root
        for token_id in sequence:
            if token_id not in node.children:
                return False
            node = node.children[token_id]
        return node.is_end_of_sequence # TODO: also return if not end of sequence

    def print_trie(self, node=None, prefix=None):
        if node is None:
            node = self.root
        if prefix is None:
            prefix = []

        # If current node marks the end of a sequence, print the prefix as a list
        if node.is_end_of_sequence: # TODO: also print if not end of sequence
            print(prefix)

        # Recursively call print_trie for all children, appending the current character/token to the prefix
        for char, child_node in node.children.items():
            self.print_trie(child_node, prefix + [char])

    def has_full_information(self):
        """
        Checks if all paths in the trie end with an is_end_of_sequence node set to True.
        Returns True if the trie has full information, False otherwise.
        """
        return self._check_full_information(self.root)

    def _check_full_information(self, node):
        # If the node has no children, check if it is marked as the end of a sequence
        if not node.children:
            return node.is_end_of_sequence

        # Recursively check all children
        return all(self._check_full_information(child) for child in node.children.values())

    def print_all_nodes(self, node=None, depth=0):
        if node is None:
            node = self.root

        # Print current node's details
        indent = "  " * depth  # Create indentation based on the depth in the trie
        node_details = (f"{indent}TrieNode(token_id={node.token_id}, token='{node.token}', "
                        f"raw_logit={node.raw_logit}, raw_score={node.raw_score}, success rate={node.success_rate}, "
                        f"children={list(node.children.keys())}, "
                        f"parent={node.parent.token_id if node.parent else None}, "
                        f"is_end_of_sequence={node.is_end_of_sequence})")
        print(node_details)

        # Recursively call print_all_nodes for all children
        for child_node in node.children.values():
            self.print_all_nodes(child_node, depth + 1)

def visualize_trie(trie_root): # TODO: problem with visualization
    def add_nodes_edges(node, graph, parent=None):
        # Create a unique identifier for the current node
        node_id = id(node)

        # Add the current node to the graph


        # If this node has a parent, add an edge from the parent to this node
        if parent:
            parent_prob_sum_gcd = sum([node.raw_logit for node in parent.children.values()])
            gcd_prob = node.raw_logit / parent_prob_sum_gcd

            parent_prob_sum_gad = sum([node.raw_logit * node.success_rate for node in parent.children.values()])
            gad_prob = node.raw_logit * node.success_rate / parent_prob_sum_gad
            
            label = f"{node.token} (ID: {node.token_id}, LLM's Prob: {node.raw_logit}, GCD Prob: {gcd_prob}, GAD Prob: {gad_prob})"
            graph.node(str(node_id), label=label)

            graph.edge(str(id(parent)), str(node_id))
        else:
            label = f"{node.token} (Root)"
            graph.node(str(node_id), label=label)

        # Recursively add nodes/edges for the children
        for child in node.children.values():
            add_nodes_edges(child, graph, parent=node)

    graph = Digraph()
    add_nodes_edges(trie_root, graph)
    return graph

def create_nodes_from_history(detailed_history):
    nested_nodes = []
    for time_step in detailed_history:
        step_nodes = []
        for batch in time_step:
            batch_nodes = []
            for node_info in batch:
                node = TrieNode(token_id=node_info['token_id'], token=node_info['token'], raw_logit=node_info['raw_logit'], raw_score=node_info['raw_score'])
                batch_nodes.append(node)
            step_nodes.append(batch_nodes)
        nested_nodes.append(step_nodes)
    return nested_nodes

def get_selected_token_id_at_time_step(tokens, time_step):
    # Assuming there is only one batch TODO: handle multiple batches
    if time_step < len(tokens[0]):
        return tokens[0][time_step].item()
    else:
        raise ValueError(f"Time step {time_step} is out of range for the given tokens")

def insert_nodes_by_generated_tokens(trie, generated_tokens, nodes):
    current_parent = trie.root

    updated_total = 0

    for time_step, candidate_list in enumerate(nodes):
        selected_token_id = get_selected_token_id_at_time_step(generated_tokens, time_step)
        found_parent_for_next_step = False

        for batch in candidate_list:
            for node in batch:
                # Insert node only if it doesn't already exist as a child of the current parent.
                if node.token_id not in current_parent.children.keys():
                    # print(f"current_parent={current_parent} at time step {time_step}")
                    # print(f"Inserting node {node.token_id} at time step {time_step}")
                    updated_total += trie.insert(current_parent, node)

                else:
                    # if args.verbose:
                        # print(f"Node {node.token_id} already exists as a child of the current parent at time step {time_step}")
                    pass

                # Check if this node matches the next token ID and should be the next parent.
                if node.token_id == selected_token_id:
                    next_parent_candidate = current_parent.children[node.token_id]
                    found_parent_for_next_step = True

        if found_parent_for_next_step:
            current_parent = next_parent_candidate
            # print(f"current_parent_token_id={current_parent.token_id} at time step {time_step}")
        else:
            print(f"No matching child found for next parent at time step {time_step}")

    return updated_total

def run_demo_trie_string_01_len_3():
    # Your input
    generated_tokens = torch.tensor([[28740, 28734, 28740, 2]])
    detailed_history = [
        [[{'token_id': 28734, 'token': '0', 'raw_logit': 2.0020976080559194e-05},
          {'token_id': 28740, 'token': '1', 'raw_logit': 3.175825986545533e-05}]],
        [[{'token_id': 28734, 'token': '0', 'raw_logit': 0.22081588208675385},
          {'token_id': 28740, 'token': '1', 'raw_logit': 0.17334003746509552}]],
        [[{'token_id': 28734, 'token': '0', 'raw_logit': 0.11201044172048569},
          {'token_id': 28740, 'token': '1', 'raw_logit': 0.1298251450061798}]],
        [[{'token_id': 2, 'token': '</s>', 'raw_logit': 0.7505450248718262}]]
    ]

    trie = Trie()

    nodes = create_nodes_from_history(detailed_history)
    insert_nodes_by_generated_tokens(trie, generated_tokens, nodes)
    #
    # for node in nodes:
    #     print(node)
    # print(trie.root)
    print(f"====updated trie====")
    updated_generated_tokens = torch.tensor([[28740, 28734, 28734, 2]])
    acceptance_details_history = [[[{'token_id': 28734, 'token': '0', 'raw_score': 2.6377220153808594,
                                    'raw_logit': 2.0020976080559194e-05},
                                   {'token_id': 28740, 'token': '1', 'raw_score': 3.0990943908691406,
                                    'raw_logit': 3.175825986545533e-05}]],
                                  [[{'token_id': 28734, 'token': '0', 'raw_score': 10.416403770446777,
                                    'raw_logit': 0.22081588208675385},
                                    {'token_id': 28740, 'token': '1', 'raw_score': 10.17432975769043,
                                    'raw_logit': 0.17334003746509552}]],
                                  [[{'token_id': 28734, 'token': '0', 'raw_score': 9.51977825164795,
                                    'raw_logit': 0.11201044172048569},
                                    {'token_id': 28740, 'token': '1', 'raw_score': 9.667374610900879,
                                    'raw_logit': 0.1298251450061798}]],
                                 [[{'token_id': 2, 'token': '</s>', 'raw_score': 11.959583282470703,
                                    'raw_logit': 0.35645970702171326}]]]

    updated_nodes = create_nodes_from_history(acceptance_details_history)
    insert_nodes_by_generated_tokens(trie, updated_generated_tokens, updated_nodes)

    # for node in nodes:
    #     print(node)
    # print(trie.root)

    trie.print_all_nodes()  # Print all nodes in the trie

    # visualize_trie(trie.root)
    trie.print_trie(trie.root)
    sequence = [28740, 28734, 28734, 2]
    prefix = torch.tensor([[28740, 28734, 28734]])
    last_parent = trie.search_last_parent(prefix)
    # print(f"last parent found is {last_parent}")
    search_result = trie.search(sequence)
    # print(f"Is the sequence in the trie? {search_result}")
    #
    # print(f"Does the trie have full information? {trie.has_full_information()}")

    return trie

def update_oracle_trie(trie, generated_tokens, detailed_history):
    nodes = create_nodes_from_history(detailed_history)
    updated_rate = insert_nodes_by_generated_tokens(trie, generated_tokens, nodes)

    return trie, updated_rate

if __name__ == "__main__":
    accpetance_details_history = [[[{'token_id': 450, 'token': 'de', 'raw_score': -0.7717133164405823, 'raw_logit': 5.726020617657923e-07}, {'token_id': 1270, 'token': 'def', 'raw_score': -0.30284133553504944, 'raw_logit': 9.15127202461008e-07}, {'token_id': 3380, 'token': 'define', 'raw_score': -0.4961015284061432, 'raw_logit': 7.543096103290736e-07}, {'token_id': 28715, 'token': 'd', 'raw_score': 1.117053747177124, 'raw_logit': 3.785594344662968e-06}]], [[{'token_id': 797, 'token': 'ef', 'raw_score': -1.157513976097107, 'raw_logit': 4.726879978989018e-06}, {'token_id': 28706, 'token': 'e', 'raw_score': -1.509982705116272, 'raw_logit': 3.32276022163569e-06}]], [[{'token_id': 262, 'token': 'in', 'raw_score': 6.868011474609375, 'raw_logit': 0.006728281732648611}, {'token_id': 473, 'token': 'ine', 'raw_score': 7.943443298339844, 'raw_logit': 0.019722331315279007}, {'token_id': 28710, 'token': 'i', 'raw_score': 4.006861209869385, 'raw_logit': 0.0003848773776553571}]], [[{'token_id': 28733, 'token': '-', 'raw_score': 13.678215980529785, 'raw_logit': 0.9493529796600342}]], [[{'token_id': 1755, 'token': 'fun', 'raw_score': 15.271430969238281, 'raw_logit': 0.889732301235199}, {'token_id': 21278, 'token': 'fu', 'raw_score': 5.882600784301758, 'raw_logit': 7.442900096066296e-05}, {'token_id': 28722, 'token': 'f', 'raw_score': 8.420909881591797, 'raw_logit': 0.0009421408758498728}]], [[{'token_id': 285, 'token': 'f', 'raw_score': 10.483269691467285, 'raw_logit': 0.32588133215904236}, {'token_id': 28705, 'token': '', 'raw_score': 4.84146785736084, 'raw_logit': 0.0011557291727513075}]], [[{'token_id': 325, 'token': '(', 'raw_score': 12.807836532592773, 'raw_logit': 0.4081755578517914}, {'token_id': 2743, 'token': '((', 'raw_score': 13.060338973999023, 'raw_logit': 0.5254209637641907}, {'token_id': 28705, 'token': '', 'raw_score': 7.036534309387207, 'raw_logit': 0.0012717515928670764}]], [[{'token_id': 28744, 'token': 'x', 'raw_score': 15.41815185546875, 'raw_logit': 0.9838594198226929}]], [[{'token_id': 325, 'token': '(', 'raw_score': 14.86163330078125, 'raw_logit': 0.9640588760375977}, {'token_id': 28705, 'token': '', 'raw_score': 7.49530029296875, 'raw_logit': 0.0006094607524573803}]], [[{'token_id': 8443, 'token': 'Bit', 'raw_score': 17.507558822631836, 'raw_logit': 0.9950070381164551}, {'token_id': 27405, 'token': 'Bi', 'raw_score': 6.443575859069824, 'raw_logit': 1.558832445880398e-05}, {'token_id': 28760, 'token': 'B', 'raw_score': 9.619821548461914, 'raw_logit': 0.00037344390875659883}]], [[{'token_id': 12790, 'token': 'Vec', 'raw_score': 21.018150329589844, 'raw_logit': 0.9996631145477295}, {'token_id': 28790, 'token': 'V', 'raw_score': 10.285603523254395, 'raw_logit': 2.1815631043864414e-05}]], [[{'token_id': 28705, 'token': '', 'raw_score': 17.537147521972656, 'raw_logit': 0.9985687732696533}]], [[{'token_id': 28784, 'token': '6', 'raw_score': 20.66687774658203, 'raw_logit': 0.9911807179450989}]], [[{'token_id': 28781, 'token': '4', 'raw_score': 21.145801544189453, 'raw_logit': 0.999852180480957}]], [[{'token_id': 743, 'token': '))', 'raw_score': 14.427480697631836, 'raw_logit': 0.13848435878753662}, {'token_id': 5429, 'token': ')))', 'raw_score': 16.240503311157227, 'raw_logit': 0.8487630486488342}, {'token_id': 28731, 'token': ')', 'raw_score': 8.08746337890625, 'raw_logit': 0.00024432403733953834}]], [[{'token_id': 325, 'token': '(', 'raw_score': 15.819817543029785, 'raw_logit': 0.9808996915817261}, {'token_id': 28705, 'token': '', 'raw_score': 8.734001159667969, 'raw_logit': 0.0008209064253605902}]], [[{'token_id': 8443, 'token': 'Bit', 'raw_score': 17.44237518310547, 'raw_logit': 0.997101366519928}, {'token_id': 27405, 'token': 'Bi', 'raw_score': 5.704566478729248, 'raw_logit': 7.962949894135818e-06}, {'token_id': 28760, 'token': 'B', 'raw_score': 7.792175769805908, 'raw_logit': 6.422598380595446e-05}]], [[{'token_id': 12790, 'token': 'Vec', 'raw_score': 21.55828857421875, 'raw_logit': 0.999810516834259}, {'token_id': 28790, 'token': 'V', 'raw_score': 9.162976264953613, 'raw_logit': 4.137156793149188e-06}]], [[{'token_id': 28705, 'token': '', 'raw_score': 17.87997055053711, 'raw_logit': 0.9991050362586975}]], [[{'token_id': 28784, 'token': '6', 'raw_score': 19.989408493041992, 'raw_logit': 0.9945782423019409}]], [[{'token_id': 28781, 'token': '4', 'raw_score': 23.145023345947266, 'raw_logit': 0.9999349117279053}]], [[{'token_id': 3847, 'token': ')(', 'raw_score': 10.47452449798584, 'raw_logit': 0.0015017997939139605}, {'token_id': 28731, 'token': ')', 'raw_score': 16.967388153076172, 'raw_logit': 0.9918063879013062}]], [[{'token_id': 28732, 'token': '(', 'raw_score': 2.8034279346466064, 'raw_logit': 4.383458872325718e-06}, {'token_id': 28744, 'token': 'x', 'raw_score': 3.5982139110565186, 'raw_logit': 9.704838703328278e-06}, {'token_id': 28771, 'token': '#', 'raw_score': 3.9475812911987305, 'raw_logit': 1.3763108654529788e-05}]], [[{'token_id': 28744, 'token': 'x', 'raw_score': 9.777077674865723, 'raw_logit': 0.3110715448856354}]], [[{'token_id': 28734, 'token': '0', 'raw_score': 12.778416633605957, 'raw_logit': 0.5037944912910461}]], [[{'token_id': 28734, 'token': '0', 'raw_score': 15.029123306274414, 'raw_logit': 0.9724351763725281}]], [[{'token_id': 28734, 'token': '0', 'raw_score': 15.574552536010742, 'raw_logit': 0.9830642342567444}]], [[{'token_id': 28734, 'token': '0', 'raw_score': 16.288055419921875, 'raw_logit': 0.9829287528991699}]], [[{'token_id': 28734, 'token': '0', 'raw_score': 13.53431510925293, 'raw_logit': 0.7429170608520508}]], [[{'token_id': 28734, 'token': '0', 'raw_score': 15.909515380859375, 'raw_logit': 0.9827061891555786}]], [[{'token_id': 28734, 'token': '0', 'raw_score': 16.22516632080078, 'raw_logit': 0.9868685603141785}]], [[{'token_id': 28734, 'token': '0', 'raw_score': 17.55841827392578, 'raw_logit': 0.9945131540298462}]], [[{'token_id': 28734, 'token': '0', 'raw_score': 16.1821231842041, 'raw_logit': 0.9771430492401123}]], [[{'token_id': 28734, 'token': '0', 'raw_score': 18.16150665283203, 'raw_logit': 0.9960071444511414}]], [[{'token_id': 28734, 'token': '0', 'raw_score': 17.7294921875, 'raw_logit': 0.9962248802185059}]], [[{'token_id': 28734, 'token': '0', 'raw_score': 18.402061462402344, 'raw_logit': 0.996239423751831}]], [[{'token_id': 28734, 'token': '0', 'raw_score': 17.487159729003906, 'raw_logit': 0.9930598735809326}]], [[{'token_id': 28734, 'token': '0', 'raw_score': 18.029006958007812, 'raw_logit': 0.9925633072853088}]], [[{'token_id': 28734, 'token': '0', 'raw_score': 16.296546936035156, 'raw_logit': 0.9543120265007019}]], [[{'token_id': 28734, 'token': '0', 'raw_score': 15.679858207702637, 'raw_logit': 0.72952800989151}, {'token_id': 28740, 'token': '1', 'raw_score': 14.12304401397705, 'raw_logit': 0.15378931164741516}]], [[{'token_id': 2, 'token': '</s>', 'raw_score': 9.176734924316406, 'raw_logit': 0.030386043712496758}]]]
    nodes = create_nodes_from_history(accpetance_details_history)
    print(nodes)
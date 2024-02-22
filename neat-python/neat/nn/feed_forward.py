from neat.graphs import feed_forward_layers


class FeedForwardNetwork(object):
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = dict((key, 0.0) for key in inputs + outputs)

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            node_inputs = []
            for i, w in links:
                node_inputs.append(self.values[i] * w)
            s = agg_func(node_inputs)
            self.values[node] = act_func(bias + response * s)

        return [self.values[i] for i in self.output_nodes]

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

        # Gather expressed connections.
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]

        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        inputs.append((inode, cg.weight))

                ng = genome.nodes[node]
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                activation_function = config.genome_config.activation_defs.get(ng.activation)
                node_evals.append((node, activation_function, aggregation_function, ng.bias, ng.response, inputs))

        return FeedForwardNetwork(config.genome_config.input_keys, config.genome_config.output_keys, node_evals)




import jax
import jax.numpy as jnp

class JAXFeedForwardNetwork(object):
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals  # This needs to be adapted for JAX
        self.max_node_index = max(inputs + outputs)  # Assuming inputs and outputs are node indices

    def activate(self, inputs):
        # Convert inputs to a JAX array
        input_values = jnp.array(inputs)
        
        # Initialize values with zeros for all nodes, then update inputs
        values = jnp.zeros(self.max_node_index + 1)  # Assuming max_node_index is defined
        values = values.at[jnp.array(self.input_nodes)].set(input_values)
        
        # Process each node according to its evaluation setup
        # This needs to be adapted to use JAX operations instead of Python loops
        for node, act_func, agg_func, bias, response, links in self.node_evals:
            # Example: Compute the sum of inputs * weights for each node
            # Note: This is a simplification. You'll need to adapt it to your structure.
            inputs, weights = zip(*links)
            node_input_values = values[jnp.array(inputs)]
            node_weights = jnp.array(weights)
            weighted_sum = jnp.dot(node_input_values, node_weights)
            
            # Apply activation function (assuming it's a JAX-compatible function)
            node_value = act_func(bias + response * weighted_sum)
            
            # Update the value for the current node
            values = values.at[node].set(node_value)
        
        # Extract and return output values
        return values[jnp.array(self.output_nodes)]
    
    @staticmethod
    def create(genome, config):
        """
        Creates a JAX-compatible feed-forward network from a genome.

        Parameters:
        - genome: The genome representing the network structure.
        - config: Configuration information including node and connection details.

        Returns:
        A JAXFeedForwardNetwork instance.
        """
        # Gather expressed connections.
        connections = [(cg.key, cg.weight) for cg in genome.connections.values() if cg.enabled]
        inputs = config.genome_config.input_keys
        outputs = config.genome_config.output_keys

        # Precompute layers based on the connections (unchanged from original function)
        layers = feed_forward_layers(inputs, outputs, [conn[0] for conn in connections])
        
        # Prepare node evaluations in a format suitable for JAX operations
        node_evals = []
        for layer in layers:
            for node in layer:
                # Collect inputs and their weights for the current node
                inputs_weights = [(inode, weight) for (inode, onode), weight in connections if onode == node]
                
                # Lookup node details from genome
                ng = genome.nodes[node]
                activation_function = config.genome_config.activation_defs.get(ng.activation)
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                
                # For JAX, it's efficient to separate weights and input node indices
                input_indices, weights = zip(*inputs_weights) if inputs_weights else ([], [])
                node_evals.append({
                    'node': node,
                    'activation': activation_function,
                    'aggregation': aggregation_function,
                    'bias': ng.bias,
                    'response': ng.response,
                    'inputs': jnp.array(input_indices, dtype=jnp.int32),
                    'weights': jnp.array(weights, dtype=jnp.float32)
                })

        # Convert node_evals to a structure that's easily processed in a vectorized manner by JAX
        # This is a placeholder for how you might structure this data
        
        return JAXFeedForwardNetwork(inputs, outputs, node_evals)


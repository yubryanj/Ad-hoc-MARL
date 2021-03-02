import torch
from torch import nn



class multiheaded_attention_sa(nn.Module):

    def __init__(self,args):
        super(multiheaded_attention_sa, self).__init__()
        self.args = args
        
        self.observation_embedding = nn.Linear(1 , args.embedding_dimension)
        self.action_embedding = nn.Linear(1, args.embedding_dimension)

        self.attention = nn.MultiheadAttention(args.embedding_dimension, args.n_heads)
        
        
        layers_dimension        = [args.hidden_dimension for _ in range(args.hidden_layers)]
        layers_dimension[0]     = args.embedding_dimension
        layers_dimension[-1]    = args.output_dimension
        self.layers = nn.ModuleList([nn.Linear(layers_dimension[i],layers_dimension[i+1]) \
                                            for i in range(args.hidden_layers-1)])


    def forward(self, observation, action):
        batch_size = observation.shape[0]
        observation = observation.reshape(batch_size,-1,1)
        embedded_observation = self.observation_embedding(observation.float())

        action = action.reshape(batch_size,-1,1)
        action = self.action_embedding(action.float())
        
        attention_input = torch.cat((embedded_observation,action),dim=1).permute(1,0,2)
        
        output = torch.rand(1, batch_size, self.args.embedding_dimension)
        attended_action_embedding = self.attention(output, attention_input, attention_input, need_weights=False)[0].permute(1,0,2)

        x = attended_action_embedding
        # Generate prediction of next observation
        for layer in self.layers:
            x = layer(x)

        predictions = x
        
        return predictions


class multiheaded_attention(nn.Module):

    def __init__(self,args):
        super(multiheaded_attention, self).__init__()
        self.args = args
        self.observation_embedding_input_dimension = args.observation_input_dimension * args.max_number_of_agents

        # Uses a linear layer because observations are continuous
        self.observation_embedding = nn.Linear(self.observation_embedding_input_dimension , args.embedding_dimension)

        self.action_embedding = nn.Linear(args.action_input_dimension, args.embedding_dimension)
        self.attention = nn.MultiheadAttention(args.embedding_dimension, args.n_heads)
        
        
        layers_dimension        = [args.hidden_dimension for _ in range(args.hidden_layers)]
        layers_dimension[0]     = args.embedding_dimension * 2
        layers_dimension[-1]    = args.output_dimension
        self.layers = nn.ModuleList([nn.Linear(layers_dimension[i],layers_dimension[i+1]) \
                                            for i in range(args.hidden_layers-1)])


    def forward(self, observation, action):
        batch_size = observation.shape[0]
        observation = observation.reshape(batch_size,-1)
        embedded_observation = self.observation_embedding(observation.float())

        action = self.action_embedding(action.float()).permute(1,0,2)
        output = torch.rand(1, batch_size, self.args.embedding_dimension)
        attended_action_embedding = self.attention(output, action, action)[0].permute(1,0,2).squeeze()

        if batch_size == 1:
            embedded_observation = embedded_observation.squeeze()
            prediction_input = torch.cat((embedded_observation,attended_action_embedding),dim=0)
        else:
            prediction_input = torch.cat((embedded_observation,attended_action_embedding),dim=1)

        x = prediction_input
        # Generate prediction of next observation
        for layer in self.layers:
            x = layer(x)

        predictions = x
        
        return predictions


class attend_over_state_and_actions(nn.Module):
    def __init__(self, args):
        super(attend_over_state_and_actions, self).__init__()

        # Uses a linear layer because observations are continuous
        self.observation_embedding = nn.Linear(args.observation_input_dimension, args.embedding_dimension)

        # There are only five actions in the discrete space.  Thus, we can strictly use an embedding.
        self.action_embedding = nn.Embedding(args.number_of_actions, args.embedding_dimension)

        # Using attention layer
        self.attention = nn.Linear(args.embedding_dimension, 1)

        layers_dimension        = [args.hidden_dimension for _ in range(args.hidden_layers)]
        layers_dimension[0]     = args.embedding_dimension * 2
        layers_dimension[-1]    = args.output_dimension
        self.layers = nn.ModuleList([nn.Linear(layers_dimension[i],layers_dimension[i+1]) \
                                            for i in range(args.hidden_layers-1)])


    def forward(self, observation, action):
        # Embed the observation and action
        embedded_observation = self.observation_embedding(observation.float())
        embedded_action = self.action_embedding(action.long())

        # Generate a tensor of shape (batch, items, dimensions)
        embedded_input = torch.cat((embedded_observation,embedded_action),axis=1)

        # Feed the tensor into the transformer to learn encodings
        weights = []
        for embedding in embedded_input:
            weights.append(self.attention(embedding))

        # Convert weights into a tensor
        weights = torch.cat(weights,dim=1).T

        # Normalize the weights
        normalized_weights = nn.functional.softmax(weights, dim=1).unsqueeze(1)

        # Apply the attention weights to the action embeddings
        predictions_input = torch.bmm(normalized_weights, embedded_input).squeeze()

        x = predictions_input
        # Generate prediction of next observation
        for layer in self.layers:
            x = layer(x)

        predictions = x
        
        return predictions


class attend_over_actions(nn.Module):
    def __init__(self, args):
        super(attend_over_actions, self).__init__()

        self.args = args
        self.observation_embedding_input_dimension = args.observation_input_dimension * args.max_number_of_agents

        # Uses a linear layer because observations are continuous
        self.observation_embedding = nn.Linear(self.observation_embedding_input_dimension , args.embedding_dimension)

        # There are only five actions in the discrete space.  Thus, we can strictly use an embedding.
        # self.action_embedding = nn.Embedding(args.number_of_actions, args.embedding_dimension)
        self.action_embedding = nn.Linear(args.action_input_dimension, args.embedding_dimension)


        # Using attention layer
        self.attention = nn.Linear(args.embedding_dimension, 1)


        layers_dimension        = [args.hidden_dimension for _ in range(args.hidden_layers)]
        layers_dimension[0]     = args.embedding_dimension * 2
        layers_dimension[-1]    = args.output_dimension
        self.layers = nn.ModuleList([nn.Linear(layers_dimension[i],layers_dimension[i+1]) \
                                            for i in range(args.hidden_layers-1)])
        # Layer to create prediction of output
        # self.output = nn.Linear(args.embedding_dimension * 2, args.output_dimension)


    def forward(self, observation, action):

        batch_size, number_of_observations, observation_dimension = observation.shape
        padding_size = self.observation_embedding_input_dimension - (observation_dimension * number_of_observations)
        observation = observation.reshape(batch_size, -1)
        observation = nn.functional.pad(observation, (0, padding_size))

        # Embed the observation and action
        embedded_observation = self.observation_embedding(observation.float()).squeeze()
        embedded_action = self.action_embedding(action.float())

        # Feed the tensor into the transformer to learn encodings
        weights = []
        for embedding in embedded_action:
            weights.append(self.attention(embedding))

        # Convert weights into a tensor
        weights = torch.cat(weights,dim=1).T

        # Normalize the weights
        normalized_weights = nn.functional.softmax(weights, dim=1).unsqueeze(1)

        # Apply the attention weights to the embeddings
        attended_action_embedding = torch.bmm(normalized_weights, embedded_action).squeeze()

        if batch_size == 1:
            embedded_observation = embedded_observation.unsqueeze(0)
            attended_action_embedding = attended_action_embedding.unsqueeze(0)

        prediction_input = torch.cat((embedded_observation,attended_action_embedding),dim=1)

        x = prediction_input
        # Generate prediction of next observation
        for layer in self.layers:
            x = layer(x)

        predictions = x
        
        return predictions


class Feedforward(nn.Module):

    def __init__(self, args):
        super(Feedforward, self).__init__()

        layers_dimension        = [args.hidden_dimension for _ in range(args.hidden_layers)]
        layers_dimension[0]     = (args.observation_input_dimension + args.action_input_dimension) * args.max_number_of_agents
        layers_dimension[-1]    = args.output_dimension

        self.layers = nn.ModuleList([nn.Linear(layers_dimension[i],layers_dimension[i+1]) \
                                            for i in range(args.hidden_layers-1)])

    def forward(self, observation, action):
        batch_size = observation.shape[0]
        observation = observation.reshape(batch_size, -1)
        action = action.reshape(batch_size, -1)
        x = torch.cat((observation,action), axis=1).float()

        for layer in self.layers:
            x = layer(x)
        return x


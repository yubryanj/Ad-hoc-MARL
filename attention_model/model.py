import torch
from torch import nn
from torch.nn.functional import softmax


class model_a(nn.Module):
    
    def __init__(self,args):
        super(model_a, self).__init__()
        self.args = args
        
        self.state_embedding = nn.Linear(args.max_number_of_agents * args.observation_dimension , args.embedding_dimension)
        self.action_embedding = nn.Linear(args.action_dimension, args.embedding_dimension)

        self.q_projections = nn.Linear(args.max_number_of_agents, args.output_dimension)       
        self.kv_projection = nn.Linear(args.embedding_dimension * 2, 2 * (args.embedding_dimension+ args.embedding_dimension))        

        self.attention = nn.MultiheadAttention(args.embedding_dimension * 2, args.n_heads)

        self.predict = nn.Linear(args.embedding_dimension * 2, 1)


    def forward(self, state, action):
        batch_size = state.shape[0]

        # Prepare the state encodings
        state = state.reshape(batch_size, 1, -1)
        state_encoding = self.state_embedding(state.float())
        state_encoding = state_encoding.repeat(1,self.args.max_number_of_agents,1)      # Repeat the state for each agent

        # Prepare the action encodings
        action_encoding = self.action_embedding(action.float())

        # Concatenate the state with each action
        x = torch.cat((state_encoding, action_encoding),dim=2)

        # Attention
        query = self.q_projections(x.permute(0,2,1)).permute(2,0,1)
        key, value = self.kv_projection(x).permute(1,0,2).chunk(2,dim=2)
        x = self.attention(query, key, value)[0].permute(1,0,2)

        # Predict the next state
        x = self.predict(x)
        
        return x


class model_b(nn.Module):
    
    def __init__(self,args):
        super(model_b, self).__init__()
        self.args = args
        
        self.observation_embedding = nn.Linear(args.observation_dimension , args.embedding_dimension)
        self.action_embedding = nn.Linear(args.action_dimension, args.embedding_dimension)

        self.q_projections = nn.Linear(args.max_number_of_agents, args.output_dimension)       
        self.kv_projection = nn.Linear(args.embedding_dimension * 2, 2 * (args.embedding_dimension+ args.embedding_dimension))        

        self.attention = nn.MultiheadAttention(args.embedding_dimension * 2, args.n_heads)

        self.predict = nn.Linear(args.embedding_dimension * 2, 1)


    def forward(self, observation, action):

        # Prepare the state encodings
        observation_encoding = self.observation_embedding(observation.float())

        # Prepare the action encodings
        action_encoding = self.action_embedding(action.float())

        # Concatenate the state with each action
        x = torch.cat((observation_encoding, action_encoding),dim=2)

        # Attention
        query = self.q_projections(x.permute(0,2,1)).permute(2,0,1)
        key, value = self.kv_projection(x).permute(1,0,2).chunk(2,dim=2)
        x = self.attention(query, key, value)[0].permute(1,0,2)

        # Predict the next state
        x = self.predict(x)
        
        return x


class model_c(nn.Module):
    
    def __init__(self,args):
        super(model_c, self).__init__()
        self.args = args
        
        self.observation_embedding = nn.Linear(1 , args.embedding_dimension)
        self.action_embedding = nn.Linear(1, args.embedding_dimension)

        q_projections = {
            '42': nn.Linear(42, args.output_dimension),
            '35': nn.Linear(35, args.output_dimension),
            '28': nn.Linear(28, args.output_dimension),
            '21': nn.Linear(21, args.output_dimension),
            '14': nn.Linear(14, args.output_dimension),
            '7': nn.Linear(7, args.output_dimension),
        }
        self.q_projections = nn.ModuleDict(q_projections)
        self.kv_projection = nn.Linear(args.embedding_dimension, 2 * args.embedding_dimension)        

        self.attention = nn.MultiheadAttention(args.embedding_dimension, args.n_heads)

        self.predict = nn.Linear(args.embedding_dimension, 1)


    def forward(self, observation, action):
        n_agents = action.shape[1]
        batch_size = observation.shape[0]
        observation_encoding = self.observation_embedding(observation.reshape(batch_size,-1,1).float())
        action_encoding = self.action_embedding(action.reshape(batch_size,-1,1).float())
        
        x = torch.cat((observation_encoding, action_encoding),dim=1)
        sequence_length = x.shape[1]

        # Attention
        query = self.q_projections[str(sequence_length)](x.permute(0,2,1)).permute(2,0,1)
        key, value = self.kv_projection(x).permute(1,0,2).chunk(2,dim=2)
        x = self.attention(query, key, value)[0].permute(1,0,2)

        x = self.predict(x)
        
        return x[:,:n_agents*2,:]


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


class test(nn.Module):
    
    def __init__(self,args):
        super(test, self).__init__()
        self.args = args
        
        self.observation_embedding = nn.Linear(1 , args.embedding_dimension)
        self.action_embedding = nn.Linear(1, args.embedding_dimension)

        self.query = nn.Linear(42, args.output_dimension)
        self.key = nn.Linear(args.embedding_dimension, args.embedding_dimension)
        self.value = nn.Linear(args.embedding_dimension, args.embedding_dimension)

        self.attention = nn.Linear(args.embedding_dimension, args.output_dimension)
        self.predict = nn.Linear(args.embedding_dimension, 1)


    def forward(self, observation, action):
        batch_size = observation.shape[0]
        observation_encoding = self.observation_embedding(observation.reshape(batch_size,-1,1).float())
        action_encoding = self.action_embedding(action.reshape(batch_size,-1,1).float())
        
        x = torch.cat((observation_encoding, action_encoding),dim=1)

        query = self.query(x.permute(0,2,1)).permute(0,2,1)
        key = self.key(x).permute(0,2,1)
        value = self.value(x)

        weights = softmax(torch.bmm(query,key),dim=2)
        
        x = torch.bmm(weights, value)
        x = self.predict(x)
        
        return x